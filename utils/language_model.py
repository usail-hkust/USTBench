import copy
import threading
import time

import numpy as np
import requests
import vllm
import torch
import re
import regex
from tqdm import tqdm
import json
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.read_utils import load_json, markdown_code_pattern

class LLM(object):
    def __init__(self, llm_path, batch_size=16, top_k=50, top_p=1.0, temperature=0.1, max_tokens=8192, memory_size=3, task_info=None, use_reflection=True):
        self.use_reflection = use_reflection
        self.tokenizer, self.model, self.generation_kwargs, self.use_api = self.initialize_llm(llm_path, top_k, top_p, temperature, max_tokens)
        llm_name = llm_path.split("/")[-1]
        self.institute_name = llm_path.split("/")[-2]
        self.provider_name = llm_path.split("/")[0]
        self.llm_name = llm_name
        self.batch_size = batch_size
        self.task_info = task_info

        # memory initialization
        self.memory, self.memory_count, self.memory_size = self.initialize_memory(memory_size)

        # prompt template
        (self.system_prompt, self.overall_template, self.data_analysis_type_descriptions,
         self.data_analysis_type_selection_template, self.data_analysis_template, self.decision_making_template,
         self.self_reflection_template, self.memory_update_template) = self.initialize_prompt_template()

        # data analysis type initialization
        self.data_analysis_types = None

    def initialize_llm(self, llm_path, top_k, top_p, temperature, max_tokens):
        # init LLM
        use_api = False
        generation_kwargs = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "openai" in llm_path.lower() or "siliconflow" in llm_path.lower():
            llm_model = OpenAI()
            use_api = True
        else:
            llm_model = vllm.LLM(
                model=llm_path,
                gpu_memory_utilization=0.7,
                tensor_parallel_size=torch.cuda.device_count(),
                max_seq_len_to_capture=max_tokens,
                enforce_eager=True,
                trust_remote_code = True
            )
            generation_kwargs = vllm.SamplingParams(**generation_kwargs)

        return None, llm_model, generation_kwargs, use_api

    def initialize_prompt_template(self):
        system_prompt = load_json("./prompts/system_prompt.json")["template"]

        if not self.task_info:
            return system_prompt, None, None, None, None, None, None, None

        # Overall
        overall_template = load_json("./prompts/agent_prompt_template.json")["template"]
        overall_template = overall_template.replace("<task_description>", self.task_info["task_description"])
        overall_template = overall_template.replace("<data_schema>", self.task_info["data_schema"])
        overall_template = overall_template.replace("<domain_knowledge>", self.task_info["domain_knowledge"])

        # Analysis Type Descriptions
        data_analysis_type_descriptions = load_json("./prompts/data_analysis_type_descriptions.json")

        # Data analysis type selection
        data_analysis_type_selection_template = load_json("./prompts/data_analysis_type_selection_template.json")["template"]

        # Data analysis
        data_analysis_template = load_json("./prompts/data_analysis_template.json")["template"]

        # Decision-making
        decision_making_template = load_json("./prompts/decision_making_template.json")["template"]
        decision_making_template = decision_making_template.replace("<task_target>", self.task_info["task_target"])

        # self-reflection
        self_reflection_template = load_json("./prompts/self_reflection_template.json")["template"]
        self_reflection_template = self_reflection_template.replace("<task_target>", self.task_info["task_target"])
        self_reflection_template = self_reflection_template.replace("<task_output_type>", self.task_info["task_output_type"])

        # memory update
        memory_update_template = load_json("./prompts/memory_update_template.json")["template"]
        memory_update_template = memory_update_template.replace("<memory_num>", str(self.memory_size))

        return (system_prompt, overall_template, data_analysis_type_descriptions, data_analysis_type_selection_template,
                data_analysis_template, decision_making_template, self_reflection_template, memory_update_template)


    def initialize_data_analysis_types(self, data_analysis_types):
        self.data_analysis_types = data_analysis_types

    def initialize_memory(self, memory_size):
        memory = list()
        memory_count = 0

        return memory, memory_count, memory_size

    def update_memory(self, sample_info):
        if not self.use_reflection:
            return

        old_experience = ""
        for exp in self.memory:
            old_experience += f"- {exp}\n"
        old_experience = old_experience[:-1]

        new_experience = ""
        for s in sample_info:
            data_text, is_correct, experience = s
            new_experience += f"- {experience}\n"
        new_experience = new_experience[:-1]

        query = copy.copy(self.overall_template)

        # construct prompt
        query = query.replace("<data_text>", sample_info[0][0])
        query = query.replace("<step_instruction>", self.memory_update_template)
        query = query.replace("<memory_size>", str(self.memory_size))
        query = query.replace("<old_experience>", old_experience)
        query = query.replace("<new_experience>", new_experience)

        # replace memory
        retry_count = 0
        while retry_count < 3:
            try:
                response = self.inference(query)
                if response is None:
                    return

                possible_answer = regex.findall(markdown_code_pattern, response)
                if len(possible_answer) != 0:
                    self.memory = json.loads(possible_answer[-1])[:self.memory_size]
                else:
                    self.memory = json.loads(response)[:self.memory_size]

                return
            except Exception as e:
                print(f"Error in update memory: {e}\nTry again...")
                # print(f"=================================\n{response}")
                retry_count += 1

    def inference(self, query, system_prompt=None):
        message = [
            {
                "role": "system",
                "content": system_prompt if system_prompt is not None else self.system_prompt,
            },
            {
                "role": "user",
                "content": query
            }
        ]

        if self.use_api:
            if "deepseek" in self.llm_name.lower():
                llm_name = "deepseek-ai/" + self.llm_name
            else:
                llm_name = self.llm_name

            retry_count = 0
            response = None
            while retry_count < 3:
                try:
                    response = self.model.chat.completions.create(
                        model=llm_name,
                        messages=message,
                        temperature=self.generation_kwargs['temperature'],
                        max_tokens=self.generation_kwargs['max_tokens'],
                        # timeout=120
                    ).choices[0].message.content
                    break
                except:
                    time.sleep(5)
                    retry_count += 1
        else:
            responses_gen = self.model.chat([message], use_tqdm=False, sampling_params=self.generation_kwargs)
            response = responses_gen[0].outputs[0].text

        return response

    def batch_inference(self, queries, system_prompt=None):
        all_responses = []
        messages = list()

        for i, q in enumerate(queries):
            messages.append([
                {
                    "role": "system",
                    "content": system_prompt if system_prompt is not None else self.system_prompt,
                },
                {
                    "role": "user",
                    "content": q
                }
            ])

            if len(messages) == self.batch_size or i == len(queries) - 1:
                if self.use_api:
                    if self.provider_name == 'siliconflow':
                        llm_name = f"{self.institute_name}/{self.llm_name}"
                    else:
                        llm_name = self.llm_name
                    threads = []
                    responses = [None for _ in range(len(messages))]

                    for j, message in enumerate(messages):
                        thread = threading.Thread(target=self.threading_inference, args=(llm_name, message, responses, j, ))
                        threads.append(thread)
                        thread.start()

                    for thread in threads:
                        thread.join()

                    all_responses.extend(responses)
                else:
                    responses_gen = self.model.chat(messages, use_tqdm=False, sampling_params=self.generation_kwargs)
                    responses = [res.outputs[0].text for res in responses_gen]
                    all_responses.extend(responses)
                    messages = list()

        return all_responses

    def batch_evaluation(self, llm_name, queries, system_prompt=None):
        messages = list()

        for i, q in enumerate(queries):
            messages.append(([
                {
                    "role": "system",
                    "content": system_prompt if system_prompt is not None else self.system_prompt,
                },
                {
                    "role": "user",
                    "content": q['instruction']
                },
                {
                    "role": "assistant",
                    "content": q['response']
                },
                {
                    "role": "user",
                    "content": f"The correct answer is: {q['answer']}\n\n"
                               f"Please evaluate the response and give me a score from 0 to 10 within the XML tag like: <Score>7<Score>."
                }
            ], i))

        retry_count = 0
        valid_responses = [dict() for _ in range(len(messages))]
        while retry_count < 3:
            retry_messages = []
            threads = []
            eval_responses = []
            responses = [None for _ in range(len(messages))]

            for j, message in enumerate(messages):
                thread = threading.Thread(target=self.threading_inference, args=(llm_name, message[0], responses, j,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            eval_responses.extend(responses)
            for i, res in enumerate(eval_responses):
                score_pattern = r'<Score>(.*?)</Score>'
                scores = re.findall(score_pattern, res)
                if len(scores) >= 1:
                    score = scores[-1]
                    if int(score) < 6:
                        return None
                    else:
                        valid_responses[messages[i][1]] = {
                            "instruction": messages[i][0][1]['content'],
                            "response": messages[i][0][2]['content'],
                            "answer": messages[i][0][3]['content'],
                        }
                else:
                    retry_messages.append(messages[i])

            if len(retry_messages) == 0:
                break
            else:
                messages = retry_messages
                retry_count += 1

        return valid_responses

    def threading_inference(self, llm_name, message, response_list, m_id):
        retry_count = 0
        response_list[m_id] = ""
        while retry_count < 2:
            time.sleep(5)
            try:
                if "openai" == self.provider_name:
                    stream = self.model.chat.completions.create(
                        model=llm_name,
                        messages=message,
                        # temperature=self.generation_kwargs['temperature'],
                        max_completion_tokens=self.generation_kwargs['max_tokens'],
                        stream=True
                    )
                else:
                    stream = self.model.chat.completions.create(
                        model=llm_name,
                        messages=message,
                        max_tokens=self.generation_kwargs['max_tokens'] if "glm-z1-9b" not in llm_name.lower() else 8000,
                        stream=True
                    )

                collected_response = "<think>\n"
                reasoning_finish_flag = False
                for chunk in stream:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        if not reasoning_finish_flag:
                            collected_response += "</think>\n"
                            reasoning_finish_flag = True
                        token = chunk.choices[0].delta.content
                        collected_response += token
                    elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        token = chunk.choices[0].delta.reasoning_content
                        collected_response += token
                response_list[m_id] = collected_response
                print(f"\nSuccess [{m_id}].")
                break

            except Exception as e:
                retry_count += 1
                print(e)

    def data_analysis_type_selection(self, data_text):
        query = copy.copy(self.overall_template)

        query = query.replace("<data_text>", data_text)
        query = query.replace("<step_instruction>", self.data_analysis_type_selection_template)

        success_flag = False
        while not success_flag:
            responses = []
            try:
                responses = self.inference(query)

                possible_answer = regex.findall(markdown_code_pattern, responses)[-1]
                data_analysis_types = json.loads(possible_answer)
                self.initialize_data_analysis_types(data_analysis_types)

                return data_analysis_types

            except Exception as e:
                print(f"Error: {e}\nTrying again...")
                # print(f"=================================\n{responses}")

    def decision_making_pipeline(self, data_texts, data_analysis_types, answer_option_form):
        data_analysis_results = []
        for data_text in data_texts:
            # data analysis
            data_analysis_samples = list()
            for analysis_type in data_analysis_types:
                predefined_type = None
                for a_type in self.data_analysis_type_descriptions:
                    if analysis_type in a_type or a_type in analysis_type:
                        predefined_type = a_type
                analysis_description = self.data_analysis_type_descriptions[predefined_type] if predefined_type else ""
                analysis_reason = data_analysis_types[analysis_type]
                data_analysis_samples.append([
                    data_text,
                    analysis_type,
                    analysis_description,
                    analysis_reason
                ])

            data_analysis_sample_results = self.data_analysis(data_analysis_samples)
            data_analysis_text = ""
            for i, result in enumerate(data_analysis_sample_results):
                data_analysis_text += f"- {data_analysis_samples[i][1]}: {result['summary']}\n"
            data_analysis_results.append(data_analysis_text)

        # decision-making
        decision_making_samples = list()
        for i, data_text in enumerate(data_texts):
            decision_making_samples.append([
                data_text,
                data_analysis_results[i]
            ])
        decision_making_results = self.decision_making(decision_making_samples, answer_option_form)

        return decision_making_results

    def hybrid_decision_making_pipeline(self, data_texts, answer_option_form):
        # decision-making
        decision_making_samples = list()
        for i, data_text in enumerate(data_texts):
            decision_making_samples.append(data_text)
        decision_making_results = self.decision_making(decision_making_samples, answer_option_form)

        return decision_making_results

    def self_reflection_pipeline(self, data_texts, data_analyses, decisions, reasons, env_changes):
        # self-reflection
        self_reflection_samples = []
        for i, data_text in enumerate(data_texts):
            self_reflection_samples.append([data_texts[i], data_analyses[i],
                                            decisions[i], reasons[i],
                                            env_changes[i]])

        self_reflections = self.self_reflection(self_reflection_samples)

        # update memory
        memory_update_samples = []
        for i, self_reflection in enumerate(self_reflections):
            memory_update_samples.append([self_reflection["data_text"],
                                          self_reflection["is_correct"],
                                          self_reflection["experience"]])
        self.update_memory(memory_update_samples)

        return self_reflections

    def hybrid_self_reflection_pipeline(self, data_texts, decisions, reasons, env_changes, answer_option_form):
        # self-reflection
        self_reflection_samples = []
        for i, data_text in enumerate(data_texts):
            self_reflection_samples.append([data_texts[i], decisions[i],
                                            reasons[i], env_changes[i]])

        self_reflections = self.self_reflection(self_reflection_samples, answer_option_form)

        # update memory
        memory_update_samples = []
        for i, self_reflection in enumerate(self_reflections):
            memory_update_samples.append([self_reflection["data_text"],
                                          self_reflection["is_correct"],
                                          self_reflection["experience"]])
        self.update_memory(memory_update_samples)

        return self_reflections

    def data_analysis(self, sample_info):
        queries = list()

        for s in sample_info:
            data_text, analysis_type, analysis_description, analysis_reason = s
            query = copy.copy(self.overall_template)

            query = query.replace("<data_text>", data_text)
            query = query.replace("<step_instruction>", self.data_analysis_template)

            # data analysis template
            query = query.replace("<analysis_type>", analysis_type)
            query = query.replace("<analysis_type>", analysis_type)
            query = query.replace("<analysis_description>", analysis_description)
            query = query.replace("<analysis_reason>", analysis_reason)

            queries.append(query)

        retry_count = 0
        while retry_count < 3:
            unsuccessful_count = 0
            failed_responses = list()
            responses = self.batch_inference(queries)

            data_analysis_results = list()
            for res in responses:
                try:
                    possible_answer = regex.findall(markdown_code_pattern, res)[-1]
                    data_analysis = json.loads(possible_answer)
                except Exception as e:
                    unsuccessful_count += 1
                    data_analysis = {"summary": "N/A"}
                    failed_responses.append(res)

                if "summary" not in data_analysis:
                    unsuccessful_count += 1
                    data_analysis = {"summary": "N/A"}
                    failed_responses.append(res)

                data_analysis_results.append(data_analysis)

            if unsuccessful_count / len(queries) <= 0.2:
                return data_analysis_results
            else:
                retry_count += 1
                print(f"Error in data analysis: key missing\nTry again...")
                # print("=================================\n".join([res for res in failed_responses]))

        return [{"summary": "N/A"} for _ in range(len(queries))]

    def decision_making(self, sample_info, answer_option_form):
        queries = list()

        for i, s in enumerate(sample_info):
            query = copy.copy(self.overall_template)

            if len(s) == 2:
                # data analysis
                data_text, data_analysis = s
                query = query.replace("<data_analysis>", data_analysis)
            else:
                data_text = s
                query = query.replace("<data_analysis>", "N/A")

            query = query.replace("<data_text>", data_text)
            query = query.replace("<step_instruction>", self.decision_making_template)
            query = query.replace("<answer_option_form>", answer_option_form[i])

            # add memory
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
            query = query.replace("<experience>", memory_text)

            queries.append((i, query))

        retry_count = 0
        decision_making_results = [{
            "answer": None,
            "summary": "N/A",
            "data_text": sample_info[i][0],
            "data_analysis": sample_info[i][1] if len(sample_info[i]) == 2 else "N/A"}
            for i in range(len(queries))
        ]
        while retry_count < 3:
            retry_queries = []
            failed_responses = list()
            responses = self.batch_inference([q for _, q in queries])

            for i, res in enumerate(responses):
                ori_query_index = queries[i][0]

                # API Failure
                if res is None:
                    decision_making_results[ori_query_index].update({
                        "data_text": sample_info[ori_query_index][0] if len(sample_info[ori_query_index]) == 2 else sample_info[ori_query_index],
                        "data_analysis": sample_info[ori_query_index][1] if len(sample_info[ori_query_index]) == 2 else "N/A"
                    })
                    continue

                # Answer Failure
                try:
                    possible_answer = regex.findall(markdown_code_pattern, res)
                    if len(possible_answer) <= 0:
                        decision = json.loads(res)
                    else:
                        decision = json.loads(possible_answer[-1])
                except Exception as e:
                    decision = {}
                    failed_responses.append(res)

                if "answer" not in decision or "summary" not in decision:
                    retry_queries.append(queries[i])
                    decision = {}
                    failed_responses.append(res)

                if decision:
                    decision.update({
                        "data_text": sample_info[ori_query_index][0] if len(sample_info[ori_query_index]) == 2 else sample_info[ori_query_index],
                        "data_analysis": sample_info[ori_query_index][1] if len(sample_info[ori_query_index]) == 2 else "N/A"
                    })
                    decision_making_results[ori_query_index].update(decision)

            if retry_queries:
                retry_count += 1
                queries = retry_queries
                print(f"Error in decision-making: key missing\nTry again...")
                # print("=================================\n".join([res for res in failed_responses]))
            else:
                return decision_making_results

        return decision_making_results

    def self_reflection(self, sample_info, answer_option_form):
        queries = list()

        for i, s in enumerate(sample_info):
            query = copy.copy(self.overall_template)
            if len(s) == 5:
                # data analysis
                data_text, data_analysis, decision, reason, env_changes = s
                query = query.replace("<data_analysis>", data_analysis)
            else:
                data_text, decision, reason, env_changes = s
                query = query.replace("<data_analysis>", "N/A")

            query = query.replace("<data_text>", data_text)
            query = query.replace("<step_instruction>", self.self_reflection_template)
            query = query.replace("<answer_option_form>", answer_option_form[i])

            # add memory
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
            query = query.replace("<experience>", memory_text)

            # decision and reason
            query = query.replace("<decision_or_prediction>", str(decision))
            query = query.replace("<decision_or_prediction_summary>", str(reason))

            # env feedback
            query = query.replace("<env_changes>", env_changes)

            queries.append((i, query))

        retry_count = 0
        self_reflection_results = [{
            "is_correct": "YES",
            "answer": None,
            "experience": "N/A",
            "data_text": sample_info[i][0]}
            for i in range(len(queries))
        ]
        if not self.use_reflection:
            return self_reflection_results

        while retry_count < 3:
            retry_queries = list()
            failed_responses = list()
            responses = self.batch_inference([q for _, q in queries])
            for i, res in enumerate(responses):
                ori_query_index = queries[i][0]

                # API Failure
                if res is None:
                    self_reflection_results[ori_query_index].update({
                        "is_correct": "YES",
                        "data_text": sample_info[ori_query_index][0]
                    })
                    continue

                # Paser the response to extract the JSON object
                try:
                    possible_answer = regex.findall(markdown_code_pattern, res)
                    if len(possible_answer) <= 0:
                        reflection = json.loads(res)
                    else:
                        reflection = json.loads(possible_answer[-1])
                except Exception as e:
                    reflection = {}
                    failed_responses.append(res)

                if "is_correct" not in reflection or "answer" not in reflection or "experience" not in reflection:
                    retry_queries.append(queries[i])
                    reflection = {}
                    failed_responses.append(res)

                if reflection:
                    reflection.update({"data_text": sample_info[ori_query_index][0]})
                    self_reflection_results[ori_query_index].update(reflection)

            if retry_queries:
                retry_count += 1
                queries = retry_queries
                print(f"Error in self reflection: key missing\nTry again...")
                # print("=================================\n".join([res for res in failed_responses]))
            else:
                return self_reflection_results

        return self_reflection_results

    def evaluate(self, samples, task):
        answered_questions = copy.copy(samples)
        for s in answered_questions:
            s.update({"reasoning": None, "decision": None, "is_correct": False})
        queries = []
        spatial_temporal_results = {}
        correct_count = 0
        all_question_num = len(samples)

        for i, s in enumerate(tqdm(samples)):
            query = s["question"] if "prompt" not in s else f"{s['prompt']}\n\n{s['test_query']}"
            queries.append((len(queries), query))

            if (i + 1) % self.batch_size == 0 or i == len(samples) - 1:
                retry_count = 0
                batch_whole_query_num = len(queries)
                while retry_count < 3:
                    retry_queries = []
                    responses = self.batch_inference([q for _, q in queries])
                    for j, res in enumerate(responses):
                        ori_index = i+1-batch_whole_query_num+queries[j][0]
                        if res is None:
                            answered_questions[ori_index].update({
                                "reasoning": None,
                                "decision": None,
                                "is_correct": False
                            })
                            continue
                        if task == 'st_understanding':
                            answer_pattern = r'<Answer>(.*?)</Answer>'
                            possible_answers = re.findall(answer_pattern, res)
                            if len(possible_answers) > 0:
                                model_answer = possible_answers[-1]
                                answered_questions[ori_index].update({
                                    "reasoning": res,
                                    "decision": model_answer,
                                    "is_correct": True if model_answer == samples[ori_index]['answer'] else False
                                })
                            else:
                                retry_queries.append(queries[j])
                        else:
                            try:
                                possible_answers = regex.findall(markdown_code_pattern, res)
                                if len(possible_answers) <= 0:
                                    answer_dict = json.loads(res)
                                else:
                                    answer_dict = json.loads(possible_answers[-1])
                                answered_questions[ori_index].update({
                                    "reasoning": res,
                                    "summary": answer_dict["summary"],
                                    "decision": answer_dict['answer'],
                                    "is_correct": True if answer_dict['answer'] == samples[ori_index]['answer'] else False
                                })
                            except:
                                retry_queries.append(queries[j])

                    if retry_queries:
                        retry_count += 1
                        queries = retry_queries
                        print(f"Retrying {len(queries)} times...")
                    else:
                        break
                queries = []

        # Spatial-temporal relation results
        if task == "st_understanding":
            for sample in answered_questions:
                st_type = sample['spatial_temporal_relation']
                if st_type in spatial_temporal_results:
                    spatial_temporal_results[st_type]['num'] += 1
                    if sample['decision'] == sample['answer']:
                        spatial_temporal_results[st_type]['correct_num'] += 1
                        correct_count += 1
                else:
                    spatial_temporal_results[st_type] = {}
                    spatial_temporal_results[st_type]['num'] = 1
                    if sample['decision'] == sample['answer']:
                        spatial_temporal_results[st_type]['correct_num'] = 1
                        correct_count += 1
                    else:
                        spatial_temporal_results[st_type]['correct_num'] = 0

            for st_type in spatial_temporal_results:
                spatial_temporal_results[st_type]["accuracy"] = (spatial_temporal_results[st_type]['correct_num'] /
                                                                 spatial_temporal_results[st_type]['num'])
        else:
            for sample in answered_questions:
                correct_count += sample['is_correct']

        overall_accuracy = correct_count / all_question_num
        return answered_questions, overall_accuracy, spatial_temporal_results

