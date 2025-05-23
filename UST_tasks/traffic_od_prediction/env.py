import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from utils.read_utils import load_json, dump_json
from .env_utils import get_data_text, get_env_change_text, mean_absolute_percentage_error, smape, accuracy


class TrafficFlowPredictionEnv:
    def __init__(self, data):
        """
        Initialize the simulation environment.
        :param data: List of data points for congestion prediction.
        """
        self.data = data
        np.random.shuffle(self.data)
        self.history = []
        self.history_dir = "./UST_tasks/traffic_flow_prediction/History"
        os.makedirs(self.history_dir, exist_ok=True)

    def run(self, llm_agent):
        """
        Run the congestion prediction using the provided LLM agent.
        :param llm_agent: The LLM agent used for prediction.
        :return: Mean Absolute Error, Mean Squared Error, Mean Absolute Percentage Error, and Accuracy.
        """
        history_path = os.path.join(self.history_dir, f"{llm_agent.llm_name}.json")

        batch_data = {"texts": [], "answers": []}
        overall_results = {"inflow_mae": [], "inflow_mse": [], "inflow_mape": [],
                           "outflow_mae": [], "outflow_mse": [], "outflow_mape": []}

        for i, data_point in enumerate(tqdm(self.data)):
            data_text = get_data_text(data_point)
            batch_data["texts"].append(data_text)
            batch_data["answers"].append({"inflow": data_point["next_inflow"], "outflow": data_point["next_outflow"],
                                          "next_times": data_point["next_times"]})

            if len(batch_data["texts"]) == llm_agent.batch_size or i == len(self.data) - 1:
                predictions = self._process_batch(batch_data, llm_agent)
                self._update_results(predictions, batch_data, overall_results)
                self._log_history(history_path, batch_data, predictions, llm_agent)

                batch_data = {"texts": [], "answers": []}

        return self._calculate_overall_metrics(overall_results)

    def _process_batch(self, batch_data, llm_agent):
        """
        Process a batch of data points for predictions.
        :param batch_data: Batch data containing texts, answers, recents, and actuals.
        :param llm_agent: The LLM agent used for predictions.
        :return: Predictions for the batch.
        """
        answer_forms = self._create_answer_forms(batch_data["answers"])
        predictions = self._make_decisions(batch_data["texts"], answer_forms, llm_agent)
        return predictions

    def _create_answer_forms(self, answers):
        """
        Create answer forms based on the provided answers.
        :param answers: List of answers.
        :return: List of generated answer forms.
        """
        answer_forms = []
        for i in range(len(answers)):
            sample_inflow = np.random.randint(low=10, high=400, size=len(answers[i]['next_times']))
            sample_outflow = np.random.randint(low=10, high=400, size=len(answers[i]['next_times']))
            answer_forms.append(json.dumps({"in-flow": [[round(float(sample_inflow[j]), 2), t] for j, t in enumerate(answers[i]['next_times'])],
                                            "out-flow": [[round(float(sample_outflow[j]), 2), t] for j, t in enumerate(answers[i]['next_times'])]}, indent=4))
        return answer_forms

    def _make_decisions(self, texts, answer_forms, llm_agent):
        """
        Make decisions using the LLM agent's decision-making pipeline.
        :param texts: List of data texts.
        :param answer_forms: List of answer forms.
        :param llm_agent: The LLM agent used for decisions.
        :return: List of predictions.
        """
        predictions = [{"answer": json.loads(ans), "data_analysis": "N/A", "summary": "N/A"} for ans in answer_forms]
        texts = [[t, i] for i, t in enumerate(texts)]
        answer_forms = [[a, i] for i, a in enumerate(answer_forms)]
        retry_count = 0

        while retry_count < 3:
            retry_flag = False
            retry_texts = []
            retry_answer_forms = []
            predictions_ = llm_agent.hybrid_decision_making_pipeline([t[0] for t in texts], [a[0] for a in answer_forms])

            for i, pre_ in enumerate(predictions_):
                ori_idx = texts[i][1]
                if pre_['answer'] is None or "in-flow" not in pre_['answer'] or "out-flow" not in pre_['answer']:
                    print("Error in predictions: missing values.\nRetrying...")
                    retry_flag = True
                    retry_texts.append([texts[i][0], texts[i][1]])
                    retry_answer_forms.append([answer_forms[i][0], answer_forms[i][1]])
                    continue
                if (not hasattr(pre_['answer']['in-flow'], '__len__') or len(pre_['answer']['in-flow']) != 12 or
                    not hasattr(pre_['answer']['out-flow'], '__len__') or len(pre_['answer']['out-flow']) != 12):
                    print("Error in predictions: wrong length.\nRetrying...")
                    retry_flag = True
                    retry_texts.append([texts[i][0], texts[i][1]])
                    retry_answer_forms.append([answer_forms[i][0], answer_forms[i][1]])
                    continue
                if (
                    any(not hasattr(vy, '__len__') or len(vy) != 2 or not isinstance(vy, list) or isinstance(vy[0], str) for vy in pre_['answer']['in-flow']) or
                    any(not hasattr(vy, '__len__') or len(vy) != 2 or not isinstance(vy, list) or isinstance(vy[0], str) for vy in pre_['answer']['out-flow'])
                ):
                    print("Error in predictions: wrong type.\nRetrying...")
                    retry_flag = True
                    retry_texts.append([texts[i][0], texts[i][1]])
                    retry_answer_forms.append([answer_forms[i][0], answer_forms[i][1]])
                    continue
                predictions[ori_idx] = pre_

            if not retry_flag:
                return predictions
            else:
                texts = retry_texts
                answer_forms = retry_answer_forms
                retry_count += 1

        return predictions

    def _update_results(self, predictions, batch_data, overall_results):
        """
        Update overall results based on the predictions and actual data.
        :param predictions: Predictions made by the LLM agent.
        :param batch_data: Batch data containing actuals.
        :param overall_results: Dictionary to store overall results.
        """
        for pred, actual in zip(predictions, batch_data["answers"]):
            overall_results['inflow_mae'].append(mean_absolute_error([v for v, y in pred['answer']['in-flow']], actual['inflow']))
            overall_results['inflow_mse'].append(mean_squared_error([v for v, y in pred['answer']['in-flow']], actual['inflow']))
            overall_results['inflow_mape'].append(smape([v for v, y in pred['answer']['in-flow']], actual['inflow']))
            overall_results['outflow_mae'].append(mean_absolute_error([v for v, y in pred['answer']['out-flow']], actual['outflow']))
            overall_results['outflow_mse'].append(mean_squared_error([v for v, y in pred['answer']['out-flow']], actual['outflow']))
            overall_results['outflow_mape'].append(smape([v for v, y in pred['answer']['out-flow']], actual['outflow']))

    def _log_history(self, history_path, batch_data, predictions, llm_agent):
        """
        Log the history of predictions and analysis.
        :param history_path: Path to save the history log.
        :param batch_data: Batch data.
        :param predictions: Predictions made by the LLM agent.
        :param llm_agent: The LLM agent used for predictions.
        """
        data_analysis = [pred['data_analysis'] for pred in predictions]
        prediction_summaries = [pred['summary'] for pred in predictions]
        env_change_texts = get_env_change_text(predictions, batch_data["answers"])
        answer_forms = self._create_answer_forms(batch_data["answers"])
        self_reflections = llm_agent.hybrid_self_reflection_pipeline(
            batch_data["texts"], [pred['answer'] for pred in predictions],
            prediction_summaries, env_change_texts, answer_forms
        )

        self.history.append({
            "data_texts": batch_data["texts"],
            "decisions": [pred['answer'] for pred in predictions],
            "data_analysis": data_analysis,
            "decision_summary": prediction_summaries,
            "env_change_texts": env_change_texts,
            "self_reflections": self_reflections if self_reflections else "N/A",
            "memory": llm_agent.memory
        })

        dump_json(self.history, history_path)

    def _calculate_overall_metrics(self, overall_results):
        """
        Calculate overall metrics from the results.
        :param overall_results: Dictionary of overall results.
        :return: Tuple containing overall MAE, MSE, MAPE, and Accuracy.
        """
        inflow_mae = np.mean(overall_results['inflow_mae'])
        inflow_mse = np.mean(overall_results['inflow_mse'])
        inflow_mape = np.mean(overall_results['inflow_mape'])
        outflow_mae = np.mean(overall_results['outflow_mae'])
        outflow_mse = np.mean(overall_results['outflow_mse'])
        outflow_mape = np.mean(overall_results['outflow_mape'])

        return inflow_mae, inflow_mse, inflow_mape, outflow_mae, outflow_mse, outflow_mape
