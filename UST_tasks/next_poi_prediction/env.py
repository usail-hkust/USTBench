import json
import os
import math
import numpy as np
from tqdm import tqdm
from utils.read_utils import load_json, dump_json
from .env_utils import get_data_text, get_env_change_text, calculate_metric


class HumanMobilityPredictionEnv:
    def __init__(self, data, candidate_dict):
        """
        Initialize the simulation environment.
        :param data: List of data points for congestion prediction.
        """
        self.data = data
        np.random.shuffle(self.data)

        self.candidate_dict = candidate_dict
        self.all_poi_ids = np.array(list(self.candidate_dict.keys()))

        self.history = []
        self.history_dir = "./UST_tasks/congestion_prediction/History"
        os.makedirs(self.history_dir, exist_ok=True)

    def run(self, llm_agent):
        """
        Run the congestion prediction using the provided LLM agent.
        :param llm_agent: The LLM agent used for prediction.
        :return: Mean Absolute Error, Mean Squared Error, Mean Absolute Percentage Error, and Accuracy.
        """
        history_path = os.path.join(self.history_dir, f"{llm_agent.llm_name}.json")

        batch_data = {"texts": [], "answers": [], "recents": [], "actuals": []}
        overall_results = {"p@10": [], "mrr@10": [], "ndcg@10": []}
        batch_candidates = []

        for i, data_point in enumerate(tqdm(self.data)):
            # prepare candidates
            ground_truth = np.int64(data_point["answer"][2])
            negative_samples = np.random.choice(self.all_poi_ids, 49, replace=False)
            negative_sample_set = set(negative_samples)

            while ground_truth in negative_sample_set:
                ground_truth_index = np.where(negative_samples == ground_truth)[0][0]
                remaining_ids = np.setdiff1d(self.all_poi_ids, negative_samples)
                new_sample = np.random.choice(remaining_ids, 1, replace=False)
                negative_samples[ground_truth_index] = new_sample
                negative_sample_set = set(negative_samples)
            candidates = [ground_truth] + list(negative_samples.astype(int))
            np.random.shuffle(candidates)

            # get data text
            data_text = get_data_text(data_point, candidates, self.candidate_dict)
            batch_data["texts"].append(data_text)
            batch_data["answers"].append(data_point["answer"])
            batch_data["recents"].append(data_point['recent'])
            batch_data["actuals"].append(data_point["answer"])
            batch_candidates.append(candidates)

            if len(batch_data["texts"]) == llm_agent.batch_size or i == len(self.data) - 1:
                predictions = self._process_batch(batch_data, batch_candidates, llm_agent)
                self._update_results(predictions, batch_data, overall_results)
                self._log_history(history_path, batch_data, batch_candidates, predictions, llm_agent)

                batch_data = {"texts": [], "answers": [], "recents": [], "actuals": []}
                batch_candidates = []

        return self._calculate_overall_metrics(overall_results)

    def _process_batch(self, batch_data, candidates, llm_agent):
        """
        Process a batch of data points for predictions.
        :param batch_data: Batch data containing texts, answers, recents, and actuals.
        :param llm_agent: The LLM agent used for predictions.
        :param candidates: List of candidate POI IDs.
        :return: Predictions for the batch.
        """
        answer_forms = self._create_answer_forms(candidates)
        predictions = self._make_decisions(batch_data["texts"], answer_forms, llm_agent)
        return predictions

    def _create_answer_forms(self, batch_candidates):
        """
        Create answer forms based on the provided answers.
        :param batch_candidates: List of candidate POI IDs.
        :return: List of generated answer forms.
        """
        forms = [json.dumps([int(can) for can in candidates[:10]], indent=4) for candidates in batch_candidates]
        return forms

    def _make_decisions(self, texts, answer_forms, llm_agent):
        """
        Make decisions using the LLM agent's decision-making pipeline.
        :param texts: List of data texts.
        :param answer_forms: List of answer forms.
        :param llm_agent: The LLM agent used for decisions.
        :return: List of predictions.
        """
        predictions = [{"answer": json.loads(candidates),
                        "data_analysis": "N/A", "summary": "N/A"} for candidates in answer_forms]

        predictions_ = llm_agent.hybrid_decision_making_pipeline(texts, answer_forms)
        for i, pre in enumerate(predictions_):
            if pre['answer'] and isinstance(pre['answer'], list):
                predictions[i] = pre

        return predictions

    def _update_results(self, predictions, batch_data, overall_results):
        """
        Update overall results based on the predictions and actual data.
        :param predictions: Predictions made by the LLM agent.
        :param batch_data: Batch data containing actuals.
        :param overall_results: Dictionary to store overall results.
        """
        overall_results['p@10'].extend(calculate_metric([pre['answer'] for pre in predictions], [act[2] for act in batch_data["actuals"]] , 'p@k'))
        overall_results['mrr@10'].extend(calculate_metric([pre['answer'] for pre in predictions], [act[2] for act in batch_data["actuals"]], 'mrr@k'))
        overall_results['ndcg@10'].extend(calculate_metric([pre['answer'] for pre in predictions], [act[2] for act in batch_data["actuals"]], 'ndcg@k'))

    def _log_history(self, history_path, batch_data, batch_candidates, predictions, llm_agent):
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
        answer_forms = self._create_answer_forms(batch_candidates)
        self_reflections = llm_agent.hybrid_self_reflection_pipeline(
            batch_data["texts"], [pred['answer'] for pred in predictions], prediction_summaries, env_change_texts, answer_forms
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
        :return: Tuple containing overall P@10, MRR@10, and NDCG@10.
        """
        p_at_10 = np.mean(overall_results['p@10'])
        mrr_at_10 = np.mean(overall_results['mrr@10'])
        ndcg_at_10 = np.mean(overall_results['ndcg@10'])
        return p_at_10, mrr_at_10, ndcg_at_10
