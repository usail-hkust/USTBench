import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from utils.read_utils import load_json, dump_json
from .env_utils import get_data_text, get_env_change_text, mean_absolute_percentage_error, accuracy


class UrbanDevelopmentPredictionEnv:
    def __init__(self, data):
        """
        Initialize the simulation environment.
        :param data: List of data points for congestion prediction.
        """
        self.data = data
        np.random.shuffle(self.data)
        self.history = []
        self.history_dir = "./UST_tasks/urban_development_prediction/History"
        os.makedirs(self.history_dir, exist_ok=True)

    def run(self, llm_agent):
        """
        Run the congestion prediction using the provided LLM agent.
        :param llm_agent: The LLM agent used for prediction.
        :return: Mean Absolute Error, Mean Squared Error, Mean Absolute Percentage Error, and Accuracy.
        """
        history_path = os.path.join(self.history_dir, f"{llm_agent.llm_name}.json")

        batch_data = {"texts": [], "answers": []}
        overall_results = {"gdp_mae": [], "gdp_mse": [], "gdp_mape": [],
                           "pop_mae": [], "pop_mse": [], "pop_mape": []}

        for i, data_point in enumerate(tqdm(self.data)):
            data_text = get_data_text(data_point)
            batch_data["texts"].append(data_text)
            batch_data["answers"].append({"gdp": data_point["next_gdp"], "population": data_point["next_pop"],
                                          "next_years": data_point["next_years"]})

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
            sample_gdps = np.random.uniform(low=6.14, high=2263.34, size=len(answers[i]['next_years'])).tolist()
            sample_pops = np.random.uniform(low=390.68, high=1158506.28, size=len(answers[i]['next_years'])).tolist()
            answer_forms.append(json.dumps({
                "gdp": [[round(sample_gdps[j], 2), y] for j, y in enumerate(answers[i]['next_years'])],
                "population": [[round(sample_pops[j], 2), y] for j, y in enumerate(answers[i]['next_years'])]
            }, indent=4))
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
                if pre_['answer'] is None or "gdp" not in pre_['answer'] or "population" not in pre_['answer']:
                    print("Error in predictions: missing values.\nRetrying...")
                    retry_flag = True
                    retry_texts.append([texts[i][0], texts[i][1]])
                    retry_answer_forms.append([answer_forms[i][0], answer_forms[i][1]])
                    continue
                if (
                    not hasattr(pre_['answer']['gdp'], '__len__') or len(pre_['answer']['gdp']) != 3 or
                    not hasattr(pre_['answer']['population'], '__len__') or len(pre_['answer']['population']) != 3
                ):
                    print("Error in predictions: wrong length.\nRetrying...")
                    retry_flag = True
                    retry_texts.append([texts[i][0], texts[i][1]])
                    retry_answer_forms.append([answer_forms[i][0], answer_forms[i][1]])
                    continue
                if (
                    any(not hasattr(vy, '__len__') or len(vy) != 2 or not isinstance(vy, list) or isinstance(vy[0], str) for vy in pre_['answer']['gdp']) or
                    any(not hasattr(vy, '__len__') or len(vy) != 2 or not isinstance(vy, list) or isinstance(vy[0], str) for vy in pre_['answer']['population'])
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
            overall_results['gdp_mae'].append(mean_absolute_error([v for v, y in pred['answer']['gdp']], actual['gdp']))
            overall_results['gdp_mse'].append(mean_squared_error([v for v, y in pred['answer']['gdp']], actual['gdp']))
            overall_results['gdp_mape'].append(mean_absolute_percentage_error([v for v, y in pred['answer']['gdp']], actual['gdp']))
            overall_results['pop_mae'].append(mean_absolute_error([v for v, y in pred['answer']['population']], actual['population']))
            overall_results['pop_mse'].append(mean_squared_error([v for v, y in pred['answer']['population']], actual['population']))
            overall_results['pop_mape'].append(mean_absolute_percentage_error([v for v, y in pred['answer']['population']], actual['population']))

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
        gdp_mae = np.mean(overall_results['gdp_mae'])
        gdp_mse = np.mean(overall_results['gdp_mse'])
        gdp_mape = np.mean(overall_results['gdp_mape'])
        pop_mae = np.mean(overall_results['pop_mae'])
        pop_mse = np.mean(overall_results['pop_mse'])
        pop_mape = np.mean(overall_results['pop_mape'])

        return gdp_mae, gdp_mse, gdp_mape, pop_mae, pop_mse, pop_mape
