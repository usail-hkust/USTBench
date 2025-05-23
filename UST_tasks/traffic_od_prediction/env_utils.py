import numpy as np


def get_data_text(raw_data):
    data_text = (f"City: {raw_data['city']}\n\n"
                 f"Country: {raw_data['country']}\n\n"
                 f"{raw_data['target_region_data']}\n\n"
                 f"{raw_data['neighbor_region_data']}\n\n"
                 f"{raw_data['adj_info']}\n\n")

    return data_text

def get_env_change_text(predictions, answers):
    env_change_texts = []
    for i in range(len(predictions)):
        pred_inflow = np.array([v for v, t in predictions[i]['answer']['in-flow']])
        pred_outflow = np.array([v for v, t in predictions[i]['answer']['out-flow']])
        actual_inflow = np.array(answers[i]['inflow'])
        actual_outflow = np.array(answers[i]['outflow'])

        inflow_mae = np.mean(np.abs(pred_inflow - actual_inflow))
        inflow_mape = smape(pred_inflow, actual_inflow)
        outflow_mae = np.mean(np.abs(pred_outflow - actual_outflow))
        outflow_mape = smape(pred_inflow, actual_inflow)

        env_change_texts.append(
            f"Your in-flow prediction: {[[float(round(v, 2)), y] for v, y in predictions[i]['answer']['in-flow']]}.\n"
            f"Actual in-flow vehicles: {[[float(round(v, 2)), y] for v, y in zip(answers[i]['inflow'], answers[i]['next_times'])]}.\n"
            f"Mean Absolute Error on in-flow prediction: {round(inflow_mae, 2)}.\n"
            f"Mean Absolute Percentage Error on in-flow prediction: {round(inflow_mape * 100, 2)}%.\n\n"
            f"Your out-flow prediction: {[[float(round(v, 2)), y] for v, y in predictions[i]['answer']['out-flow']]}.\n"
            f"Actual out-flow vehicles: {[[float(round(v, 2)), y] for v, y in zip(answers[i]['outflow'], answers[i]['next_times'])]}.\n"
            f"Mean Absolute Error on out-flow prediction: {round(outflow_mae, 2)}.\n"
            f"Mean Absolute Percentage Error on out-flow prediction: {round(outflow_mape * 100, 2)}%.")

    return env_change_texts


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero_indices = y_true != 0
    y_true = y_true[nonzero_indices]
    y_pred = y_pred[nonzero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true))


def smape(y_pred, y_true):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-8)) * 100

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    return correct_count / len(y_true)
