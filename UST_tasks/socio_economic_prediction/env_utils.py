import numpy as np


def get_data_text(raw_data):
    data_text = (f"City: {raw_data['city']}\n\n"
                 f"{raw_data['target_region_data']}\n\n"
                 f"{raw_data['nearby_region_data']}\n\n"
                 f"{raw_data['adjacency_info']}\n\n"
                 f"{raw_data['example_region_data']}\n\n")

    return data_text


def get_env_change_text(predictions, answers):
    env_change_texts = []
    for i in range(len(predictions)):
        pred_gdp = np.array([v for v, y in predictions[i]['answer']['gdp']])
        pred_pop = np.array([v for v, y in predictions[i]['answer']['population']])
        actual_gdp = np.array(answers[i]['gdp'])
        actual_pop = np.array(answers[i]['population'])

        gdp_mae = np.mean(np.abs(pred_gdp - actual_gdp))
        gdp_mape = np.mean(np.abs(pred_gdp - actual_gdp) / actual_gdp)
        pop_mae = np.mean(np.abs(pred_pop - actual_pop))
        pop_mape = np.mean(np.abs(pred_pop - actual_pop) / actual_pop)

        env_change_texts.append(
            f"Your GDP prediction: {[[float(round(v, 2)), y] for v, y in predictions[i]['answer']['gdp']]}.\n"
            f"Actual GDP: {[[float(round(v, 2)), y] for v, y in zip(answers[i]['gdp'], answers[i]['next_years'])]}.\n"
            f"Mean Absolute Error on GDP: {round(gdp_mae, 2)}.\n"
            f"Mean Absolute Percentage Error on GDP: {round(gdp_mape * 100, 2)}%.\n\n"
            f"Your population prediction: {[[float(round(v, 2)), y] for v, y in predictions[i]['answer']['population']]}.\n"
            f"Actual population: {[[float(round(v, 2)), y] for v, y in zip(answers[i]['population'], answers[i]['next_years'])]}.\n"
            f"Mean Absolute Error on population: {round(pop_mae, 2)}.\n"
            f"Mean Absolute Percentage Error on population: {round(pop_mape * 100, 2)}%.")

    return env_change_texts


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero_indices = y_true != 0
    y_true = y_true[nonzero_indices]
    y_pred = y_pred[nonzero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true))


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct_count += 1

    return correct_count / len(y_true)
