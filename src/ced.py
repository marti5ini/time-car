"""
Implementation to estimate the derivative of causal effect
"""
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_causal_effect_derivative(data, proper_std, scm, n_samples=1000, inc=1.2, karimi=False):
    """
    Calculates the derivative of the causal effect for each variable in the data.

    :param data: Input data as a pandas DataFrame.
    :param proper_std: Dictionary of standard deviations for each variable.
    :param scm: Structural Causal Model object.
    :param n_samples: Number of samples (default: 1000).
    :param inc: Increment value for perturbing the variables (default: 1.2).
    :return: Tuple of results (dictionary of causal effect derivatives) and df_do (perturbed data).
    """

    scm_do = {}  # Dictionary to store the do operation for each variable
    df_do = {}  # Dictionary to store the perturbed data for each variable

    results = {}  # Dictionary to store the results (causal effect derivatives) for each variable

    # Iterate over each variable in the data
    for index in data.columns:
        # Perform do operation on the variable using the SCM
        scm_do[index] = scm.do(index)

        if karimi:
            df_do[index] = scm_do[index].sample(n_samples=n_samples,
                                                set_values={index: data[index] + inc * (data[index].std())})
        else:
            increase = inc / np.mean(list(proper_std.values()))  # Calculate the increase factor

            # Perturb the variable by adding an increased value
            df_do[index] = scm_do[index].sample(n_samples=n_samples,
                                                set_values={index: data[index] + increase * proper_std[index]})

            # Filter out rows with outliers from the perturbed data
            df_do[index] = df_do[index][(np.abs(stats.zscore(df_do[index])) < 3).all(axis=1)]

        # Assign the Intervention column with the variable name
        df_do[index] = df_do[index].assign(Intervention=index)

        # Calculate the causal effect derivative as the difference in means divided by the increment
        results[index] = round(((df_do[index]["Y"].mean() - data["Y"].mean()) / inc), 3)

    return results, df_do


def get_interventional_data(df, df_do):
    # Filter the rows of the dataframe df to remove outliers
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    # Concatenate the "I" and "E" columns with the dataframe df
    df = pd.concat([df_do["I"], df_do["E"], df], ignore_index=True)

    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit and transform the dataframe
    normalized = scaler.fit_transform(df.iloc[:, :-1])

    df_normalized = pd.DataFrame(normalized, columns=list(df.columns)[:-1])

    # Convert the "Intervention" column to string type
    df_normalized["Intervention"] = df["Intervention"].astype(str)

    # Reorder the columns of the dataframe df
    df = df_normalized[['A', 'E', 'I', 'L', 'Y', 'Intervention']]

    df = df.replace({'Intervention': 'nan'}, 'O')

    return df
