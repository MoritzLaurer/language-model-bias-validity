

import pandas as pd
import os
import pickle
import gzip

dataset_lst = ["pimpo", "coronanet", "cap-merge", "cap-sotu"]
directory = "./results/"

data_dic_lst = []
# Loop over all files in the directory
for dataset in dataset_lst:
    directory_dataset = directory+dataset
    for filename in os.listdir(directory_dataset):
        # Check if the current file is a pickle file
        if filename.endswith('.pkl.gz'):
            # Construct the full file path
            file_path = os.path.join(directory_dataset, filename)
            # Open the file in read-binary mode
            with gzip.open(file_path, 'rb') as f:
                # Load the data from the file
                data_dic_lst.append(pickle.load(f))



## error correlation analysis tests
from scipy.stats import pearsonr, chi2_contingency
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


error_dic_lst = []
for data_dic in data_dic_lst:
    df_test = data_dic["df_test_results"]
    df_test["error"] = df_test["label_pred"] != df_test["label_gold_pred_aligned"]
    df_test['error_int'] = df_test['error'].astype(int)

    experiment_metadata_to_keep = [
        'dataset', 'group_sample_strategy', 'group_col', 'method', 'model_name', 'sample_size_train',
        'group_members', 'seed_run', 'n_run', 'date', 'train_time', 'model_size', 'task'
    ]
    experiment_metadata = {key: value for key, value in data_dic["experiment_metadata"].items() if key in experiment_metadata_to_keep}

    group_col = data_dic["experiment_metadata"]["group_col"]
    group_member = data_dic["experiment_metadata"]["group_members"]

    # Perform the chi-square test
    df_crosstab = pd.crosstab(df_test[group_col], df_test['error'])
    chi2, p, _, _ = chi2_contingency(df_crosstab)

    # logistic regression
    # one-hot encoding for group_member used for sampling/biasing
    if group_member != "randomall":
        try:
            onehot_group_member = [0 if x == group_member else 1 for x in df_test[group_col]]
            model = sm.Logit(df_test['error_int'], sm.add_constant(onehot_group_member)).fit()
            coefficients = model.params["x1"]
            p_values = model.pvalues["x1"]
        except Exception as e:
            print(f"\n\nIssue with {group_col} {group_member}. Error:\n{e}\n\n")
            continue
    else:
        coefficients = np.nan
        p_values = np.nan

    # append
    error_dic_lst.append({**experiment_metadata, "chi2": chi2, "p-chi2": p, "coef-reg": coefficients, "p-reg": p_values})

df_errors = pd.DataFrame(error_dic_lst)


# inspect results
df_errors_grouped = df_errors.groupby(["method", "group_col", "group_sample_strategy" ], as_index=False)[["chi2", "p-chi2", "coef-reg", "p-reg"]].mean().round(3)

df_errors_grouped_aggreg = df_errors.groupby(["method", "group_sample_strategy"], as_index=False)[["chi2", "p-chi2", "coef-reg", "p-reg"]].mean().round(3)

print(df_errors_grouped)
print(df_errors_grouped_aggreg)

