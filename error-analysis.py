

from scipy.stats import pearsonr, chi2_contingency
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
import pickle
import gzip


dataset_lst = ["pimpo", "coronanet", "cap-merge", "cap-sotu"]
directory = "./results/"

data_dic_lst = []
# Load data: loop over all files in the directory
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



## statistical error analysis tests
# currently running separate statistical tests on each run.
# one run (= one dictionary) are the results from one model trained on one group tested on the held-out testset.
error_dic_lst = []
for data_dic in data_dic_lst:

    # extract test data with predictions for specific run
    df_test = data_dic["df_test_results"]
    df_test["error"] = df_test["label_pred"] != df_test["label_gold_pred_aligned"]
    df_test['error_int'] = df_test['error'].astype(int)

    # extract some metadata information on the specific run
    experiment_metadata_to_keep = [
        'dataset', 'group_sample_strategy', 'group_col', 'method', 'model_name', 'sample_size_train',
        'group_members', 'seed_run', 'n_run', 'date', 'train_time', 'model_size', 'task'
    ]
    experiment_metadata = {key: value for key, value in data_dic["experiment_metadata"].items() if key in experiment_metadata_to_keep}

    # extract the group and specific group-member the respective model was trained on
    # if the specific run was on unbiased/random data, group_member == "randomall".
    group_col = data_dic["experiment_metadata"]["group_col"]
    group_member = str(data_dic["experiment_metadata"]["group_members"])

    # Perform the chi-square test
    df_crosstab = pd.crosstab(df_test[group_col], df_test['error'])
    chi2, p, _, _ = chi2_contingency(df_crosstab)

    # logistic regression
    # one-hot encoding for group_member used for sampling/biasing
    # "randomall" are the scenarios where the training data is fully random and not biased
    if group_member != "randomall":
            onehot_group_member = [1 if str(x) == group_member else 0 for x in df_test[group_col]]
            model = sm.Logit(df_test['error_int'], sm.add_constant(onehot_group_member)).fit()
            coefficients = model.params["x1"]
            p_values = model.pvalues["x1"]
    else:
        coefficients = np.nan
        p_values = np.nan

    # append
    error_dic_lst.append({**experiment_metadata, "chi2": chi2, "p-chi2": p, "coef-reg": coefficients, "oddsratio-reg": np.exp(coefficients), "p-reg": p_values})

df_errors = pd.DataFrame(error_dic_lst)


## some cleaning / post-proccessing
# for better wording / formatting in paper
method_map = {
    "classical_ml": "logistic reg.", "standard_dl": "BERT-base", "nli_short": "BERT-NLI"
}
df_errors["method"] = df_errors["method"].map(method_map)
group_map = {
    'parfam_text': "party fam", 'country_iso': "country", 'decade': "decade",
    'year': "year", 'continent': "continent", 'ISO_A3': "country",
    'domain': "domain", 'pres_party': "party", 'phase': "phase"
}
df_errors["group_col"] = df_errors["group_col"].map(group_map)
dataset_map = {
    'pimpo': "PImPo", 'coronanet': "CoronaNet", 'cap-merge': "CAP-SotU+Court", 'cap-sotu': "CAP-SotU"
}
df_errors["dataset"] = df_errors["dataset"].map(dataset_map)
df_errors = df_errors.rename(columns={"group_col": "group"})


# sort values
df_errors.loc[:,'method'] = pd.Categorical(
    df_errors['method'], categories=["logistic reg.", "BERT-base", "BERT-NLI"], ordered=True
)
df_errors.loc[:,'dataset'] = pd.Categorical(
    df_errors['dataset'], categories=['PImPo', 'CoronaNet', 'CAP-SotU', 'CAP-SotU+Court'], ordered=True
)

# remove nli_void/meaningless for now
df_errors = df_errors[df_errors["method"] != "nli_void"]

# remove randomall, because regression not properly possible on randomall and already doing randomall vs. biased comparison in absolute f1 macro analysis
df_errors = df_errors[df_errors["group_sample_strategy"] != "randomall"]


## results disaggregated
# inspect results
df_errors_grouped = df_errors.groupby(["method", "group", "group_sample_strategy"], as_index=False)[["chi2", "p-chi2", "coef-reg", "oddsratio-reg", "p-reg"]].mean().round(2)
# add dataset column again
group_dataset_map = dict(zip(df_errors['group'], df_errors['dataset']))
df_errors_grouped.insert(0, "dataset", df_errors_grouped["group"].map(group_dataset_map))
# move method column
col_to_move = df_errors_grouped.pop('method')
df_errors_grouped.insert(2, 'method', col_to_move)
# sort
df_errors_grouped = df_errors_grouped.sort_values(["dataset", "group", "method"])
df_errors_grouped["chi2"] = df_errors_grouped["chi2"].round(1)

## results aggregated
df_errors_grouped_aggreg = df_errors.groupby(["method", "group_sample_strategy"], as_index=False)[["chi2", "p-chi2", "coef-reg", "oddsratio-reg", "p-reg"]].mean().round(2)
df_errors_grouped_aggreg["chi2"] = df_errors_grouped_aggreg["chi2"].round(1)

print(df_errors_grouped)
print(df_errors_grouped_aggreg)



# save to disk
save_to_disk = True
if save_to_disk:
    df_errors_grouped.to_excel("./viz/error-analysis-disaggreg.xlsx", index=False)
    df_errors_grouped_aggreg.to_excel("./viz/error-analysis-aggreg.xlsx", index=False)


### example interpretation
# GPT4 conversation: https://chat.openai.com/share/c6920a60-1342-4afe-a893-49a383aa97e6
## coefficients:
# coefficients are negative. this means that for the favoured group in biased df_train, the odds of making an error are lower than for the other groups not in df_train.
# the coefficients are more negative the less transfer learning. this means that less transfer learning means more bias.

## odds ratio: e.g. 0.81 for nli, 0.7 for classical_ml.
# For NLI this means: The odds of an error occurring when the group is the biased group are 0.81 times the odds when the group is not the biased group.
# More intuitively: The odds of an error occurring when the group is the biased group are 19% lower than when the group is not the biased group.
# for classical_ml: The odds of an error occurring when the group is the biased group are 30% lower than when the group is not the biased group.
# the odds of making an error would be equal if odds ratio == 1 or higher if odds ratio > 1.
# => both are biased in favour of the df_train group. classical_ml is even more biased. It is less likely to make an error on the biased group than NLI.

