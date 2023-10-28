# recalculate classification metrics depending on balanced/downsampled df_test or not

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score

SAVE_RESULTS = True
USE_DOWNSAMPLED_DATA = False


# load data
if USE_DOWNSAMPLED_DATA:
    df_test = pd.read_parquet("./results/df_test_concat_downsampled.parquet.gzip")
else:
    df_test = pd.read_parquet("./results/df_test_concat.parquet.gzip")


# function to recalculate metrics with more balanced test sets
def compute_metrics(label_pred, label_gold):
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_pred, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_pred, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_gold, label_pred)
    acc_not_balanced = accuracy_score(label_gold, label_pred)

    metrics = {
        'eval_f1_macro': f1_macro,
        'eval_f1_micro': f1_micro,
        'eval_accuracy_balanced': acc_balanced,
        'eval_accuracy_not_b': acc_not_balanced,
        'eval_precision_macro': precision_macro,
        'eval_recall_macro': recall_macro,
        'eval_precision_micro': precision_micro,
        'eval_recall_micro': recall_micro,
    }
    return metrics


# creating one group per experiment
# while there are many columns to group by, they always have the same value, so it's only one groupby per experiment
# doing this to maintain metadata info
df_test_grouped = df_test.groupby([
    'file_name', 'dataset', 'group_sample_strategy', 'group_col', 'method',  # "group_members_test",
    'model_name', 'sample_size_train', 'group_members_train', 'seed_run', 'n_run',
    'train_time_sec', 'model_size', 'task', 'data_train_biased'
])

# Applying the function to get a Series of dictionaries
df_test_results = df_test_grouped.apply(
    lambda x: compute_metrics(x["label_pred"], x["label_gold"])
)
# Convert the Series to DataFrame
df_test_results = df_test_results.apply(pd.Series)
# Reset index to make groupby columns part of the dataframe
df_test_results = df_test_results.reset_index()



## some renaming and cleaning
df_test_results.drop(columns=[
    'eval_accuracy_not_b', 'eval_precision_macro', 'eval_recall_macro', 'eval_precision_micro', 'eval_recall_micro',
    #'eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch',
    "seed_run", "n_run", "train_time_sec"
    ], inplace=True
)
# renaming some values / columns for paper
method_map = {
    "classical_ml": "logistic reg.", "standard_dl": "BERT-base", "nli_short": "BERT-NLI", "nli_void": "BERT-NLI-void",
}
df_test_results["method"] = df_test_results["method"].map(method_map)
group_map = {
    'parfam_text': "party_fam", 'country_iso': "country", 'decade': "decade",
    'year': "year", 'continent': "continent", 'ISO_A3': "countries_3",
    'domain': "domain", 'pres_party': "party", 'phase': "phase", "randomall": "randomall"
}
df_test_results["group_col"] = df_test_results["group_col"].map(group_map)
dataset_map = {
    'pimpo': "PImPo", 'coronanet': "CoronaNet", 'cap-merge': "CAP-SotU+Court", 'cap-sotu': "CAP-SotU"
}
df_test_results["dataset"] = df_test_results["dataset"].map(dataset_map)
df_test_results = df_test_results.rename(columns={"group_col": "group"})


## tests
# note that file_name is not a unique identifier for runs anymore, because I have multiple runs per file_name for random_all
assert len(df_test.file_name.unique()) != len(df_test_grouped)
# make sure that same amount of experiments per dataset and group
df_test_results.groupby(["dataset", "group"]).apply(lambda x: len(x))




## save data for downstream analyses
if SAVE_RESULTS and USE_DOWNSAMPLED_DATA:
    df_test_results.to_csv("./results/df_results_downsampled.csv.gz", compression="gzip", index=False)
elif SAVE_RESULTS and not USE_DOWNSAMPLED_DATA:
    df_test_results.to_csv("./results/df_results.csv.gz", compression="gzip", index=False)


