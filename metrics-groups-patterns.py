

import pandas as pd
import os
import pickle
import gzip

dataset_lst = ["pimpo", "coronanet", "cap-merge"]
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


data_dic_results_lst = []
for data_dic in data_dic_lst:
    experiment_metadata_to_keep = [
        'dataset', 'group_sample_strategy', 'group_col', 'method', 'model_name', 'sample_size_train', 'sample_size_test',
        'group_members', 'seed_run', 'n_run', 'date', 'train_time', 'model_size', 'task'
    ]
    experiment_metadata = {key: value for key, value in data_dic["experiment_metadata"].items() if key in experiment_metadata_to_keep}
    experiment_results = {key: value for key, value in data_dic["experiment_results"].items()}  # if key not in ["epoch", "eval_loss"]
    data_dic_results_lst.append({**experiment_metadata, **experiment_results})

df_results = pd.DataFrame(data_dic_results_lst)

# give different method name to standard_dl with electra or NLI
df_results.loc[(df_results["method"] == "standard_dl") & (df_results["model_name"].str.contains("electra")), "method"] = "standard_dl_electra"
df_results.loc[(df_results["method"] == "standard_dl") & (df_results["model_name"].str.contains("nli")), "method"] = "standard_dl_nli"

# !! Only do some tests on NLI data
df_results = df_results[df_results["method"].str.contains("nli")]


# calculate mean over the random seeds
df_results.drop(columns=[
    'eval_accuracy_not_b', 'eval_precision_macro', 'eval_recall_macro', 'eval_precision_micro', 'eval_recall_micro',
    'eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch',
    "seed_run", "n_run", "sample_size_test",
    "train_time"
    ], inplace=True)

# calculate mean over groups and methods
df_results_mean = df_results.groupby(["group_sample_strategy", "method", "sample_size_train", "model_size", "group_col"]).mean().reset_index()
df_results_mean.index = df_results_mean["group_sample_strategy"] + "_" + df_results_mean["method"] \
                        + "_" + df_results_mean["sample_size_train"].astype(int).astype(str) \
                        + "_" + df_results_mean["model_size"] + "_" + df_results_mean["group_col"]


df_results_mean = df_results_mean.round(2)
df_results_mean.rename(columns={'eval_f1_macro': 'f1_macro', 'eval_f1_micro': "f1_micro",
                                      'eval_accuracy_balanced': "accuracy_balanced"}, inplace=True)
#df_results_mean.drop(columns=["f1_micro"], inplace=True)

results_difference_to_randomall = {}
for index in df_results_mean.index:
    if not "randomall" in index:
        corresponding_random_all_index = "randomall_" + "_".join(index.split("_")[1:])
        difference_to_randomall = df_results_mean.loc[corresponding_random_all_index][["f1_macro", "f1_micro", "accuracy_balanced"]] \
                                  - df_results_mean.loc[index][["f1_macro", "f1_micro", "accuracy_balanced"]]
        results_difference_to_randomall.update({index: difference_to_randomall.to_dict()})

df_results_mean_difference = df_results_mean.merge(pd.DataFrame(results_difference_to_randomall).T,
                                                   how="left", left_index=True, right_index=True, suffixes=["", "_diff"])

df_results_mean_difference.sort_values(["sample_size_train", "method", "group_sample_strategy"], ascending=True, inplace=True)
#df_results_mean_difference.sort_index(ascending=True, inplace=True)

# inspect individual run results
df_results.sort_values(["group_col", "group_members", "method"], ascending=True, inplace=True)



# write to disk
#df_metrics_comparison.to_excel("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_metrics_comparison_parfam.xlsx")
#df_increase.to_excel("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_metrics_increase_parfam.xlsx")
#df_metrics_comparison_merged.to_excel(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_metrics_decrease_countries_samp{n_sample}.xlsx")


