
import pandas as pd
import os
import pickle
import gzip


# choose classification metric for figure
METRIC = "f1_macro"  # "f1_macro", "f1_micro", "accuracy_balanced"


## Load data: loop over all files in the directory
dataset_lst = ["pimpo", "coronanet", "cap-merge", "cap-sotu"]
directory = "./results/"

data_dic_lst = []
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


## convert all experiment result data into one dataframe
data_dic_results_lst = []
for data_dic in data_dic_lst:
    experiment_metadata_to_keep = [
        'dataset', 'group_sample_strategy', 'group_col', 'method', 'model_name', 'sample_size_train',
        'group_members', 'seed_run', 'n_run', 'date', 'train_time', 'model_size', 'task'
    ]
    experiment_metadata = {key: value for key, value in data_dic["experiment_metadata"].items() if key in experiment_metadata_to_keep}
    experiment_results = {key: value for key, value in data_dic["experiment_results"].items()}  # if key not in ["epoch", "eval_loss"]
    data_dic_results_lst.append({**experiment_metadata, **experiment_results})

df_results = pd.DataFrame(data_dic_results_lst)


## some cleaning
df_results.drop(columns=[
    'eval_accuracy_not_b', 'eval_precision_macro', 'eval_recall_macro', 'eval_precision_micro', 'eval_recall_micro',
    'eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch',
    "seed_run", "n_run", "train_time"
    ], inplace=True)

# renaming some values / columns for paper
method_map = {
    "classical_ml": "logistic reg.", "standard_dl": "BERT-base", "nli_short": "BERT-NLI"
}
df_results["method"] = df_results["method"].map(method_map)
group_map = {
    'parfam_text': "party fam", 'country_iso': "country", 'decade': "decade",
    'year': "year", 'continent': "continent", 'ISO_A3': "country",
    'domain': "domain", 'pres_party': "party", 'phase': "phase"
}
df_results["group_col"] = df_results["group_col"].map(group_map)
dataset_map = {
    'pimpo': "PImPo", 'coronanet': "CoronaNet", 'cap-merge': "CAP-SotU+Court", 'cap-sotu': "CAP-SotU"
}
df_results["dataset"] = df_results["dataset"].map(dataset_map)
df_results = df_results.rename(columns={"group_col": "group"})



AGGREGATE_METHOD_ONLY = True
if AGGREGATE_METHOD_ONLY:
    variables_to_group_by = ["method"]  # ["dataset", "group", "method"] ["method"]
else:
    variables_to_group_by = ["dataset", "group", "method"]  #["dataset", "group", "method"] ["method"]

# remove nli_void for now
df_results = df_results[df_results["method"] != "nli_void"]
# removing experiments with 100 data train because too instable and unrealistic
df_results = df_results[df_results["sample_size_train"] == 500]


## calculate means and standard deviations
df_results_mean = df_results.groupby(["group_sample_strategy"] + variables_to_group_by).mean().reset_index()

# add standard deviation for error bar
df_results_mean[["eval_f1_macro_std", "eval_f1_micro_std", "eval_accuracy_balanced_std"]] = df_results.groupby(["group_sample_strategy"] + variables_to_group_by, as_index=True)[["eval_f1_macro", "eval_f1_micro", "eval_accuracy_balanced"]].std().reset_index(drop=True)
#df_results_mean = pd.merge(df_results_mean, df_results_std, on=variables_to_group_by + ["sample_size_train", "group_sample_strategy"], how='outer', suffixes=["_mean", "_std"])

# merge random1 and randomall metrics into same row
df_results_mean_random1 = df_results_mean[df_results_mean["group_sample_strategy"] == "random1"]
df_results_mean_randomall = df_results_mean[df_results_mean["group_sample_strategy"] == "randomall"]
df_results_mean_random1.drop(columns=["group_sample_strategy"], inplace=True)
df_results_mean_randomall.drop(columns=["group_sample_strategy"], inplace=True)
# TODO: fix somehow: sample size train does not have same amount of rows for each scenario.
df_results_mean_random1.drop(columns=["sample_size_train"], inplace=True)
df_results_mean_randomall.drop(columns=["sample_size_train"], inplace=True)

df_merged = pd.merge(df_results_mean_random1, df_results_mean_randomall, on=variables_to_group_by, how='outer', suffixes=["_random1", "_randomall"])


# order values
df_merged.loc[:,'method'] = pd.Categorical(
    df_merged['method'], categories=["BERT-NLI", "BERT-base", "logistic reg."], ordered=True
)
#df_merged.loc[:,'dataset'] = pd.Categorical(
#    df_merged['dataset'], categories=['PImPo', 'CoronaNet', 'CAP-SotU', 'CAP-SotU+Court'], ordered=True
#)
#df_merged.loc[:,'group'] = pd.Categorical(
#    df_merged['group'], categories=["BERT-NLI", "BERT-base", "logistic reg."], ordered=True
#)
df_merged = df_merged.sort_values(variables_to_group_by)  #(["dataset", "group", "method"])




### viz
import matplotlib.pyplot as plt
import numpy as np

# create strings for y-axis labels. maybe improve later.
if AGGREGATE_METHOD_ONLY:
    categories = df_merged.method.astype(str) #+ " - " + df_merged.group + " - " + df_merged.dataset
else:
    categories = df_merged.method.astype(str) + " - " + df_merged.group.astype(str) + " - " + df_merged.dataset.astype(str)


# Y-axis positions for categories
y_positions = np.arange(len(categories))

# Set the size and style of the figure
if AGGREGATE_METHOD_ONLY:
    plt.figure(figsize=(12, 5))
else:
    plt.figure(figsize=(12, 10))

plt.style.use('seaborn-whitegrid')  # Use seaborn's whitegrid style

# Plot
#plt.errorbar(df_merged.eval_f1_macro_random1, y_positions, xerr=df_merged.eval_f1_macro_std_random1, fmt='o', color='r', label='random 1')
#plt.errorbar(df_merged.eval_f1_macro_randomall, y_positions, xerr=df_merged.eval_f1_macro_std_randomall, fmt='o', color='b', label='random all')
if not AGGREGATE_METHOD_ONLY:
    # Get the errorbar object for 'random 1'
    line1, caplines1, barlinecols1 = plt.errorbar(df_merged[f"eval_{METRIC}_random1"], y_positions, xerr=df_merged[f"eval_{METRIC}_std_random1"], fmt='o', color='r', label='data_train biased')
    # Make the error bar for 'random 1' dotted
    for bar in barlinecols1:
        bar.set_linestyle('--')
    # Get the errorbar object for 'random all'
    line2, caplines2, barlinecols2 = plt.errorbar(df_merged[f"eval_{METRIC}_randomall"], y_positions, xerr=df_merged[f"eval_{METRIC}_std_randomall"], fmt='o', color='b', label='data_train random')
    # Make the error bar for 'random all' dotted
    for bar in barlinecols2:
        bar.set_linestyle(':')
else:
    plt.scatter(df_merged[f"eval_{METRIC}_random1"], y_positions, color='r', label='data_train biased', marker='o')
    plt.scatter(df_merged[f"eval_{METRIC}_randomall"], y_positions, color='b', label='data_train random', marker='o')

# Add arrows
if AGGREGATE_METHOD_ONLY:
    for y, x1, x2 in zip(y_positions, df_merged[f"eval_{METRIC}_random1"], df_merged[f"eval_{METRIC}_randomall"]):
        plt.annotate('', xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1, ls="-"))
        # Calculate the difference and position the value slightly above the arrow
        difference = x2 - x1
        plt.text((x1 + x2) / 2, y, f'{difference:.3f}', ha='center', va='bottom', fontsize=13, color='black')

plt.xlabel(METRIC, fontsize=14)  # Adjusted font size

if AGGREGATE_METHOD_ONLY:
    plt.yticks(y_positions, categories, fontsize=14)  # Adjusted font size
    plt.legend(loc='upper right', fontsize=12)  # Adjusted font size
    #plt.ylabel('Methods', fontsize=14)  # Added y-label with adjusted font size
    plt.ylim(y_positions[0] - 0.5, y_positions[-1] + 0.5)
    plt.title(f'Average performance ({len(df_results.dataset.unique())} datasets, {len(df_results.group.unique())+1} groups) & bias penalty', fontsize=16)  # Adjusted font size
else:
    plt.yticks(y_positions, categories, fontsize=12)  # Adjusted font size
    plt.legend(loc='upper right', fontsize=12)  # Adjusted font size
    plt.ylabel('Methods & Datasets & Groups', fontsize=14)  # Added y-label with adjusted font size
    plt.title(f'Disaggregated performance on biased and unbiased data ({len(df_results.dataset.unique())} datasets, {len(df_results.group.unique())+1} groups)', fontsize=16)  # Adjusted font size

# Add horizontal and vertical grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.axhline(0, color='black', linewidth=0.5)
#plt.axvline(0, color='black', linewidth=0.5)

# Layout adjustment for better appearance
plt.tight_layout()
plt.show()





