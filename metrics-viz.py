# visualize results with classical classification metrics and bias penalty

import pandas as pd

# choose classification metric for figure
METRIC = "f1_macro"  # "f1_macro", "f1_micro", "accuracy_balanced"
AGGREGATE_METHOD_ONLY = True
DOWNSAMPLE_FOR_BALANCED_TEST = True

# load data
if DOWNSAMPLE_FOR_BALANCED_TEST:
    df_results = pd.read_csv("./results/df_results_downsampled.csv.gz")
else:
    df_results = pd.read_csv("./results/df_results.csv.gz")




### calculate means and standard deviations

# decide on degree of aggregation
if AGGREGATE_METHOD_ONLY:
    variables_to_group_by = ["method"]  # ["dataset", "group", "method"] ["method"]
else:
    variables_to_group_by = ["dataset", "group", "method"]  #["dataset", "group", "method"] ["method"]

# Note: this mean calculation gives higher value to datasets with more groups in aggregate case
df_results_mean = df_results.groupby(["data_train_biased"] + variables_to_group_by).mean().reset_index()

# add standard deviation for error bar
df_results_mean[["eval_f1_macro_std", "eval_f1_micro_std", "eval_accuracy_balanced_std"]] = df_results.groupby(
    ["data_train_biased"] + variables_to_group_by, as_index=True
)[["eval_f1_macro", "eval_f1_micro", "eval_accuracy_balanced"]].std().reset_index(drop=True)
#df_results_mean = pd.merge(df_results_mean, df_results_std, on=variables_to_group_by + ["sample_size_train", "data_train_biased"], how='outer', suffixes=["_mean", "_std"])


## merge biased and randomall metrics into same row
df_results_mean_biased = df_results_mean[df_results_mean["data_train_biased"] == True]
df_results_mean_randomall = df_results_mean[df_results_mean["data_train_biased"] == False]
df_results_mean_biased.drop(columns=["data_train_biased", "sample_size_train"], inplace=True)
df_results_mean_randomall.drop(columns=["data_train_biased", "sample_size_train"], inplace=True)

df_merged = pd.merge(df_results_mean_biased, df_results_mean_randomall, on=variables_to_group_by, how='outer', suffixes=["_biased", "_random"])

# order values
df_merged.loc[:,'method'] = pd.Categorical(
    df_merged['method'], categories=["BERT-NLI", "BERT-NLI-void", "BERT-base", "logistic reg."], ordered=True
)
df_merged = df_merged.sort_values(variables_to_group_by)




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
    line1, caplines1, barlinecols1 = plt.errorbar(df_merged[f"eval_{METRIC}_biased"], y_positions, xerr=df_merged[f"eval_{METRIC}_std_biased"], fmt='o', color='r', label='data_train biased')
    # Make the error bar for 'random 1' dotted
    for bar in barlinecols1:
        bar.set_linestyle('--')
    # Get the errorbar object for 'random all'
    line2, caplines2, barlinecols2 = plt.errorbar(df_merged[f"eval_{METRIC}_random"], y_positions, xerr=df_merged[f"eval_{METRIC}_std_random"], fmt='o', color='b', label='data_train random')
    # Make the error bar for 'random all' dotted
    for bar in barlinecols2:
        bar.set_linestyle(':')
else:
    plt.scatter(df_merged[f"eval_{METRIC}_biased"], y_positions, color='r', label='data_train biased', marker='o')
    plt.scatter(df_merged[f"eval_{METRIC}_random"], y_positions, color='b', label='data_train random', marker='o')

# Add arrows
if AGGREGATE_METHOD_ONLY:
    for y, x1, x2 in zip(y_positions, df_merged[f"eval_{METRIC}_biased"], df_merged[f"eval_{METRIC}_random"]):
        plt.annotate('', xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="<-", color="black", lw=1, ls="-"))
        # Calculate the difference and position the value slightly above the arrow
        difference = x2 - x1
        plt.text((x1 + x2) / 2, y, f'{difference:.3f}', ha='center', va='bottom', fontsize=13, color='black')

plt.xlabel(METRIC, fontsize=14)  # Adjusted font size

if AGGREGATE_METHOD_ONLY:
    plt.yticks(y_positions, categories, fontsize=14)  # Adjusted font size
    plt.legend(loc='upper right', fontsize=12)  # Adjusted font size
    #plt.ylabel('Methods', fontsize=14)  # Added y-label with adjusted font size
    plt.ylim(y_positions[0] - 0.5, y_positions[-1] + 0.5)
    plt.title(f'Average performance ({len(df_results.dataset.unique())} datasets, {len(df_results.group.unique())} groups) & bias penalty', fontsize=16)  # Adjusted font size
else:
    plt.yticks(y_positions, categories, fontsize=12)  # Adjusted font size
    plt.legend(loc='upper right', fontsize=12)  # Adjusted font size
    plt.ylabel('Methods & Datasets & Groups', fontsize=14)  # Added y-label with adjusted font size
    plt.title(f'Disaggregated performance on biased and unbiased data ({len(df_results.dataset.unique())} datasets, {len(df_results.group.unique())} groups)', fontsize=16)  # Adjusted font size

# Add horizontal and vertical grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# Layout adjustment for better appearance
plt.tight_layout()
plt.show()




