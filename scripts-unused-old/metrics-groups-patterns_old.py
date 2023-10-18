



import pandas as pd

# load list of group dfs
n_sample = "1000"

if n_sample == "0500":
    group_random1 = ["nld", "esp", "deu", "dnk"] #["nld", "esp", "dnk", "deu"]  # ["nld", "esp", "deu", "dnk", "aut"]
    group_lst = group_random1 + ["random3", "randomall"]  #["random3"], ["random2"], ["nld", "esp", "dnk", "deu"], ["CHR", "LEF", "LIB", "NAT", "SOC"]
elif n_sample == "1000":
    #group_random1 = ["nld", "esp", "deu", "dnk", "aut"] #["nld", "esp", "dnk", "deu"]  # ["nld", "esp", "deu", "dnk", "aut"]
    group_lst = ["random2", "random3", "randomall"]  #["random3"], ["random2"], ["nld", "esp", "dnk", "deu"], ["CHR", "LEF", "LIB", "NAT", "SOC"]


method_lst = ["classical_ml", "standard_dl", "nli", "nli_long", "nli_void"]
#n_tokens_remove_lst = [0, 5, 10]

df_metrics = []
for group in group_lst:
    df_group_lst = []
    #for n_tokens_remove in n_tokens_remove_lst:
    for method in method_lst:
        for seed in [102, 435, 860, 270, 106]:  # [102, 435, 860, 270, 106]
            # TODO: only temporary fix for 500 samp files
            if n_sample == "0500":
                if seed in [270, 106]:  # and (group == "random")
                    continue
                # name harmonization where older files have "random" instead of "randomall"
                if group == "randomall" and method not in ["nli_long", "nli_void"]:
                    df_group_step = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_results_pimpo_random_{method}_samp{n_sample}_seed{seed}_20230427.csv")
                    #df_group_step = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_results_pimpo_{group}_{method}_samp{n_sample}_seed{seed}_20230427.csv")
                else:
                    df_group_step = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_results_pimpo_{group}_{method}_samp{n_sample}_seed{seed}_20230427.csv")
            elif n_sample == "1000":
                df_group_step = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_results_pimpo_{group}_{method}_samp{n_sample}_seed{seed}_20230427.csv")

            # TODO: fix inconsistency between random (500 samp) and randomall (1k samp)
            if group == "random":
                group_fix = "randomall"
            else:
                group_fix = group
            #df_group_step[["group", "seed", "method"]] = group, "seed_"+str(seed), method
            df_group_step[["sample", "seed", "method"]] = group_fix, "seed_"+str(seed), method
            #df_group_step[["group", "seed", "n_tokens_removed"]] = group, "seed_"+str(seed), "tokrm_"+str(n_tokens_remove)

            # order and clean columns
            #df_group_step = df_group_step[["group", "seed"] + df_group_step.columns.tolist()]
            df_group_lst.append(df_group_step.drop(columns=["eval_label_gold_raw", "eval_label_predicted_raw"]))

    df_group = pd.concat(df_group_lst)
    df_metrics.append(df_group)

# calculate mean over the random seeds
df_metrics = pd.concat(df_metrics, axis=0, ignore_index=True)
df_metrics.drop(columns=[
    'eval_accuracy_not_b', 'eval_precision_macro', 'eval_recall_macro', 'eval_precision_micro', 'eval_recall_micro',
    'eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch'
    ], inplace=True)

# calculate mean over groups and methods
df_metrics_mean = df_metrics.groupby(["sample", "method"]).mean().reset_index()

# calculate overall mean of metrics for single group samples
#df_metrics_mean_groups = df_metrics_mean[df_metrics_mean["sample"] != "random"].groupby("method", as_index=True, group_keys=True).mean().reset_index()
if n_sample == "0500":
    df_metrics_mean_random1 = df_metrics_mean[df_metrics_mean["sample"].str.contains("|".join(group_random1))].groupby("method", as_index=True, group_keys=True).mean().reset_index()
    df_metrics_mean_random1.insert(0, "sample", ["random1"] * len(df_metrics_mean_random1))
elif n_sample == "1000":
    # naming it random1 instead of 2 to harmonize names
    df_metrics_mean_random1 = df_metrics_mean[df_metrics_mean["sample"] == "random2"]
# create df with metrics more than single group samples
df_metrics_mean_randomall = df_metrics_mean[df_metrics_mean["sample"] == "randomall"]
df_metrics_mean_random3 = df_metrics_mean[df_metrics_mean["sample"] == "random3"]

df_metrics_comparison = pd.concat([df_metrics_mean_random1, df_metrics_mean_random3, df_metrics_mean_randomall])

# order columns and add index for col names for final df
# sort df_metrics_comparison by methods column with values in order of method_lst
df_metrics_comparison = df_metrics_comparison.set_index("method").loc[method_lst].reset_index()
# add group-method index for names of columns in final df
index_col = [["random1_" + method, "random3_" + method, "randomall_" + method] for method in method_lst] #+ ["random_" + method for method in method_lst]
index_col = [item for sublist in index_col for item in sublist]
df_metrics_comparison.index = index_col
df_metrics_comparison = df_metrics_comparison.round(2)
df_metrics_comparison.rename(columns={'eval_f1_macro': 'f1_macro', 'eval_f1_micro': "f1_micro",
                                      'eval_accuracy_balanced': "accuracy_balanced"}, inplace=True)
df_metrics_comparison.drop(columns=["f1_micro"], inplace=True)


## create df comparing the relative decrease from group-sample to random-sample per method
# hypothesis:
# 1. the performance on data-random is higher than on data-group with both classical and instruction models
# 2. the performance decrease from data-group to data-random is stronger for classical than for instruction models
dic_metrics_comparison_dic = {}
for method in method_lst:
    # randomall - random1
    dic_metrics_comparison_dic.update({
        f"random1_{method}": df_metrics_comparison[["f1_macro", "accuracy_balanced"]].loc[f"random1_{method}"].sub(
            df_metrics_comparison[["f1_macro", "accuracy_balanced"]].loc[f"randomall_{method}"])})
for method in method_lst:
    # randomall - random3
    dic_metrics_comparison_dic.update({
        f"random3_{method}": df_metrics_comparison[["f1_macro", "accuracy_balanced"]].loc[f"random3_{method}"].sub(
            df_metrics_comparison[["f1_macro", "accuracy_balanced"]].loc[f"randomall_{method}"])})

df_increase = pd.DataFrame(dic_metrics_comparison_dic).round(2).T

# merge df_metrics_comparison and df_increase on index
df_metrics_comparison_merged = df_metrics_comparison.merge(df_increase, how="left", left_index=True, right_index=True, suffixes=["", "_decrease"])
# clean
sample_name_map = {"random1": "biased (1 group)", "random2": "biased (2 groups)", "random3": "biased (3 groups)", "randomall": "unbiased"}
df_metrics_comparison_merged["sample"] = df_metrics_comparison_merged["sample"].map(sample_name_map)
method_name_map = {"classical_ml": "logistic regression", "standard_dl": "BERT-base", "nli": "BERT-instruct-short", "nli_long": "BERT-instruct-long", "nli_void": "BERT-instruct-meaningless"}
df_metrics_comparison_merged["method"] = df_metrics_comparison_merged["method"].map(method_name_map)

df_metrics_comparison_merged.drop(columns=["accuracy_balanced", "accuracy_balanced_decrease"], inplace=True)
df_metrics_comparison_merged.reset_index(inplace=True, drop=True)

# write to disk
#df_metrics_comparison.to_excel("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_metrics_comparison_parfam.xlsx")
#df_increase.to_excel("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_metrics_increase_parfam.xlsx")
#df_metrics_comparison_merged.to_excel(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_metrics_decrease_countries_samp{n_sample}.xlsx")


## analyses for standard deviation
#df_metrics_std = df_metrics.groupby(["group", "method"]).std().reset_index()
#df_metrics_std_groups = df_metrics_std[df_metrics_std["sample"] != "random"].groupby("method", as_index=True, group_keys=True).std().reset_index()
#df_metrics_std_random = df_metrics_std[df_metrics_std["sample"] == "random"]


