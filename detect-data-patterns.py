

import pandas as pd
import numpy as np

# load list of group dfs
#df_group = pd.read_csv("/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/data-classified/pimpo/df_feat_importance_pimpo_deu_1000_20230427.csv")#.reset_index(drop=True)

group_name_lst = ["nld", "esp", "dnk", "deu"] # ["CHR", "LEF", "LIB", "NAT", "SOC"]   # ["nld", "esp", "dnk", "deu"]

df_coef_diff_all = []
df_merged_all = []
for group_name in group_name_lst:

    df_group_lst = []
    for seed in [102, 435, 860]:  # 102, 435, 860, 270, 106
        df_group_lst.append(pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/data-classified/pimpo/df_feat_importance_pimpo_{group_name}_samp0500_seed{seed}_20230427.csv"))
    # merge all dfs in list of group dfs on "token" column and calculate mean of coefficients
    df_group = pd.concat(df_group_lst, axis=0, ignore_index=True)
    # filter rows with only X occurence of token to downweight rare tokens
    df_group = df_group.groupby("token").filter(lambda x: len(x) >= 2)
    df_group = df_group.groupby("token").mean().reset_index()

    # import list of random dfs
    """df_random_lst = []
    for seed in [102, 435, 860]:
        df_random_lst.append(pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/data-classified/pimpo/df_feat_importance_pimpo_random_samp1000_seed{seed}_20230427.csv"))
    # merge all dfs in list of random dfs on "token" column and calculate mean of coefficients
    df_random = pd.concat(df_random_lst, axis=0, ignore_index=True)
    # filter rows with only X occurence of token to downweight rare tokens
    df_random = df_random.groupby("token").filter(lambda x: len(x) >= 3)
    df_random = df_random.groupby("token").mean().reset_index()
    #df_random = pd.concat(df_random_lst, axis=0, ignore_index=True).groupby("token").apply(lambda x: pd.DataFrame(x).mean() if len(x) >= 2 else None)#.reset_index()
    """
    # import df_feat_importance trained on entire dataset
    df_random = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/data-classified/pimpo/df_feat_importance_pimpo_randomall_samp0500_seed102_20230427.csv")

    # merge df_group and df_random
    df_merged = pd.merge(df_group, df_random, on="token", how="outer", suffixes=("_group", "_random"))

    class_col_names = df_group.columns[1:5].tolist()

    # check if two numbers have different sign
    def diff_sign(x, y):
        return (x > 0) != (y > 0)

    df_coef_diff = df_merged[["token", "inverse_doc_freq_group", "inverse_doc_freq_random"]]
    for class_name in class_col_names:
        df_coef_diff[class_name + "_diff"] = df_merged[class_name + "_group"] - df_merged[class_name + "_random"]
        # interpretation: positive number means group has more positive coefficient,
        # negative number means random has more positive coefficient
        # boolean for whether the group and random have different sign
        df_coef_diff[class_name + "_diff_sign"] = df_merged.apply(lambda x: diff_sign(x[class_name + "_group"], x[class_name + "_random"]), axis=1)

    # add "_diff" suffix to class_col_names
    class_col_names = [x + "_diff" for x in class_col_names]

    # use inverse_doc_freq from random samples
    df_coef_diff.drop(columns=["inverse_doc_freq_group"], inplace=True)
    df_coef_diff = df_coef_diff.rename(columns={"inverse_doc_freq_random": "inverse_doc_freq"})

    # add sum of coefficients column
    df_coef_diff.insert(1, "coef_diff_sum", df_coef_diff[class_col_names].abs().sum(axis=1))
    df_merged.insert(1, "coef_diff_sum", df_coef_diff[class_col_names].abs().sum(axis=1))
    # add sum of different signs column
    col_sign_diff_sum = df_coef_diff[[col for col in df_coef_diff.columns if "sign" in col]].sum(axis=1)
    df_coef_diff.insert(2, "sign_diff_sum", col_sign_diff_sum)
    df_merged.insert(2, "sign_diff_sum", col_sign_diff_sum)
    # further cleaning
    df_coef_diff.insert(0, "group", group_name)
    #df_coef_diff.sort_values(by=["sign_diff_sum", "inverse_doc_freq"], ascending=[False, True], inplace=True)
    df_coef_diff.sort_values(by=["sign_diff_sum", "coef_diff_sum"], ascending=[False, False], inplace=True)
    df_merged.sort_values(by=["sign_diff_sum", "coef_diff_sum"], ascending=False, inplace=True)
    # append
    df_coef_diff_all.append(df_coef_diff.round(1))
    df_merged_all.append(df_merged.round(1))

    # write to csv
    #df_coef_diff.to_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/data-classified/pimpo/df_tokens_pimpo_{group_name}_samp1000_20230427.csv")

# write list of dfs to excel as separate sheets

"""
with pd.ExcelWriter(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/results/pimpo/df_tokens_pimpo_countries_samp0500_20230427.xlsx") as writer:
    for i, df in enumerate(df_coef_diff_all):
        df.to_excel(writer, sheet_name=group_name_lst[i], index=False)
"""

#df_coef_diff_all_country = df_coef_diff_all
#df_coef_diff_all_parfam = df_coef_diff_all
#df_coef_diff_all_country = pd.concat(df_coef_diff_all_country)
#df_coef_diff_all_parfam = pd.concat(df_coef_diff_all_parfam)


coef_diff_sum_country_mean = df_coef_diff_all_country.groupby("group").apply(lambda x: x[x.sign_diff_sum > 2].coef_diff_sum.sum()).mean()
print(coef_diff_sum_country_mean)

coef_diff_sum_parfam_mean = df_coef_diff_all_parfam.groupby("group").apply(lambda x: x[x.sign_diff_sum > 2].coef_diff_sum.sum()).mean()
print(coef_diff_sum_parfam_mean)


# average share of tokens that have at least X different signs
df_coef_diff_all_country.groupby("group").apply(lambda x: len(x[x.sign_diff_sum >= 2])).mean() / (len(df_coef_diff_all_country) / 4)
df_coef_diff_all_parfam.groupby("group").apply(lambda x: len(x[x.sign_diff_sum >= 2])).mean() / (len(df_coef_diff_all_parfam) / 5)


