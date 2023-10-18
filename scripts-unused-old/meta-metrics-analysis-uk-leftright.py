
## load libraries
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame, count, selection_rate, false_positive_rate

import functools
import pandas as pd
import numpy as np


## load data
DATASET = "uk-leftright-econ"
MAX_SAMPLE = 500
DATE = "20230207"
N_ITERATIONS_MAX = 5

n_sample_str = MAX_SAMPLE
while len(str(n_sample_str)) <= 3:
    n_sample_str = "0" + str(n_sample_str)

### load data
#df = pd.read_csv(f"./data-classified/uk-leftright/df_{DATASET}_{n_sample_str}_{DATE}.zip")
# load all dfs for different classifiers from different random runs
df_iter_lst = []
for i in range(N_ITERATIONS_MAX):
    df_step = pd.read_csv(f"data-classified/uk-leftright-econ/df_{DATASET}_{n_sample_str}_{DATE}_{i}.zip")
    df_step["classifier_iteration"] = i
    df_iter_lst.append(df_step)
df_iter = pd.concat(df_iter_lst)

# remove data that was used for training (data without predictions)
#df_cl = df[~df.label_pred.isna()].copy(deep=True)
df_iter_cl = df_iter[~df_iter.label_pred.isna()].copy(deep=True)
# TODO: improve calculation of metrics with concatenated data from multiple classifiers/samples
# unsure whether this introduces downstream issues
df_cl = df_iter_cl.copy(deep=True)

## add some meta-data
# https://manifesto-project.wzb.eu/down/data/2020a/codebooks/codebook_MPDataset_MPDS2020a.pdf
#df_cl["decade"] = [str(date)[:3] + "0" for date in df_cl.year]
#df_cl.year.value_counts()


###### meta-metric analysis

### test meta-metrics with fairlearn
# good guide: https://fairlearn.org/v0.8/user_guide/assessment/perform_fairness_assessment.html

y_true = df_cl.label
y_pred = df_cl.label_pred

## decide on metrics
# need to use functools here to specifify the averaging mode
f1_macro = functools.partial(skm.f1_score, average='macro')
precision_micro = functools.partial(skm.precision_score, average='micro')
recall_micro = functools.partial(skm.recall_score, average='micro')
precision_macro = functools.partial(skm.precision_score, average='macro')
recall_macro = functools.partial(skm.recall_score, average='macro')

# ! don't understand why balanced_accuracy != recall_macro. They are the same in overall metrics, but not when disagg

metrics_dic = {
    "f1_macro": f1_macro,
    'accuracy': skm.accuracy_score,
    'accuracy_balanced': skm.balanced_accuracy_score, #"precision_micro": precision_micro, #"recall_micro": recall_micro,
    "precision_macro": precision_macro, "recall_macro": recall_macro,
    #"selection_rate": selection_rate,  # does not seem to work on multi-class
    #"false_positive_rate": false_positive_rate,  # does not seem to work on multi-class
    'count': count,
}

# groups to analyse
meta_data_groups = ["label_text", "party", "year"]  # "decade",

## create metric frame for each group
mf_group_dic = {}
for group in meta_data_groups:
    mf = MetricFrame(
        metrics=metrics_dic,
        y_true=y_true,
        y_pred=y_pred,
        # can look at intersection between features by passing df with multiple columns
        sensitive_features=df_cl[[group]],  # df_cl[["country_iso", "language_iso", "parfam_name", "parfam_rile", "label_rile", "decade"]]
        # Control features: "When the data are split into subgroups, control features (if provided) act similarly to sensitive features. However, the ‘overall’ value for the metric is now computed for each subgroup of the control feature(s). Similarly, the aggregation functions (such as MetricFrame.group_max()) are performed for each subgroup in the conditional feature(s), rather than across them (as happens with the sensitive features)."
        # https://fairlearn.org/v0.8/user_guide/assessment/intersecting_groups.html
        #control_features=df_cl[["parfam_text"]]
    )
    mf_group_dic.update({group: mf})


## calculate overall metric (is same for all metricframes)
overall = mf_group_dic[meta_data_groups[0]].overall
df_metrics_overall = pd.DataFrame({"dataset": [DATASET] * len(overall), "metric_name": overall.index, "metric_values": overall})
df_metrics_overall = df_metrics_overall[df_metrics_overall.metric_name != "count"].reset_index(drop=True)

## main methods/attributes of MetricFrames
#print("## Metrics overall:\n", mf.overall, "\n")
#print("## Metrics by group:\n", mf.by_group, "\n")  #.to_dict()
#print("## Metrics min:\n", mf.group_min(), "\n")
#print("## Metrics max:\n", mf.group_max(), "\n")
#print("## Metrics difference min-max:\n", mf.difference(method='between_groups'), "\n")  # to_overall, between_groups    # difference or ratio of the metric values between the best and the worst slice
#print("## Metrics difference ratio min-max:\n", mf.ratio(method='between_groups'),  "\n") # to_overall, between_group  # difference or ratio of the metric values between the best and the worst slice
# scalar values from difference/ratio/min/max can be used for hp tuning  # https://fairlearn.org/main/user_guide/assessment.html#scalar-results-from-metricframe

## create statistics on each group based on metric frames
df_meta_metrics_diffmax_lst = []
df_meta_metrics_mean_lst = []
df_meta_metrics_std_lst = []
#df_meta_metrics_var_lst = []

df_meta_metrics_disaggregated_lst = []
for key_group, value_mf_group in mf_group_dic.items():
    ## full per-group disaggregated values
    by_group = value_mf_group.by_group  #.drop(columns=["count"])
    df_by_group = pd.DataFrame({"group_analysed": [key_group]*len(by_group), "group_members": by_group.index, **{key: value.values() for key, value in by_group.to_dict().items()}})
    df_meta_metrics_disaggregated_lst.append(df_by_group)
    ## macro-averaged meta-metric
    by_group_mean = by_group.mean()
    df_by_group_mean = pd.DataFrame({"group_analysed": [key_group]*len(by_group_mean), "metric": by_group_mean.index, "metric_mean": by_group_mean})
    df_meta_metrics_mean_lst.append(df_by_group_mean[df_by_group_mean.metric != "count"])
    ## standard deviation (std)
    by_group_std = by_group.std()
    df_by_group_std = pd.DataFrame({"group_analysed": [key_group]*len(by_group_std), "metric": by_group_std.index, "metric_std": by_group_std})
    df_meta_metrics_std_lst.append(df_by_group_std[df_by_group_std.metric != "count"])
    ## min-max difference
    difference = value_mf_group.difference(method='between_groups')  # ! can try: to_overall, between_groups
    df_diffmax = pd.DataFrame({"group_analysed": [key_group]*len(difference), "metric": difference.index, "metric_min_max_diff": difference})
    df_meta_metrics_diffmax_lst.append(df_diffmax)
    ## variance (var)  (is less useful here because value harder to interpret; square of 0.x gets smaller; and less expressive because different metric units harder to compare https://www.investopedia.com/terms/v/variance.asp)
    #by_group_var = by_group.var()
    #df_by_group_var = pd.DataFrame({"group_analysed": [key_group]*len(by_group_var), "metric": by_group_var.index, "metric_var": by_group_var})
    #df_meta_metrics_var_lst.append(df_by_group_var[df_by_group_var.metric != "count"])

df_meta_metrics_mean = pd.concat(df_meta_metrics_mean_lst).reset_index(drop=True)
df_meta_metrics_std = pd.concat(df_meta_metrics_std_lst).reset_index(drop=True)
df_meta_metrics_diffmax = pd.concat(df_meta_metrics_diffmax_lst).reset_index(drop=True)
df_meta_metrics_disaggregated = pd.concat(df_meta_metrics_disaggregated_lst).reset_index(drop=True)
#df_meta_metrics_var = pd.concat(df_meta_metrics_var_lst).reset_index(drop=True)


### calculate label distribution per group-member (true and predicted)
dic_value_counts_label = {}
dic_value_counts_label_pred = {}
key_dic_created_label = []
key_dic_created_label_pred = []
for group in meta_data_groups:
    ## value counts by groups for both label and label_pred
    # known issue of different return formats with groupby https://github.com/pandas-dev/pandas/issues/13056
    group_label_count = df_cl.groupby(by=group).apply(lambda x: x.label.value_counts(sort=True).to_frame())  #.stack()  #.to_frame()
    group_label_count_pred = df_cl.groupby(by=group).apply(lambda x: x.label_pred.astype(int).value_counts(sort=True).to_frame())
    # create dic for rows for actual label counts
    for key, value in next(iter(group_label_count.to_dict().values())).items():
        if key[0] not in key_dic_created_label:
            dic_value_counts_label.update({key[0]: {key[1]: value}})
            key_dic_created_label.append(key[0])
        if key[0] in key_dic_created_label:
            dic_value_counts_label[key[0]].update({key[1]: value})
    # create dic for rwos for predicted label counts
    for key, value in next(iter(group_label_count_pred.to_dict().values())).items():
        if key[0] not in key_dic_created_label_pred:
            dic_value_counts_label_pred.update({key[0]: {key[1]: value}})
            key_dic_created_label_pred.append(key[0])
        if key[0] in key_dic_created_label_pred:
            dic_value_counts_label_pred[key[0]].update({key[1]: value})

df_group_label_counts = pd.DataFrame({"group_members": dic_value_counts_label.keys(), "label_distribution": dic_value_counts_label.values()})
df_group_label_counts_pred = pd.DataFrame({"group_members": dic_value_counts_label_pred.keys(), "label_distribution_pred": dic_value_counts_label_pred.values()})

## merge label distribution (true and predicted) with disaggregated df
df_meta_metrics_disaggregated = df_meta_metrics_disaggregated.merge(df_group_label_counts, on="group_members")
df_meta_metrics_disaggregated = df_meta_metrics_disaggregated.merge(df_group_label_counts_pred, on="group_members")
# sort
df_meta_metrics_disaggregated = df_meta_metrics_disaggregated.sort_values(by=["group_analysed", "f1_macro"])

### get single metric summarising performance across meta-data
# excluding label_text from this, because aggregated metrics less meaningful

# "country_iso", "language_iso", "parfam_text", "label_text", "parfam_rile", "label_rile", "decade"
# ["label_text", "party", "decade", "year"]
relevant_groups = ["label_text", "party", "year"]

# mean of mean across meta-data
df_meta_metrics_mean_mean = df_meta_metrics_mean[df_meta_metrics_mean.group_analysed.isin(relevant_groups)].groupby("metric", as_index=False, group_keys=False).apply(lambda x: x.mean())
df_meta_metrics_mean_mean = df_meta_metrics_mean_mean.rename(columns={"metric_std": "metric_std_mean"})
# mean of std across meta-data
df_meta_metrics_std_mean = df_meta_metrics_std[df_meta_metrics_std.group_analysed.isin(relevant_groups)].groupby("metric", as_index=False, group_keys=False).apply(lambda x: x.mean())
df_meta_metrics_std_mean = df_meta_metrics_std_mean.rename(columns={"metric_std": "metric_std_mean"})
# mean of minmax across meta-data
df_meta_metrics_diffmax_mean = df_meta_metrics_diffmax[df_meta_metrics_diffmax.group_analysed.isin(relevant_groups)].groupby("metric", as_index=False, group_keys=False).apply(lambda x: x.mean())
df_meta_metrics_diffmax_mean = df_meta_metrics_diffmax_mean.rename(columns={"metric_std": "metric_std_mean"})




###### create meta-metric df to merge with correlation df
# to investigate between meta-metric std and correlation validity performance

## convert sentence level classification predictions to aggregate manifesto scores
df_aggreg_classifier = df_cl.groupby("manifestoid", as_index=False, group_keys=False).apply(lambda x: x.label_scale_pred.mean())
df_aggreg_classifier = df_aggreg_classifier.rename(columns={None: "score_manif_mean"})

## load expert data
df_aggreg_gold = pd.read_csv("./data-clean/benoit_leftright_manifestos_gold.zip")
df_aggreg_experts = df_aggreg_gold[(df_aggreg_gold.source == "Experts") & (df_aggreg_gold.scale == "Economic")]

## expert surveys
df_aggreg_surveys = pd.read_excel("./data-clean/uk-leftright-surveys-frompdf.xlsx")
# 1992 is missing! Until I get it, I just insert the values for 1987, from figure 2 in paper they seem to have almost the same value.
# TODO: ask benoit about expert survey data
df_dummy = df_aggreg_surveys[df_aggreg_surveys.Year == 1987]
df_dummy["Year"] = 1992
df_aggreg_surveys = pd.concat([df_aggreg_surveys, df_dummy])
# select dimension
df_aggreg_surveys = df_aggreg_surveys[df_aggreg_surveys["Dimension"] == "Economic"]
# select parties that were annotated
df_aggreg_surveys = df_aggreg_surveys[df_aggreg_surveys.Party.isin(["Con", "Lab", "LD"])]
# manifesto id for merging with annotated data
df_aggreg_surveys["manifestoid"] = df_aggreg_surveys["Party"] + " " + df_aggreg_surveys["Year"].astype(str)
df_aggreg_surveys = df_aggreg_surveys.sort_values("manifestoid").reset_index(drop=True)




## correlate
from scipy.stats import pearsonr

pearson_cor = pearsonr(df_aggreg_surveys["Mean"], df_aggreg_classifier["score_manif_mean"])
print(pearson_cor)

# create df to save to disk
df_pearson_cor = pd.DataFrame({"pearson-r": pearson_cor[0], "p-value": pearson_cor[1]},
                              index=["correlation of classifier output (trained on 1k crowd annotations, prediction on 5k) with expert data"])



#### create summary dict for relevant meta-metric analysis results and write to disk
meta_metrics_analysis_dic = {
    "df_meta_metrics_disaggregated": df_meta_metrics_disaggregated,
    "df_meta_metrics_mean_mean": df_meta_metrics_mean_mean,
    "df_meta_metrics_diffmax_mean": df_meta_metrics_diffmax_mean,
    "df_meta_metrics_std_mean": df_meta_metrics_std_mean,
    "df_metrics_overall": df_metrics_overall,
    "df_pearson_cor": df_pearson_cor,
}

## write to disk
#with pd.ExcelWriter("./data-analysed/df_meta_metrics_benoit1_year_party_fairlearn.xlsx", engine='xlsxwriter') as writer:
#    for key_df_name, value_df in meta_metrics_analysis_dic.items():
#        value_df.to_excel(writer, sheet_name=key_df_name)
#    writer.save()


