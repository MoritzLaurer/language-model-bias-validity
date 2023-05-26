
## load libraries
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame, count, selection_rate, false_positive_rate

import functools
import pandas as pd
import numpy as np


## load one df
DATASET = "pimpo"
"""DATASET = "pimpo"
TASK = "immigration"
METHOD = "nli"  # nli, dl_embed, classical_ml
MODEL_SIZE = "large"  # classical, base, large
HYPOTHESIS = "long"
VECTORIZER = "en"  # en, multi, tfidf
MAX_SAMPLE_LANG = 500
DATE = 221111  # 230127, 221111
LANGS = ["en", "de"]
langs_concat = "_".join(LANGS)
# df_pimpo_pred_immigration_nli_large_long_en_500samp_en_de_sv_fr_221111, df_pimpo_pred_immigration_dl_embed_classical_long_multi_500samp_en_221111
df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.zip")
#df = pd.read_csv("./data-classified/pimpo/df_pimpo_pred_immigration_classical_ml_None_long_tfidf_500samp_en_221111.zip")
"""

### load multiple dfs
## loop to iterate over all predictions from all classifiers
from os import listdir
from os.path import isfile, join
path = "/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/results/pimpo"
file_name_lst = [f for f in listdir(path) if isfile(join(path, f)) and f != ".DS_Store" and "230127" not in f]



# dictionary with columns for correlation df - for creating df to merge with correlation df
meta_metric_row_dic = {
   "meta data": [], "algorithm": [], "language representation": [], "algorithm size": [], "training languages": [],
   "f1_macro_meta_std_mean": [], "accuracy_meta_std_mean": [], #"accuracy_balanced_std_mean": [],
    "f1_macro_meta_maxdiff_mean": [], "accuracy_meta_maxdiff_mean": [], #"accuracy_balanced_min_max_diff_mean": [],
    "f1_macro_meta_mean": [], "accuracy_meta_mean": [],
    "f1_macro_overall": [], "accuracy_overall": [],
}

# dictionary for detailed meta-metric analyses per classifier
meta_metric_per_classifier_dic = {}

## loop over each file
for file_name in file_name_lst:
    # load df
    df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/results/pimpo/{file_name}")

    # extract model identifiers from file name
    if "long_en" in file_name:
        vectorizer = "en"
    elif "long_multi" in file_name:
        vectorizer = "multi"
    else:
        raise Exception("vectorizer not found for ", file_name)
    if "base" in file_name:
        size = "base"
    elif "large" in file_name:
        size = "large"
    elif "classical_long" in file_name:
        size = "base"
    else:
        raise Exception("size not found for ", file_name)
    if "samp_en_de_sv_fr" in file_name:
        langs = "en-de-sv-fr"
    elif "samp_en_de" in file_name:
        langs = "en-de"
    elif "samp_en" in file_name:
        langs = "en"
    else:
        raise Exception("langs not found for ", file_name)
    if "nli" in file_name:
        model = "NLI-Transformer"
    elif "standard_dl" in file_name:
        model = "Transformer"
    elif "dl_embed" in file_name:
        model = "Sent.-Transformer"
    else:
        raise Exception("model not found for ", file_name)


    ####### meta metric code
    # !! start of 1 indentation for loop

    # remove data that was used for training (data without predictions)
    df_cl = df[~df.label_pred.isna()].copy(deep=True)

    # remove all data from languages used during training. otherwise too imbalanced/no data for non-other classes
    # !! this means that the metrics are slightly different here than in the multilingual paper
    #df_cl = df_cl[~df_cl.language_iso.isin(LANGS)]
    LANGS = langs.split("-")
    df_cl = df_cl[~df_cl.language_iso.isin(LANGS)]

    # add some meta-data
    # https://manifesto-project.wzb.eu/down/data/2020a/codebooks/codebook_MPDataset_MPDS2020a.pdf
    df_cl["parfam_rile"] = ["left" if parfam in [10, 20, 30] else "right" if parfam in [50, 60, 70, 80, 90] else "other" for parfam in df_cl["parfam"]]
    cmp_code_left = [103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701]
    cmp_code_right = [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606]
    cmp_code_other = np.unique([cmp_code for cmp_code in df_cl["cmp_code"] if cmp_code not in cmp_code_left + cmp_code_right])
    df_cl["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df_cl["cmp_code"]]
    df_cl["decade"] = [str(date)[:3]+"0" for date in df_cl.date]




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
    meta_data_groups = ["label_text", "country_iso", "language_iso", "parfam_text", "parfam_rile", "label_rile", "decade"]

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
        df_diffmax = pd.DataFrame({"group_analysed": [key_group]*len(difference), "metric": difference.index, "metric_maxdiff": difference})
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

    # "country_iso", "language_iso", "parfam_text", "parfam_rile", "label_rile", "decade", "label_text",
    # !! choose groups to calculate the mean across
    groups_for_cross_group_average = ["country_iso", "language_iso", "parfam_text", "parfam_rile", "label_rile", "decade"]

    # mean of mean across meta-data
    df_meta_metrics_mean_mean = df_meta_metrics_mean[df_meta_metrics_mean.group_analysed.isin(groups_for_cross_group_average)].groupby("metric", as_index=False, group_keys=False).apply(lambda x: x.mean())
    df_meta_metrics_mean_mean = df_meta_metrics_mean_mean.rename(columns={"metric_std": "metric_std_mean"})
    # mean of std across meta-data
    df_meta_metrics_std_mean = df_meta_metrics_std[df_meta_metrics_std.group_analysed.isin(groups_for_cross_group_average)].groupby("metric", as_index=False, group_keys=False).apply(lambda x: x.mean())
    df_meta_metrics_std_mean = df_meta_metrics_std_mean.rename(columns={"metric_std": "metric_std_mean"})
    # mean of minmax across meta-data
    df_meta_metrics_diffmax_mean = df_meta_metrics_diffmax[df_meta_metrics_diffmax.group_analysed.isin(groups_for_cross_group_average)].groupby("metric", as_index=False, group_keys=False).apply(lambda x: x.mean())
    df_meta_metrics_diffmax_mean = df_meta_metrics_diffmax_mean.rename(columns={"metric_std": "metric_std_mean"})

    meta_metric_per_classifier_dic.update(
        {f"{model}_{vectorizer}_{size}_{langs}":
            {
            "df_meta_metrics_disaggregated": df_meta_metrics_disaggregated,
            "df_meta_metrics_mean_mean": df_meta_metrics_mean_mean,
            "df_meta_metrics_diffmax_mean": df_meta_metrics_diffmax_mean,
            "df_meta_metrics_std_mean": df_meta_metrics_std_mean,
            "df_metrics_overall": df_metrics_overall,
            }
        }
    )

    #### extract values for rows to merge with correlation df
    meta_metric_row_dic["meta data"].append("parfam_text")
    meta_metric_row_dic["algorithm"].append(model)
    meta_metric_row_dic["language representation"].append(vectorizer)
    meta_metric_row_dic["algorithm size"].append(size)
    meta_metric_row_dic["training languages"].append(langs)
    # mean std
    meta_metric_row_dic["f1_macro_meta_std_mean"].append(df_meta_metrics_std_mean.loc[df_meta_metrics_std_mean.metric == "f1_macro", "metric_std_mean"].iloc[0])
    meta_metric_row_dic["accuracy_meta_std_mean"].append(df_meta_metrics_std_mean.loc[df_meta_metrics_std_mean.metric == "accuracy", "metric_std_mean"].iloc[0])
    #meta_metric_row_dic["accuracy_balanced_std_mean"].append(df_meta_metrics_std_mean.loc[df_meta_metrics_std_mean.metric == "accuracy_balanced", "metric_std_mean"].iloc[0])
    # mean maxdiff
    meta_metric_row_dic["f1_macro_meta_maxdiff_mean"].append(df_meta_metrics_diffmax_mean.loc[df_meta_metrics_diffmax_mean.metric == "f1_macro", "metric_maxdiff"].iloc[0])
    meta_metric_row_dic["accuracy_meta_maxdiff_mean"].append(df_meta_metrics_diffmax_mean.loc[df_meta_metrics_diffmax_mean.metric == "accuracy", "metric_maxdiff"].iloc[0])
    #meta_metric_row_dic["accuracy_balanced_min_max_diff_mean"].append(df_meta_metrics_diffmax_mean.loc[df_meta_metrics_diffmax_mean.metric == "accuracy_balanced", "metric_maxdiff"].iloc[0])
    # mean meta-metrics
    meta_metric_row_dic["f1_macro_meta_mean"].append(df_meta_metrics_mean_mean.loc[df_meta_metrics_mean_mean.metric == "f1_macro", "metric_mean"].iloc[0])
    meta_metric_row_dic["accuracy_meta_mean"].append(df_meta_metrics_mean_mean.loc[df_meta_metrics_mean_mean.metric == "accuracy", "metric_mean"].iloc[0])
    # overall metrics
    meta_metric_row_dic["f1_macro_overall"].append(df_metrics_overall.loc[df_metrics_overall.metric_name == "f1_macro", "metric_values"].iloc[0])
    meta_metric_row_dic["accuracy_overall"].append(df_metrics_overall.loc[df_metrics_overall.metric_name == "accuracy", "metric_values"].iloc[0])

    # !! stop of 1 indentation for loop


## good summary dic with maint results for each classifier
meta_metric_per_classifier_dic





###### create meta-metric df to merge with correlation df
# to investigate between meta-metric std and correlation validity performance
# keys for merging: vectorizer, size, langs, model
# ?? What would the results show? 'more even performance on parfam_text increases probability of correlation between distribution of predicted data and crowd data on same variable (parfam_text)'

## create meta-metrics df to merge with correlation df
df_meta_metrics_for_corr = pd.DataFrame(meta_metric_row_dic)

# load raw correlation df from multilingual paper
df_corr = pd.read_excel("/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/results/viz-a2/df_results_pimpo_raw.xlsx")
# remove rows for sample-extrapolation test
df_corr = df_corr[~df_corr.accuracy.isna()].drop(columns=["Unnamed: 0"])
df_corr_cl = df_corr[['meta data', 'algorithm', 'language representation', 'algorithm size',
                       'training languages', 'average correlation', 'average p-value',
                       #'F1 macro', 'accuracy', #'corr_labels_all', 'p-value-labels_all',
                       #'corr_immigration_neutral', 'p-value-immigration_neutral',
                       #'corr_immigration_sceptical', 'p-value-immigration_sceptical',
                       #'corr_immigration_supportive', 'p-value-immigration_supportive',
                       #'precision_macro', 'recall_macro', 'precision_micro', 'recall_micro'
                      ]]

## merge
df_corr_meta = df_corr_cl.merge(df_meta_metrics_for_corr, on=["meta data", "algorithm", "language representation",
                                            "algorithm size", "training languages"], how="right")

### correlate meta-metrics (std) with correlation r performance
from scipy.stats import pearsonr

meta_metric_cols_for_correlation = [key for key in meta_metric_row_dic if key not in ["meta data", "algorithm", "language representation", "algorithm size", "training languages"]]
pearsonr_dic = {}
for col in meta_metric_cols_for_correlation:
    pearson_cor = pearsonr(df_corr_meta["average correlation"], df_corr_meta[col])
    pearsonr_dic.update({col: (round(pearson_cor[0], 3), round(pearson_cor[1], 3))})
    print(col, pearson_cor)

# display results in one df
df_corr_parfamdistribution_meta_metrics = pd.DataFrame(pearsonr_dic, index=["r", "p"])
df_corr_parfamdistribution_meta_metrics




#### insights on differences between algos
# against my expectation, NLI has a higher std and minmax-diff than embed+log-reg and (undertuned) tfidf
# puts hypo into question that correlation performance is linked to more harmonious cross-group performance of NLI and content anchoring
# but need to investigate this more: differentiate by groups, do systematic correlation comparison of r and std across models, try different dataset (r performance seems spurious here)

## update from correlation analysis:
# values change a bit (but not much) when calculating means across more meta-data groups. trends are the same
# the best predictor seems to be simply F1-macro; mean meta-metrics also good correlation, but not better
# regarding variance: only accuracy with std and maxdiff also provides a significant predictor, but also not better than simple F1-macro

# Why is std on f1-macro not predictive at all and std on accuracy is?
# maybe the groups are randomly~ class imbalanced, which makes them very sensitive to f1-macro, while accuracy does not care about imbalance?

# ! should double check if mistakes made somewhere


#### manually inspect groups with issues
### insights
# quasi-sentence splitting causes issues: misclassifications caused by very short quasi-sentences, where e.g. single words like "ethnic" are annotated (in context
#   would be better to just have full sentences. (negative effect of CMP's need to have only one label per sequence)
# distinction between integration/immigration causes issues. I should maybe rather merge the two than convert one to "no-topic".
#   Found sentences where it seemed reasonable to classify as "immigration" if the "integration" class didn't implicitly exist
# noisy gold standard: there are several misclassifications or unclear cases in the gold standard
# ! hard to find issues that are specific to a group like swe or left;  would take deeper qualitative, comparative analyses

# inspect misclassifications for problematic group
#df_issues = df_cl[(df_cl.country_iso == "swe") & (df_cl.label != df_cl.label_pred)]




#### create summary dict for relevant meta-metric analysis results and write to disk


## write analysis dics for selected classifiers
"""classifiers_selected = ["NLI-Transformer_en_large_en-de", "Sent.-Transformer_multi_base_en", "Transformer_en_large_en-de"]

for classifier in classifiers_selected:

    meta_metrics_analysis_dic = {**meta_metric_per_classifier_dic[classifier]}

    ## write to disk
    with pd.ExcelWriter(f"./data-analysed/df_meta_metrics_pimpo_{classifier}.xlsx", engine='xlsxwriter') as writer:
        for key_df_name, value_df in meta_metrics_analysis_dic.items():
            value_df.to_excel(writer, sheet_name=key_df_name)
        writer.save()
"""

## write the overall convergent validity correlation df
#df_corr_parfamdistribution_meta_metrics.to_excel(f"./data-analysed/df_pimpo_corr_parfamdistribution_meta_metrics.xlsx")



