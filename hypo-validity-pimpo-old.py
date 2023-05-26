
## load libraries
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers_a2
import importlib  # in case of manual updates in .py file
importlib.reload(helpers_a2)

from helpers_a2 import load_data, calculate_distribution, compute_correlation, compute_metrics_standard, merge_true_pred


## select analysed dataset
DATASET = "pimpo"
TASK = "immigration"
METHOD = "dl_embed"  # nli, dl_embed, classical_ml
MODEL_SIZE = "classical"  # classical, base, large
HYPOTHESIS = "long"
VECTORIZER = "en"  # en, multi, tfidf
MAX_SAMPLE_LANG = 500
DATE = 221111  # 230127, 221111
LANGUAGES = ["en", "de"]
META_DATA = "parfam_text"  #["parfam_text", "country_iso", "language_iso", "decade"]
langs_concat = "_".join(LANGUAGES)
NORMALIZE = True
EXCLUDE_NO_TOPIC = True
# df_pimpo_pred_immigration_nli_large_long_en_500samp_en_de_sv_fr_221111, df_pimpo_pred_immigration_dl_embed_classical_long_multi_500samp_en_221111
#df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.zip")
#df = pd.read_csv("./data-classified/pimpo/df_pimpo_pred_immigration_classical_ml_None_long_tfidf_500samp_en_221111.zip")

# load data
df_cl, df_train = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
# process data
df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_meta_proportions = merge_true_pred(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts, meta_data=META_DATA)
#metrics_dic = compute_metrics_standard(label_gold=df_cl.label, label_pred=df_cl.label_pred)

## subset data for problematic group
selected_meta = "country_iso"
selected_meta_group = "dnk"  # "swe", "dnk"
df_meta_selected = df_cl[df_cl[selected_meta] == selected_meta_group]
# calculate proportions for selected country
df_viz_pred_counts_selected = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text_pred", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_viz_true_counts_selected = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text", exclude_no_topic=EXCLUDE_NO_TOPIC, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_meta_proportions_selected = merge_true_pred(df_viz_true_counts=df_viz_true_counts_selected, df_viz_pred_counts=df_viz_pred_counts_selected, meta_data=META_DATA)


#### 1. Hypo: salience of migration vs. other topics
# Hypothesis: migration should be very salient compared to other topics
"""p 1071: The intensity of the current public debate on immigration might give the impression that it is one of the most important topics. A valid measure should thus reflect this expectation in its saliency scores. A first glance at descriptive statistics shows that, on average, less than 5 per cent of quasi-sentences in the manifestos are about immigration or immigrant integration, which appears to be fairly low. However, if one compares this value to the saliency values of other topics from the Manifesto Project the picture changes.23 If we look at topics in approximately the same time frame as our dataset, only one topic is, on average, much more important than immigration: the positive category for the welfare state. This category accounts on average for nearly 10 per cent of quasi-sentences in the manifestos. For all other categories the average is around or below 5 per cent. Put in context, our findings of an average saliency of 5 per cent are not low, but rather high; Ruedin and Morales’ (2017) findings point in a similar direction. The topic is most salient in Denmark (see also Green-Pedersen & Otjes 2017), where parties devote an average of about 10 per cent of their manifestos to immigration/immigrant integration issues. This corresponds to findings from Benoit and Laver’s (2006: 158) expert survey, which presents Denmark as one of the countries where immigration was the most important policy dimension."""
# ! Note that their hypo 1 is about migration + integration, while mine only about migration
# ! my relabeling of integration as no-topic probably biases the numbers for hypo 1 + 2

## get label names for CMP codes/labels
# translating cmp label codes to label text with codebook mapping. MPDS2020a-1, see codebook https://manifesto-project.wzb.eu/down/papers/handbook_2011_version_4.pdf
# Note that the "main" codes are from v4 for backwards compatibility with older data. for new v5 categories: everything was aggregated up into the old v4 categories, except for 202.2, 605.2 und 703.2, which where added to 000.
df_cmp_label_map = pd.read_csv("./data-raw/codebook_categories_MPDS2020a-1.csv")
df_cmp_label_map.domain_name = df_cmp_label_map.domain_name.fillna("No other category applies")  # for some reason domain_name in case of no label is NaN. replace with expressive string

# translating label codes to label text with codebook mapping
# info on two column cmp_codes (v5 codebook) and cmp_code_hb4 (v4 codebook - backwardscompatible): "Außerdem enthält die Spalte cmp_code jetzt einfach die unmodifizierten original cmp_codes (also auch die neuen handbuch 5 Kategorien, wo sie angewendet wurden). Dafür gibt es jetzt cmp_code_hb4, in der dann alles in hb4 umgewandelt wurde (also 605.2 zu "000", 202.2 zu "000" und 703.2 zu "000", alle übrigen 5er Kategorien hochaggregiert)
# mapping of numeric codes to domain and subcat titles. only use v4 codebook numeric codes with XX.0 floats, ignore XX.1 codes from codebook because not present in masterfile shared by Tobias due to backwords compatibility
code_to_domain_map = {int(row["code"]): row["domain_name"] for i, row in df_cmp_label_map.iterrows() if str(row["code"])[-1] == "0"}  # only take labels which don't have old sub category. old subcategories indicated by XX.1 floats, main categories indicated by XX.0 floats
code_to_subcat_map = {int(row["code"]): row["title"] for i, row in df_cmp_label_map.iterrows() if str(row["code"])[-1] == "0"}

# labels were name changed from v4 to v5 - but not changing it because working with v4.
df_cl["label_cmp_domain_text"] = df_cl.cmp_code.astype(int).map(code_to_domain_map)
df_cl["label_cmp_subcat_text"] = df_cl.cmp_code.astype(int).map(code_to_subcat_map)
df_meta_selected["label_cmp_domain_text"] = df_meta_selected.cmp_code.astype(int).map(code_to_domain_map)
df_meta_selected["label_cmp_subcat_text"] = df_meta_selected.cmp_code.astype(int).map(code_to_subcat_map)

## compare most frequent CMP labels with migration labels

# overall salience
df_issue_salience = pd.DataFrame({
    "migration_label_text": df_cl.label_text.unique().tolist(),
    "migration_freq_gold": df_cl.label_text.value_counts(normalize=True)[df_cl.label_text.unique()].tolist(),
    "migration_freq_pred": df_cl.label_text_pred.value_counts(normalize=True)[df_cl.label_text.unique()].tolist(),
    "cmp_label_text": df_cl.label_cmp_subcat_text.value_counts(normalize=True).index[:len(df_cl.label_text.unique())].tolist(),
    "cmp_top_freq": df_cl.label_cmp_subcat_text.value_counts(normalize=True, ascending=True)[-len(df_cl.label_text.unique()):].tolist(),
})
df_issue_salience = df_issue_salience.sort_values("migration_freq_gold", ascending=False)

# salience for problematic country
df_issue_salience_selected = pd.DataFrame({
    "migration_label_text": df_cl.label_text.unique().tolist(),
    "migration_freq_gold": df_meta_selected.label_text.value_counts(normalize=True)[df_cl.label_text.unique()].tolist(),
    "migration_freq_pred": df_meta_selected.label_text_pred.value_counts(normalize=True)[df_cl.label_text.unique()].tolist(),
    "cmp_label_text": df_meta_selected.label_cmp_subcat_text.value_counts(normalize=True).index[:len(df_cl.label_text.unique())].tolist(),
    "cmp_top_freq": df_meta_selected.label_cmp_subcat_text.value_counts(normalize=True, ascending=True)[-len(df_cl.label_text.unique()):].tolist(),
})
df_issue_salience_selected = df_issue_salience_selected.sort_values("migration_freq_gold", ascending=False)


#### 1. Hypo interpretation
# findings summary: classifiers indicate overall direction correctly (migration very salient), but strongly overestimate its salience compared to crowd

### classical, dl_embed, en_de, trans en, 500 max
# crowd says 6~% is about migration, classifier 12%. Strongly overestimates salience of migration
## for dnk
# crowd: 18~% about migration, classifier 24~%;   overall salience higher in dnk, which is in line with lit
## for swe
# crowd says 5~% is about migration, classifier 10%. Strongly overestimates salience of migration

### best NLI
# ! NLI actually slightly worse than other classifier. Is in line with finding that NLI overestimates minority and underestimates majority classes
# overall essentially same: # crowd says 6~% is about migration, classifier 12%. Strongly overestimates salience of migration
## for dnk
# crowd: 18~% about migration, classifier 25~%;   overall salience higher in dnk, which is in line with lit
## for swe
# crowd says 5~% is about migration, classifier 14%. Strongly overestimates salience of migration



#### 3. Hypo: extreme-right talks more about migration
df_viz_pred_counts_w_notopic = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_viz_true_counts_w_notopic = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_meta_proportions_w_notopic = merge_true_pred(df_viz_true_counts=df_viz_true_counts_w_notopic, df_viz_pred_counts=df_viz_pred_counts_w_notopic, meta_data=META_DATA)

# for selected country
df_viz_pred_counts_selected_w_notopic = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text_pred", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_viz_true_counts_selected_w_notopic = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA, algorithm=METHOD, representation=VECTORIZER, size=MODEL_SIZE, languages=LANGUAGES)
df_meta_proportions_selected_w_notopic = merge_true_pred(df_viz_true_counts=df_viz_true_counts_selected_w_notopic, df_viz_pred_counts=df_viz_pred_counts_selected_w_notopic, meta_data=META_DATA)


#### 3. Hypo interpretation
"""p. 1071:  We also expect to see clear differences in positions and saliency between party families. In particular, radical right parties are believed to spend more time talking about immigration and immigrant integration than their mainstream competitors. This expectation is confirmed by our data: while all party families dedicate less than 10 per cent of their manifesto to these topics, radical right parties place a much larger emphasis on these topics than their competitors, and use more than 10 per cent of their manifestos to talk about these topics. These descriptive findings are in line with research stressing that the radical right differentiates itself from the mainstream by its emphasis on immigration and immigrant integration and tries to put it on the larger public agenda (see, e.g., Ivarsflaten 2008; Mudde"""
# findings summary: classifiers indicate directions correctly for NAT, but overestimate migration salience compared to NAT

### classical, dl_embed, en_de, trans en, 500 max
# gold: NAT around 13% migration, others around 7-3%
# pred: NAT talk about migration 22%, others 14-10%
## dnk
# gold: NAT only 0.6 no topic, while other parties around 0.73-0.85
# pred: NAT only 0.5 no topic, while other parties around 0.65-0.86

### best NLI
# gold: NAT around 13% migration, others around 7-3%
# pred:  NAT talk about migration 22%, others 13-7%
## dnk
# gold: NAT only 0.6 no topic, while other parties around 0.73-0.85
# pred: NAT only 0.5 no topic, while other parties around 0.65-0.85





#### 4. Hypo: left more pro migration; right more con migration

## calculate difference between truth and prediction
# correlation can miss this absolute difference
df_meta_proportions["label_count_difference"] = df_meta_proportions.label_count_true - df_meta_proportions.label_count_pred
# average difference between true and pred proportions
df_meta_proportions["label_count_difference"].abs().mean()
# without immigration neutral
df_meta_proportions[~df_meta_proportions.label_text.str.contains("neutral")]["label_count_difference"].abs().mean()

# aggregate numbers for left/right parties overall
parfam_aggreg_map = {"ECO": "Left", "LEF": "Left", "SOC": "Left",
                     "CHR": "Right", "CON": "Right", "NAT": "Right",
                     "LIB": "Other", "AGR": "Other", "ETH": "Other", "SIP": "Other"}
df_meta_proportions["parfam_aggreg_text"] = df_meta_proportions.parfam_text.map(parfam_aggreg_map)
# aggregate by parfam and stances
df_meta_aggreg = df_meta_proportions.groupby(["parfam_aggreg_text", "label_text"], as_index=False, group_keys=False).apply(lambda x: x.mean())


## investigate specific countries based on bad meta-metric performance
df_meta_proportions_selected

## aggregation
df_meta_proportions_selected["parfam_aggreg_text"] = df_meta_proportions_selected.parfam_text.map(parfam_aggreg_map)
# aggregate by parfam and stances
df_meta_aggreg_selected = df_meta_proportions_selected.groupby(["parfam_aggreg_text", "label_text"], as_index=False, group_keys=False).apply(lambda x: x.mean())



### visualisation
def create_figure(df_func=None, df_counts_func=None, label_count_col="label_count_pred", show_legend=True):
    x_axis = []
    #data_dic = {label: [] for label in df_func.label_text.unique()}
    data_dic = {label: [] for label in df_counts_func.label_text.unique()}
    for group_key, group_df_viz in df_counts_func.groupby(by=META_DATA):
        x_axis.append(group_key)
        # append label count for each label to data_dic with respective label key
        for key_label_text in data_dic:
            data_dic[key_label_text].append(group_df_viz[group_df_viz["label_text"] == key_label_text][label_count_col].iloc[0])

    # order of labels
    label_order = ["immigration_sceptical", "immigration_neutral", "immigration_supportive"]
    data_dic = {k: data_dic[k] for k in label_order}

    fig = go.Figure()
    colors_dic = {"immigration_neutral": "#3F65C5", "immigration_sceptical": "#E63839", "immigration_supportive": "#1f9134"}  # order: neutral, sceptical,  supportive
    for key_label_text in data_dic:
        fig.add_bar(x=x_axis, y=data_dic[key_label_text], name=key_label_text, marker_color=colors_dic[key_label_text],
                    showlegend=show_legend)

    fig.update_layout(barmode="relative", title=f"True")

    return fig


language_str = "_".join(LANGUAGES)

fig_pred = create_figure(df_func=df_cl, df_counts_func=df_meta_proportions, label_count_col="label_count_pred", show_legend=False)  # df_label_col="label_text_pred"
fig_true = create_figure(df_func=df_cl, df_counts_func=df_meta_proportions, label_count_col="label_count_true", show_legend=True)  # df_label_col="label_text"
fig_pred.update_layout(barmode="relative", title=f"Predicted - trained on: {language_str} - Method: {METHOD}-{VECTORIZER}")

## try making subplots
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/

subplot_titles = ["Ground truth from PImPo dataset", f"Prediction by BERT-NLI"]
fig_subplot = make_subplots(rows=1, cols=2,  # start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                            subplot_titles=subplot_titles,
                            x_title="Party families" , y_title="Proportion of stances towards immigration in corpus")

fig_subplot.add_traces(fig_true["data"], rows=1, cols=1)
fig_subplot.add_traces(fig_pred["data"], rows=1, cols=2)
fig_subplot.update_layout(barmode="relative", title=f"Comparison of true and predicted distribution of stances towards {TASK} by party family",
                          title_x=0.5, legend={'traceorder': 'reversed'}, template="ggplot2")
fig_subplot.show(renderer="browser")




#### Hypo 4: interpretation

### classical, dl_embed, en_de, en, 500 max
## Overall
# makes ETH parties more sceptical than NAT, NAT less sceptical overall; makes LEF parties more sceptical than LIB
# is on average 10% off in predicted proportion. Makes left 8% more sceptical; makes right 14% more supportive; others 11% more sceptical
## Sweden
# makes left 24~% more sceptical; right 20% more supportive; flips results for nationalists 75-25 supportive instead of actually sceptical the other way round (very little data)
## Denmark
# makes left 35~% more supportive; right 30~% more supportive; makes NAT slightly supportive instead of clearly sceptical

print("Run done.")














