
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
N_SAMPLE = "1000"
N_ITER = "20"
DATE = 20230207
MODEL_NAME = "DeBERTa-v3-base"  #"logistic"  #"DeBERTa-v3-base"
META_DATA = "parfam_text"  #["parfam_text", "country_iso", "language_iso", "decade"]
NORMALIZE = True

# TODO: need to rerun everything on full 200k texts to get the aggregate numbers right (otherwise no-topic underestimated, all crowd label counts distorted etc.)
# TODO: rerun everything with 2+2 surrounding sentences, like in paper; (generally look at pre-processing again to try and increase data quality)
# TODO: for random samples: make average over 5 classifiers work

### load data
#df_cl, _df_train = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
df = pd.read_csv(f"./data-classified/pimpo/df_pimpo_{N_SAMPLE}_{MODEL_NAME}_{DATE}_{N_ITER}.zip")

### preprocessing
# add meta-data
df["decade"] = df["date"].apply(lambda x: str(x)[:3] + "0" if int(str(x)[3]) < 5 else str(x)[:3] + "5")

# remove rows which were in training data
df_train = df[df.label_text_pred.isna()]
df_cl = df[~df.label_text_pred.isna()]

# create subset df for selected suspicious group (based on meta-metrics)
selected_meta = "country_iso"
selected_meta_group = "dnk"  # "swe", "dnk"
df_meta_selected = df_cl[df_cl[selected_meta] == selected_meta_group]



#### 1. Hypo: salience of migration vs. other topics
# Hypothesis: migration should be very salient compared to other topics
"""p 1071: The intensity of the current public debate on immigration might give the impression that it is one of the most important topics. A valid measure should thus reflect this expectation in its saliency scores. A first glance at descriptive statistics shows that, on average, less than 5 per cent of quasi-sentences in the manifestos are about immigration or immigrant integration, which appears to be fairly low. However, if one compares this value to the saliency values of other topics from the Manifesto Project the picture changes.23 If we look at topics in approximately the same time frame as our dataset, only one topic is, on average, much more important than immigration: the positive category for the welfare state. This category accounts on average for nearly 10 per cent of quasi-sentences in the manifestos. For all other categories the average is around or below 5 per cent. Put in context, our findings of an average saliency of 5 per cent are not low, but rather high; Ruedin and Morales’ (2017) findings point in a similar direction. The topic is most salient in Denmark (see also Green-Pedersen & Otjes 2017), where parties devote an average of about 10 per cent of their manifestos to immigration/immigrant integration issues. This corresponds to findings from Benoit and Laver’s (2006: 158) expert survey, which presents Denmark as one of the countries where immigration was the most important policy dimension."""
# TODO: need to rerun everything (translation + lemmatization) on full corpus with all no-topic. otherwise no-topic gets underestimated and findings on salience are distorted
# I find 15%~ are about migration/integration in gold data, while paper mentions 5%.

## get label names for CMP codes/labels
# translating cmp label codes to label text with codebook mapping. MPDS2020a-1, see codebook https://manifesto-project.wzb.eu/down/papers/handbook_2011_version_4.pdf
# Note that the "main" codes are from v4 for backwards compatibility with older data. for new v5 categories: everything was aggregated up into the old v4 categories, except for 202.2, 605.2 und 703.2, which where added to 000.
df_cmp_label_map = pd.read_csv("./data-raw/codebook_categories_MPDS2020a-1.csv")
#.domain_name = df_cmp_label_map.domain_name.fillna("No other category applies")  # for some reason domain_name in case of no label is NaN. replace with expressive string
# translating label codes to label text with codebook mapping
# info on two column cmp_codes (v5 codebook) and cmp_code_hb4 (v4 codebook - backwardscompatible): "Außerdem enthält die Spalte cmp_code jetzt einfach die unmodifizierten original cmp_codes (also auch die neuen handbuch 5 Kategorien, wo sie angewendet wurden). Dafür gibt es jetzt cmp_code_hb4, in der dann alles in hb4 umgewandelt wurde (also 605.2 zu "000", 202.2 zu "000" und 703.2 zu "000", alle übrigen 5er Kategorien hochaggregiert)
# mapping of numeric codes to domain and subcat titles. only use v4 codebook numeric codes with XX.0 floats, ignore XX.1 codes from codebook because not present in masterfile shared by Tobias due to backwords compatibility
code_to_domain_map = {int(row["code"]): row["domain_name"] for i, row in df_cmp_label_map.iterrows() if str(row["code"])[-1] == "0"}  # only take labels which don't have old sub category. old subcategories indicated by XX.1 floats, main categories indicated by XX.0 floats
code_to_subcat_map = {int(row["code"]): row["title"] for i, row in df_cmp_label_map.iterrows() if str(row["code"])[-1] == "0"}
# try removing "no other category applies" category to align topic count with pimpo paper (they get higher counts for welfare state expansion)
code_to_subcat_map.update({0: np.nan})
# labels were name changed from v4 to v5 - but not changing it because working with v4.
df_cl["label_cmp_domain_text"] = df_cl.cmp_code.astype(int).map(code_to_domain_map)
df_cl["label_cmp_subcat_text"] = df_cl.cmp_code.astype(int).map(code_to_subcat_map)
# do same for tests with selected suspicious data
df_meta_selected["label_cmp_domain_text"] = df_meta_selected.cmp_code.astype(int).map(code_to_domain_map)
df_meta_selected["label_cmp_subcat_text"] = df_meta_selected.cmp_code.astype(int).map(code_to_subcat_map)

## compare most frequent CMP labels with migration labels
# overall salience
df_issue_salience = pd.DataFrame({
    "migration_label_text": df_cl.label_text.unique().tolist(),
    "migration_freq_gold": df_cl.label_text.value_counts(normalize=True)[df_cl.label_text.unique()].tolist(),
    "migration_freq_pred": df_cl.label_text_pred.value_counts(normalize=True)[df_cl.label_text.unique()].tolist(),
    "cmp_label_text": df_cl.label_cmp_subcat_text.value_counts(normalize=True, ascending=True).index[-len(df_cl.label_text.unique()):].tolist(),
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
# findings summary from multiling paper results
# classifiers indicate overall direction correctly (migration very salient), but strongly overestimate its salience compared to crowd

### classical, 1k: df_pimpo_1000_logistic_20230207_0.zip

### NLI active learning 1k, 20 iter
# quite good!



#### 2. Hypo: extreme-right talks more about migration overall
"""p. 1071:  We also expect to see clear differences in positions and saliency between party families. In particular, radical right parties are believed to spend more time talking about immigration and immigrant integration than their mainstream competitors. This expectation is confirmed by our data: while all party families dedicate less than 10 per cent of their manifesto to these topics, radical right parties place a much larger emphasis on these topics than their competitors, and use more than 10 per cent of their manifestos to talk about these topics. These descriptive findings are in line with research stressing that the radical right differentiates itself from the mainstream by its emphasis on immigration and immigrant integration and tries to put it on the larger public agenda (see, e.g., Ivarsflaten 2008; Mudde"""

df_viz_pred_counts_w_notopic = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA)
df_viz_true_counts_w_notopic = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA)
df_meta_proportions_w_notopic = merge_true_pred(df_viz_true_counts=df_viz_true_counts_w_notopic, df_viz_pred_counts=df_viz_pred_counts_w_notopic, meta_data=META_DATA)
# df with only proportion on migration per parfam
df_meta_proportions_w_notopic_migration = df_meta_proportions_w_notopic[df_meta_proportions_w_notopic.label_text == "no_topic"]
df_meta_proportions_w_notopic_migration[["label_count_true", "label_count_pred"]] = (df_meta_proportions_w_notopic_migration[["label_count_true", "label_count_pred"]] - 1) * -1
df_meta_proportions_w_notopic_migration["label_text"] = "about_migration"
df_meta_proportions_w_notopic_migration = df_meta_proportions_w_notopic_migration.sort_values("label_count_true", ascending=False)

# for selected country
df_viz_pred_counts_selected_w_notopic = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text_pred", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA)
df_viz_true_counts_selected_w_notopic = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text", exclude_no_topic=False, normalize=NORMALIZE, meta_data=META_DATA)
df_meta_proportions_selected_w_notopic = merge_true_pred(df_viz_true_counts=df_viz_true_counts_selected_w_notopic, df_viz_pred_counts=df_viz_pred_counts_selected_w_notopic, meta_data=META_DATA)


#### 2. Hypo interpretation
# findings summary from multiling paper results:
# classifiers indicate directions correctly for NAT, but overestimate migration salience compared to NAT

### classical, 1k: df_pimpo_1000_logistic_20230207_0.zip

### NLI active learning 1k, 20 iter
# quite good!



#### 3. Hypo: left more pro migration; right more con migration

## calculate aggregate label frequencies per meta-data group
df_viz_pred_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred", exclude_no_topic=True, normalize=NORMALIZE, meta_data=META_DATA)
df_viz_true_counts = calculate_distribution(df_func=df_cl, df_label_col="label_text", exclude_no_topic=True, normalize=NORMALIZE, meta_data=META_DATA)
df_meta_proportions = merge_true_pred(df_viz_true_counts=df_viz_true_counts, df_viz_pred_counts=df_viz_pred_counts, meta_data=META_DATA)
# calculate proportions for selected country
df_viz_pred_counts_selected = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text_pred", exclude_no_topic=True, normalize=NORMALIZE, meta_data=META_DATA)
df_viz_true_counts_selected = calculate_distribution(df_func=df_meta_selected, df_label_col="label_text", exclude_no_topic=True, normalize=NORMALIZE, meta_data=META_DATA)
df_meta_proportions_selected = merge_true_pred(df_viz_true_counts=df_viz_true_counts_selected, df_viz_pred_counts=df_viz_pred_counts_selected, meta_data=META_DATA)

## calculate difference between truth and prediction
# correlation can miss this absolute difference
df_meta_proportions["label_count_difference"] = df_meta_proportions.label_count_true - df_meta_proportions.label_count_pred
# average difference between true and pred proportions
print("Average difference between true and pred proportions: ", df_meta_proportions["label_count_difference"].abs().mean())
print("Average difference between true and pred proportions (without neutral): ", df_meta_proportions[~df_meta_proportions.label_text.str.contains("neutral")]["label_count_difference"].abs().mean())
# same for selected group
df_meta_proportions_selected["label_count_difference"] = df_meta_proportions_selected.label_count_true - df_meta_proportions_selected.label_count_pred
# average difference between true and pred proportions
print("Average difference between true and pred proportions for selected group: ", df_meta_proportions_selected["label_count_difference"].abs().mean())
print("Average difference between true and pred proportions for selected group (without neutral): ", df_meta_proportions_selected[~df_meta_proportions_selected.label_text.str.contains("neutral")]["label_count_difference"].abs().mean())

## label text aggregation to simple left/right/other instead of full parfam_text
parfam_aggreg_map = {"ECO": "left", "LEF": "left", "SOC": "left",
                     "CHR": "right", "CON": "right", "NAT": "right",
                     "LIB": "other", "AGR": "other", "ETH": "other", "SIP": "other"}
# for all countries
df_meta_proportions["parfam_text_aggreg"] = df_meta_proportions.parfam_text.map(parfam_aggreg_map)
# aggregate by parfam and positions
df_meta_proportions_leftright = df_meta_proportions.groupby(["parfam_text_aggreg", "label_text"], as_index=False, group_keys=False).apply(lambda x: x.mean())
df_meta_proportions_leftright = df_meta_proportions_leftright.sort_values(["parfam_text_aggreg", "label_count_difference"])

## simplified merge of pro/con position (ignoring integration vs. immigration)
procon_map = {'integration_sceptical': "sceptical", 'immigration_sceptical': "sceptical",
       'immigration_supportive': "supportive", 'integration_neutral': "neutral",
       'immigration_neutral': "neutral", 'integration_supportive': "supportive"
}
df_meta_proportions_leftright["label_text_aggreg"] = df_meta_proportions_leftright.label_text.map(procon_map)
df_meta_proportions_leftright_procon = df_meta_proportions_leftright.groupby(["parfam_text_aggreg", "label_text_aggreg"], as_index=False, group_keys=False).apply(lambda x: x.sum(numeric_only=True))
# merge of pro/con positions for each individual party family
df_meta_proportions["label_text_aggreg"] = df_meta_proportions.label_text.map(procon_map)
df_meta_proportions_procon = df_meta_proportions.groupby(["parfam_text", "label_text_aggreg"], as_index=False, group_keys=False).apply(lambda x: x.sum(numeric_only=True))

## overall very simplified left/right pro/con
df_meta_proportions_leftright_procon = df_meta_proportions_leftright.groupby(by=["parfam_text_aggreg", "label_text_aggreg"], as_index=False, group_keys=False).apply(lambda x: x.sum(numeric_only=True))



# average difference between true and pred proportions
print("Average difference between true and pred proportions for simplified left/right/others: ", df_meta_proportions_leftright["label_count_difference"].abs().mean())
print("Average difference between true and pred proportions for simplified left/right/others (without neutral): ", df_meta_proportions_leftright[~df_meta_proportions_leftright.label_text.str.contains("neutral")]["label_count_difference"].abs().mean())

## for selected suspicious meta-data
df_meta_proportions_selected["parfam_text_aggreg"] = df_meta_proportions_selected.parfam_text.map(parfam_aggreg_map)
# aggregate by parfam and stances for selected group
df_meta_proportions_leftright_selected = df_meta_proportions_selected.groupby(["parfam_text_aggreg", "label_text"], as_index=False, group_keys=False).apply(lambda x: x.mean())
df_meta_proportions_leftright_selected = df_meta_proportions_leftright_selected.sort_values(["parfam_text_aggreg", "label_count_difference"])
# average difference between true and pred proportions
print("Average difference between true and pred proportions for simplified left/right/others: ", df_meta_proportions_leftright_selected["label_count_difference"].abs().mean())
print("Average difference between true and pred proportions for simplified left/right/others (without neutral): ", df_meta_proportions_leftright_selected[~df_meta_proportions_leftright_selected.label_text.str.contains("neutral")]["label_count_difference"].abs().mean())


### visualisation
def create_figure(df_counts_func=None, label_count_col="label_count_pred", meta_data=None, show_legend=True):
    x_axis = []
    data_dic = {label: [] for label in df_counts_func.label_text.unique()}
    for group_key, group_df_viz in df_counts_func.groupby(by=meta_data):
        x_axis.append(group_key)
        # append label count for each label to data_dic with respective label key
        for key_label_text in data_dic:
            data_dic[key_label_text].append(group_df_viz[group_df_viz["label_text"] == key_label_text][label_count_col].iloc[0])

    # order of labels
    #label_order = ["immigration_sceptical", "immigration_neutral", "immigration_supportive"]
    label_order = ["immigration_sceptical", "integration_sceptical", "immigration_neutral", "integration_neutral", "immigration_supportive", "integration_supportive"]
    data_dic = {k: data_dic[k] for k in label_order}

    fig = go.Figure()
    # hex color picker: https://www.w3schools.com/colors/colors_picker.asp
    colors_dic = {"immigration_neutral": "#3b62c4", "immigration_sceptical": "#e63333", "immigration_supportive": "#1f9134",
                  "integration_neutral": "#4e71ca", "integration_sceptical": "#e94949", "integration_supportive": "#24a83c",
    }  # order: neutral, sceptical,  supportive
    for key_label_text in data_dic:
        fig.add_bar(x=x_axis, y=data_dic[key_label_text], name=key_label_text, marker_color=colors_dic[key_label_text],
                    showlegend=show_legend)

    fig.update_layout(barmode="relative", title=f"True")
    fig.update_traces(marker_line_width=0)  # removes white separator line between stacked bars

    return fig


fig_pred = create_figure(df_counts_func=df_meta_proportions, label_count_col="label_count_pred", meta_data=META_DATA, show_legend=False)  # df_label_col="label_text_pred"
fig_true = create_figure(df_counts_func=df_meta_proportions, label_count_col="label_count_true", meta_data=META_DATA, show_legend=True)  # df_label_col="label_text"
#fig_pred.update_layout(barmode="relative", title=f"Predicted - trained on {N_SAMPLE} texts")
# add left/right aggregated
#fig_pred_leftright = create_figure(df_counts_func=df_meta_proportions_leftright, label_count_col="label_count_pred", meta_data=META_DATA+"_aggreg", show_legend=False)  # df_label_col="label_text_pred"
#fig_true_leftright = create_figure(df_counts_func=df_meta_proportions_leftright, label_count_col="label_count_true", meta_data=META_DATA+"_aggreg", show_legend=False)  # df_label_col="label_text"

## make comparative subplots
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/

subplot_titles = ["Ground truth from PImPo dataset (by crowd)", f"Prediction by classifier",
                  #"Ground truth simplified", "Prediction simplified"
                  ]
fig_subplot = make_subplots(rows=1, cols=2,  # start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                            subplot_titles=subplot_titles,
                            x_title="Party families" , y_title="Proportion of stances towards immigration in corpus")

fig_subplot.add_traces(fig_true["data"], rows=1, cols=1)
fig_subplot.add_traces(fig_pred["data"], rows=1, cols=2)
# add left/righ aggregated
#fig_subplot.add_traces(fig_pred_leftright["data"], rows=2, cols=1)
#fig_subplot.add_traces(fig_true_leftright["data"], rows=2, cols=2)
# update layout
fig_subplot.update_layout(barmode="relative", title=f"Comparison of true and predicted distribution of stances by party family",
                          title_x=0.5, legend={'traceorder': 'reversed'}, template="ggplot2")

## add correlation to figure
from scipy.stats import pearsonr
#pearson_cor_parfam = pearsonr(df_meta_proportions.label_count_true, df_meta_proportions.label_count_pred)
# calculate separate correlations for labels to avoid interdependency in array values
corr_parfam_mean = df_meta_proportions.groupby("label_text", as_index=False, group_keys=False).apply(lambda x: pearsonr(x.label_count_true, x.label_count_pred)[0]).mean()[0]
p_parfam_mean = df_meta_proportions.groupby("label_text", as_index=False, group_keys=False).apply(lambda x: pearsonr(x.label_count_true, x.label_count_pred)[1]).mean()[0]

fig_subplot.add_annotation(
    dict(
        font=dict(color='black',size=20),
        #x="irl",
        y=0.9,
        showarrow=False,
        #text=f"r: {round(pearson_cor_parfam[0], 3):.2f}   p: {round(pearson_cor_parfam[1], 4):.3f}",
        text=f"r: {round(corr_parfam_mean, 3):.2f}   p: {round(p_parfam_mean, 4):.3f}",
        textangle=0,
    )
)

# show
fig_subplot.show(renderer="browser")



#### Hypo 3: interpretation

### classical, 1k: df_pimpo_1000_logistic_20230207_0.zip
# important issues

### NLI active learning 1k, 20 iter
# df_meta_proportions_leftright_procon: gets overall pro/con tendency for left vs right correct
# df_meta_proportions: for specific parfam and positions, still quite far off (10%), for many others quite close
# df_meta_proportions_leftright: overall better, still issues distinguishing integration vs. immigration
# df_meta_proportions_procon: for simplified pro/con without integration vs. immigration: overall better, but relevant issues for some party families




#### 4. Hypo: differences between settler; emigration; receiving states
"""p. 1072: The migration literature has for a long time stressed differences between European and settler countries, though this gap is hypothesised to have become smaller over the last decade (for a good discussion, see Dauvergne 2016: Chapters 1 and 2). Our data still shows remarkable differences between Europe and the Anglo-Saxon world. Parties in settler countries address immigration differently from those in Continental Europe. Figure 5 shows how the parties in each type of country divided their attention between immigration and immigrant integration. The traditional settler countries and the former emigration countries focus more on immigration than on immigrant integration, while the postwar recipient states in Western Europe are more concerned with questions of immigrant integration. The focus on immigrant integration is, however, less pronounced for the countries with a longer history of radical right party presence, and in the traditional settler countries. In fact, a number of parties in these countries do not talk about integration issues at all."""
#...

## add simplified label_text label for migration vs. integration
df_cl["label_text_simple"] = ["immigration" if "immigration" in label_text else "integration" if "integration" in label_text else label_text for label_text in df_cl.label_text]
df_cl["label_text_pred_simple"] = ["immigration" if "immigration" in label_text else "integration" if "integration" in label_text else label_text for label_text in df_cl.label_text_pred]
df_cl["label_text_simple"].value_counts()


# TODO: trying to count manifestos that mention neither integration or immigration
# TODO: (not sure if relevant) should probably always do dropna=False in groupby? Not sure if groupby otherwise drops some relevant rows
#test = df_cl[df_cl["label_text_pred_simple"] != "no_topic"]
# df_cl = df.copy(deep=True)
test2 = df_cl.groupby(by=["country_iso", "partyname"], as_index=True, group_keys=True, dropna=False).apply(lambda x: x["label_text_simple"].value_counts(normalize=True, dropna=False))
# 100% no-topic shows if party did not talk about migration at all
# potential solution: seems decently visualised by including no-topic in code below

## calculate aggregate label frequencies per country
df_viz_pred_counts_country = calculate_distribution(df_func=df_cl, df_label_col="label_text_pred_simple", exclude_no_topic=False, normalize=NORMALIZE, dropna=True, meta_data="country_iso")
df_viz_true_counts_country = calculate_distribution(df_func=df_cl, df_label_col="label_text_simple", exclude_no_topic=False, normalize=NORMALIZE, dropna=True, meta_data="country_iso")
df_meta_proportions_country = merge_true_pred(df_viz_true_counts=df_viz_true_counts_country, df_viz_pred_counts=df_viz_pred_counts_country, meta_data="country_iso")
# reorder countries in theoretically informed order from paper
countries_order = ["esp", "irl", "swe", "fin", "nor", "deu", "nld", "che", "aut", "dnk", "usa", "can", "aus", "nzl"]
df_meta_proportions_country["country_iso"] = pd.Categorical(df_meta_proportions_country["country_iso"], countries_order)
df_meta_proportions_country = df_meta_proportions_country.sort_values(["country_iso", "label_text"]).reset_index(drop=True)

# pimpo paper somehow includes info in figure 5 that some parties do not talk about migration/integration at all
#test = df_cl.groupby(by=["country_iso", "partyname"], as_index=True, group_keys=True).apply(lambda x: x["label_text_simple"].value_counts(normalize=True, dropna=False))
# TODO: fix in entire script: my current method for aggregation gives each sentence the same weight, which overestimates parties that have long manifestos

### visualisation
def figure_countrytypes(df_counts_func=None, label_count_col="label_count_pred", meta_data=None, show_legend=True):
    x_axis = []
    data_dic = {label: [] for label in df_counts_func.label_text.unique()}
    for group_key, group_df_viz in df_counts_func.groupby(by=meta_data):
        x_axis.append(group_key)
        # append label count for each label to data_dic with respective label key
        for key_label_text in data_dic:
            data_dic[key_label_text].append(group_df_viz[group_df_viz["label_text"] == key_label_text][label_count_col].iloc[0])

    # order of labels
    label_order = ["integration", "immigration"]
    data_dic = {k: data_dic[k] for k in label_order}

    fig = go.Figure()
    # hex color picker: https://www.w3schools.com/colors/colors_picker.asp
    colors_dic = {"immigration": "#1f9134", "integration": "#3b62c4"}
    for key_label_text in data_dic:
        fig.add_bar(x=x_axis, y=data_dic[key_label_text], name=key_label_text, marker_color=colors_dic[key_label_text],
                    showlegend=show_legend)

    fig.update_layout(barmode="relative", title=f"True")
    fig.update_traces(marker_line_width=0)  # removes white separator line between stacked bars

    return fig


fig_pred_countrytypes = figure_countrytypes(df_counts_func=df_meta_proportions_country, label_count_col="label_count_pred", meta_data="country_iso", show_legend=False)  # df_label_col="label_text_pred"
fig_true_countrytypes = figure_countrytypes(df_counts_func=df_meta_proportions_country, label_count_col="label_count_true", meta_data="country_iso", show_legend=True)  # df_label_col="label_text"
#fig_pred_countrytypes.update_layout(barmode="relative", title=f"Predicted - trained on {N_SAMPLE} texts")

# trying to create different subplots for countrytypes
#countries_order_dic = {"emigration": ["esp", "irl"], "receiver": ["swe", "fin", "nor", "deu", "nld", "che", "aut", "dnk"], "settler": ["usa", "can", "aus", "nzl"]}
#fig_countrytypes_dic = {"crowd": [], "prediction": []}
#for countrytype in ["emigration", "receiver", "settler"]:
#    fig_countrytypes_dic["crowd"].append(figure_countrytypes(df_counts_func=df_meta_proportions_country[df_meta_proportions_country.country_iso.isin(countries_order_dic[countrytype])].reset_index(drop=True), label_count_col="label_count_pred", meta_data="country_iso", show_legend=False))  # df_label_col="label_text_pred"
#    fig_countrytypes_dic["prediction"].append(figure_countrytypes(df_counts_func=df_meta_proportions_country[df_meta_proportions_country.country_iso.isin(countries_order_dic[countrytype])].reset_index(drop=True), label_count_col="label_count_true", meta_data="country_iso", show_legend=False))  # df_label_col="label_text"


### make comparative subplots
from plotly.subplots import make_subplots  # https://plotly.com/python/subplots/

subplot_titles_countrytypes = ["Ground truth from PImPo dataset (by crowd)", f"Prediction by classifier"]
fig_subplot_countrytypes = make_subplots(rows=1, cols=2,  # start_cell="top-left", horizontal_spacing=0.1, vertical_spacing=0.2,
                            subplot_titles=subplot_titles_countrytypes,
                            x_title="Party families" , y_title="Proportion of stances towards immigration in corpus")

fig_subplot_countrytypes.add_traces(fig_true_countrytypes["data"], rows=1, cols=1)
fig_subplot_countrytypes.add_traces(fig_pred_countrytypes["data"], rows=1, cols=2)

# update layout
fig_subplot_countrytypes.update_layout(barmode="relative", title=f"Comparison of true and predicted distribution of stances by party family",
                          title_x=0.5, legend={'traceorder': 'reversed'}, template="ggplot2")

## add correlation to figure
from scipy.stats import pearsonr
# calculate separate correlations for labels to avoid interdependency in array values
corr_country_mean = df_meta_proportions_country.groupby("label_text", as_index=False, group_keys=False).apply(lambda x: pearsonr(x.label_count_true, x.label_count_pred)[0]).mean()[0]
p_country_mean = df_meta_proportions_country.groupby("label_text", as_index=False, group_keys=False).apply(lambda x: pearsonr(x.label_count_true, x.label_count_pred)[1]).mean()[0]

fig_subplot_countrytypes.add_annotation(
    dict(
        font=dict(color='black',size=20),
        #x="irl",
        y=0.25,
        showarrow=False,
        text=f"r: {round(corr_country_mean, 3):.2f}   p: {round(p_country_mean, 4):.3f}",
        textangle=0,
    )
)

# show
fig_subplot_countrytypes.show(renderer="browser")


#### 4. Hypo: interpretation

### classical, 1k: df_pimpo_1000_logistic_20230207_0.zip
# gets tendency surprisingly well, some small deviations




##### Other tests

### deletable: analyse sampling from active learner
"""test0 = df_train.groupby(by="parfam_text", group_keys=True, as_index=True).apply(
    lambda x: x.label_text.value_counts()#.sum()
)
test = df_train.parfam_text.value_counts()
test1 = df_train.parfam_text.value_counts(normalize=True)
test2 = df_cl.parfam_text.value_counts()
test3 = df_cl.parfam_text.value_counts(normalize=True)
# looking at al n_sample per parfam, vs. total_n per parfam
# !! => al actually oversamples NAT, it is confused about it. Equal sampling per parfam would probably be detrimental help
test4 = pd.concat([test, test1, test3, test2], axis=1)


#### cleanlab tests
# https://docs.cleanlab.ai/stable/index.html
import cleanlab
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning

import ast
label_probs = np.array([ast.literal_eval(lst) for lst in df_cl.label_probabilities.astype('object')])

# https://docs.cleanlab.ai/stable/cleanlab/filter.html
ranked_label_issues = find_label_issues(
    labels=df_cl.label,
    pred_probs=label_probs,
    return_indices_ranked_by="self_confidence",
)

df_cl_issues = df_cl.reindex(ranked_label_issues)

test = df_train[["text_prepared", "label_text", "parfam_text"]]"""






print("Run done.")














