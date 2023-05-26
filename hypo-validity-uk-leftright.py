
## load libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr


## load data
DATASET = "uk-leftright-econ"
MAX_SAMPLE = 500
DATE = "20230207"
EXTERNAL_DATA = "surveys"
INTERNAL_DATA = "classifier"  # classifier, experts, surveys, crowd
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
df_iter_cl = df_iter[~df_iter.label_pred.isna()].copy(deep=True)

# !! Note: all downstream analyses are not done on predictions from a single classifier, but on the average predictions on N classifiers



### Hypothesis 1: Right-wing shift of labour in 90s
"""p287: Substantively... sharp rightwards shift of Labour between 1987 and 1997 on both economic and social policy, a shift identified by expert text processing and independent expert surveys." (p287)"""

## add some meta-data
# https://manifesto-project.wzb.eu/down/data/2020a/codebooks/codebook_MPDataset_MPDS2020a.pdf
#df_iter_cl["decade"] = [str(date)[:3] + "0" for date in df_iter_cl.year]

## convert sentence level classification predictions to aggregate manifesto scores
df_aggreg_classifier = df_iter_cl.groupby("manifestoid", as_index=False, group_keys=False).apply(lambda x: x.label_scale_pred.mean())
df_aggreg_classifier = df_aggreg_classifier.rename(columns={None: "score_manif_mean"})

### load gold external data
df_aggreg_gold = pd.read_csv("./data-clean/benoit_leftright_manifestos_gold.zip")
# expert annotation
df_aggreg_experts = df_aggreg_gold[(df_aggreg_gold.source == "Experts") & (df_aggreg_gold.scale == "Economic")].reset_index(drop=True)
# crowd annotation
df_aggreg_crowd = df_aggreg_gold[(df_aggreg_gold.source == "Crowd") & (df_aggreg_gold.scale == "Economic")].reset_index(drop=True)
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

## all in one plot
df_plot = pd.DataFrame({
    "year": df_aggreg_experts.year,
    "score_surveys": df_aggreg_surveys.Mean,
    "score_experts": df_aggreg_experts.score_manif_mean,
    "score_crowd": df_aggreg_crowd.score_manif_mean,
    "score_classifier": df_aggreg_classifier.score_manif_mean,
    "color": ["blue" if party=="Con" else "red" if party=="Lab" else "yellow" for party in df_aggreg_experts.party]
})


## correlate
pearson_cor = pearsonr(df_plot[f"score_{INTERNAL_DATA}"], df_plot[f"score_{EXTERNAL_DATA}"])
print(pearson_cor)



### visualise
# for substantive interpretation


fig = go.Figure()

# adding texts: https://plotly.com/python/text-and-annotations/

fig.add_trace(
    go.Scatter(
        x=df_plot[f"score_{EXTERNAL_DATA}"],
        y=df_plot[f"score_{INTERNAL_DATA}"],
        text=df_plot["year"].astype(str).apply(lambda x: x[2:]),
        mode='markers+text',
        textposition="top center",
        marker=dict(
            color=df_plot.color,
            size=30,
            line=dict(
                color='MediumPurple',
                width=2
            )
        ),
        textfont=dict(
                family="sans serif",
                size=18,
                color="black",
        ),
        showlegend=False
    ),
)

fig.update_xaxes(title_text=f"{EXTERNAL_DATA} judgement")
fig.update_yaxes(title_text=f'{INTERNAL_DATA} prediction')
fig.update_layout(
    title_text=f'UK Manifestos 1987 - 2010: left/right comparison {INTERNAL_DATA} vs. {EXTERNAL_DATA}  ({MAX_SAMPLE} data-train)',
    title_x=0.5
)

## add regression line
# ! probably not correct. not sure what should by x_train, what y_train, or test
#https://plotly.com/python/ml-regression/
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df_plot[f"score_{EXTERNAL_DATA}"].array.reshape(-1, 1), df_plot[f"score_{INTERNAL_DATA}"].array.reshape(-1, 1))
x_range = np.linspace(df_plot[f"score_{EXTERNAL_DATA}"].min(), df_plot[f"score_{EXTERNAL_DATA}"].max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))
y_range = y_range.reshape(-1, len(y_range))[0]

fig.add_trace(go.Scatter(
    x=x_range, y=y_range, name='trendline', marker=dict(color="grey"), showlegend=False
    )
)

## add correlation to figure
# https://plotly.com/python/text-and-annotations/
pearson_cor
fig.add_annotation(
    dict(
        font=dict(color='black',size=18),
        # place text with a 10% ofset from the min/max values for each axis
        x=df_plot[f"score_{EXTERNAL_DATA}"].min() + (df_plot[f"score_{EXTERNAL_DATA}"].min() * 0.2),
        y=df_plot[f"score_{INTERNAL_DATA}"].max() - (df_plot[f"score_{INTERNAL_DATA}"].max() * 0.2),
        showarrow=False,
        text=f"r: {round(pearson_cor[0], 3):.2f}   p: {round(pearson_cor[1], 4):.3f}",
        textangle=0,
        #xanchor='left',
        #xref="paper",
        #yref="paper"
    )
)

## show plot
fig.show(renderer="browser")



#### Hypothesis 1. interpretation
## undertuned classical algo
# does not capture the labour rightward shift; despite good correlation (can be tracherous)
# issues revealed by meta-metrics seem to 'correlate' with this too: worse performance for later years! And also worse performance for labour
# also no improvements with fairlearn debiasing







