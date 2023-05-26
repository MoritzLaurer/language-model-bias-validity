

## load libraries
import pandas as pd
import numpy as np
import os

## select analysed dataset
N_SAMPLE = "1000"
N_ITER = "0"
DATE = 20230207
MODEL_NAME = "logistic"  #"logistic"  #"DeBERTa-v3-base"
META_DATA = ["parfam_text"]  #["label_text", "parfam_text", "parfam_text_aggreg", "country_iso", "language_iso", "decade"]
#NORMALIZE = True

# TODO: need to rerun everything on full 200k texts to get the aggregate numbers right (otherwise no-topic underestimated, all crowd label counts distorted etc.)
# TODO: rerun everything with 2+2 surrounding sentences, like in paper; (generally look at pre-processing again to try and increase data quality)
# TODO: for random samples: make average over 5 classifiers work

### load data
#df_cl, _df_train = load_data(languages=LANGUAGES, task=TASK, method=METHOD, model_size=MODEL_SIZE, hypothesis=HYPOTHESIS, vectorizer=VECTORIZER, max_sample_lang=MAX_SAMPLE_LANG, date=DATE)
df = pd.read_csv(f"./data-classified/pimpo/df_pimpo_{N_SAMPLE}_{MODEL_NAME}_{DATE}_{N_ITER}.zip")

# add  meta-data
# decades
df["decade"] = df["date"].apply(lambda x: str(x)[:3] + "0" if int(str(x)[3]) < 5 else str(x)[:3] + "5")
# parties left/other/right
parfam_aggreg_map = {"ECO": "left", "LEF": "left", "SOC": "left",
                     "CHR": "right", "CON": "right", "NAT": "right",
                     "LIB": "other", "AGR": "other", "ETH": "other", "SIP": "other"}
df["parfam_text_aggreg"] = df.parfam_text.map(parfam_aggreg_map)


# remove rows which were in training data
df_train = df[df.label_text_pred.isna()]
df_cl = df[~df.label_text_pred.isna()]


# code from Andreu: https://github.com/CasAndreu/moritz



### Logistic regression inspired by "bias in regression, can we fix it" paper
# https://realpython.com/logistic-regression-python/

#df_cl["error_binary"] = [0 if label_gold == label_pred else 1 for label_gold, label_pred in zip(df_cl.label_text, df_cl.label_text_pred)]
#df_cl["parfam_text_factor"], parfam_text_unique = pd.factorize(df_cl["parfam_text"])

y_error = [0 if label_gold == label_pred else 1 for label_gold, label_pred in zip(df_cl.label_text, df_cl.label_text_pred)]

# factorize each group
"""group_dic = {}
for group in META_DATA:
    group_factorized, group_unique = pd.factorize(df_cl[group])
    group_dic.update({group: [group_factorized, group_unique]})"""

# factorize each group and make each group member a binary variable
group_dic = {}
for group in META_DATA:
    group_factorized, group_unique = pd.factorize(df_cl[group])
    # make each group-member a binary variable
    for i, group_member in enumerate(group_unique):
        group_member_binary = [1 if i == label else 0  for label in group_factorized]
        group_dic.update({group_member: [group_member_binary]})


[key for key, value in group_dic.items()]  # checking order of outputs
x_groups = np.column_stack(([value[0] for key, value in group_dic.items()]))

## with sklearn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=42)
# single variable
#model.fit(np.array(x_group).reshape(-1, 1), y_error)
# multiple variables
model.fit(x_groups, y_error)


model.classes_
print(model.intercept_)
print(model.coef_)


## with statsmodels
import statsmodels.api as sm

x_sm = sm.add_constant(x_groups)

model = sm.Logit(y_error, x_sm)

result = model.fit(method='newton')

"""sm_intercept_coef, sm_group_coef = result.params
sm_intercept_p, sm_group_p = result.pvalues
sm_intercept_stderr, sm_group_stderr = result.bse"""

# https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.html
# interpretation: https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
result.summary(yname="error_pred", xname=["(intercept)"] + list(group_dic.keys()), title=f"Model: {MODEL_NAME}", alpha=0.05, yname_list=None)

#result_logistic
#result_nli



