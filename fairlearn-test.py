

import pandas as pd
import numpy as np
import sklearn.metrics as skm

import os
os.getcwd()

## load data
df = pd.read_csv("./data/df_test_many2anchor_svm_500_embed_predictions.csv")



### tests based on fairlearn docs
# https://fairlearn.org/main/user_guide/assessment.html

##
#y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1]
#y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
y_true = df.label
y_pred = df.prediction

##
#group_membership_data = ['d', 'a', 'c', 'b', 'b', 'c', 'c', 'c', 'b', 'd', 'c', 'a', 'b', 'd', 'c', 'c']
group_membership_data = df.language_iso  # .language_iso .date .parfam .party .label_domain_text .label_subcat_text .country_iso  # in raw data: .coderid .testresult
# https://manifesto-project.wzb.eu/down/data/2020a/codebooks/codebook_MPDataset_MPDS2020a.pdf
mapping_parfam = {10: "ECO: Ecological parties", 20: "LEF: Socialist or other left parties",
                  30: "SOC: Social democratic parties", 40: "LIB: Liberal parties", 50: "CHR: Christian democratic parties (in Isreal also Jewish parties)",
                  60: "CON: Conservative parties", 70: "NAT: Nationalist parties", 80: "AGR: Agrarian parties",
                  90: "ETH: Ethnic and regional parties", 95: "SIP: Special issue parties", 98: "DIV: Electoral alliances of diverse origin without dominant party",
                  999: "MI: Missing information"
}
df["parfam_name"] = df.parfam.map(mapping_parfam)
df["parfam_rile"] = ["left" if parfam in [10, 20, 30] else "right" if parfam in [50, 60, 70, 80, 90] else "other" for parfam in df["parfam"]]
cmp_code_left = [103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701]
cmp_code_right = [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606]
cmp_code_other = np.unique([cmp_code for cmp_code in df["cmp_code"] if cmp_code not in cmp_code_left + cmp_code_right])
df["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df["cmp_code"]]
df["decade"] = [str(date)[:3]+"0" for date in df.date]


##
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import count

import functools
f1_macro = functools.partial(skm.f1_score, average='macro')
precision_micro = functools.partial(skm.precision_score, average='micro')
recall_micro = functools.partial(skm.recall_score, average='micro')
precision_macro = functools.partial(skm.precision_score, average='macro')
recall_macro = functools.partial(skm.recall_score, average='macro')

grouped_metric = MetricFrame(metrics={'accuracy': skm.accuracy_score,
                                      #'accuracy_balanced': skm.balanced_accuracy_score,
                                      #"precision_micro": precision_micro,
                                      #"recall_micro": recall_micro,
                                      "f1_macro": f1_macro,
                                      #"precision_macro": precision_macro,
                                      #"recall_macro": recall_macro,
                                      'count': count
                                      },
                             y_true=y_true,
                             y_pred=y_pred,
                             # can look at intersection between features by passing df with multiple columns
                             sensitive_features=df[["label_rile"]],  # df[["language_iso", "parfam_name", "parfam_rile", "label_rile", "decade"]]
                             #control_features=df[["parfam_name"]]
)

print("## Metrics overall:\n", grouped_metric.overall, "\n")
print("## Metrics by group:\n", grouped_metric.by_group, "\n")  #.to_dict()
#print("## Metrics min:\n", grouped_metric.group_min(), "\n")
#print("## Metrics max:\n", grouped_metric.group_max(), "\n")
print("## Metrics difference min-max:\n", grouped_metric.difference(method='between_groups'), "\n")  # to_overall, between_groups  # difference or ratio of the metric values between the best and the worst slice
#print(grouped_metric.ratio(method='between_groups')) # to_overall, between_group  # difference or ratio of the metric values between the best and the worst slice

# scalar values from difference/ratio/min/max can be used for hp tuning  # https://fairlearn.org/main/user_guide/assessment.html#scalar-results-from-metricframe






### confusion matrix
# on multi-class classification  https://fairlearn.org/main/auto_examples/plot_metricframe_beyond_binary_classification.html#multiclass-nonscalar-results
conf_mat = functools.partial(skm.confusion_matrix, labels=np.unique(y_true))

grouped_metric = MetricFrame(metrics={"conf_mat": conf_mat},
                             y_true=y_true,
                             y_pred=y_pred,
                             sensitive_features=df[["label_rile"]],  # df[["language_iso", "parfam_name", "parfam_rile", "label_rile", "decade"]]
                             #control_features=df[["parfam_name"]]
)
grouped_metric.overall
grouped_metric.by_group

import matplotlib.pyplot as plt
for key_row, value_row in grouped_metric.by_group.iterrows():
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=value_row["conf_mat"], display_labels=np.unique(df[["label_domain_text"]]))
    print(key_row)
    disp.plot(xticks_rotation='vertical')



#### Unfairness mitigation techniques
# https://fairlearn.org/main/user_guide/mitigation.html#mitigation
## Preprocessing
# ! can transform input features to reduce correlation between sensitive attributes and other variables

## Postprocessing


### tests on full dataset
#df_raw = pd.read_csv("/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/data-raw/manifesto_all_2021a.csv")
#print("Left wing categories: ", df_raw[df_raw.cmp_code.isin(cmp_code_left)].title_variable.unique())
#print("Right wing categories: ", df_raw[df_raw.cmp_code.isin(cmp_code_right)].title_variable.unique())



