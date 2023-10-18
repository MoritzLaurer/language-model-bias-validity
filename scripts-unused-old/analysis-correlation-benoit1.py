

import pandas as pd
import numpy as np

### correlation analysis for benoit data

## read gold data
df_gold = pd.read_csv(f"./data-clean/benoit_leftright_manifestos_gold.zip")
# subset relevant part for task
df_gold_cl = df_gold[(df_gold.scale == "Economic") & (df_gold.source == "Experts")]
# sort by same column to ensure row alignment for correlation
df_gold_cl = df_gold_cl.sort_values("manifestoid")

## read prediction data
df_pred = pd.read_csv(f"./data-classified/uk-leftright-econ/df_benoit1_pred_test1.zip")
# remove training data
df_pred_cl = df_pred[~df_pred.label_pred.isna()]
# aggregate to manifesto level
df_pred_manifesto = df_pred_cl.groupby(by=["manifestoid"], as_index=False, group_keys=False).apply(lambda x: x.label_scale_pred.mean())
df_pred_manifesto = df_pred_manifesto.rename(columns={None: "score_manif_mean"})
# sort by same column to ensure row alignment for correlation
df_pred_manifesto = df_pred_manifesto.sort_values("manifestoid")


from scipy.stats import pearsonr
pearson_cor = pearsonr(df_gold_cl.score_manif_mean, df_pred_manifesto.score_manif_mean)
print(pearson_cor)










