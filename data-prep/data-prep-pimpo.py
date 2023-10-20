### This scripts downloads and cleans the data for the PimPo dataset

# load packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


## load already translated data
df = pd.read_csv(f"./data-raw/df_pimpo_samp_trans_m2m_100_1.2B.zip", engine='python')

## prepare data
# add metadata
parfam_aggreg_map = {"ECO": "left", "LEF": "left", "SOC": "left",
                     "CHR": "right", "CON": "right", "NAT": "right",
                     "LIB": "other", "AGR": "other", "ETH": "other", "SIP": "other"}
df["parfam_text_aggreg"] = df.parfam_text.map(parfam_aggreg_map)
df["decade"] = df["date"].apply(lambda x: int(str(x)[:3] + "0"))

# select columns
df_cl = df[['label', 'label_text', 'country_iso', 'language_iso', 'doc_id',
           'text_original', 'text_preceding', 'text_following', 'selection',
           'certainty_selection', 'topic', 'certainty_topic', 'direction',
           'certainty_direction', 'rn', 'cmp_code', 'partyname', 'partyabbrev',
           'parfam', 'parfam_text', "parfam_text_aggreg", 'date', "decade", 'language_iso_fasttext',
           'text_preceding_trans', 'text_original_trans', 'text_following_trans',
           'language_iso_trans',
           #'text_concat', 'text_concat_embed_multi',
           #'text_trans_concat',
           #'text_trans_concat_embed_en',
           #'text_trans_concat_tfidf', #'text_prepared'
            ]]

# harmonization of colum names
df_cl = df_cl.rename(columns={"label": "labels"})

## merge labels to simpler task
task_label_text_map = {
    'immigration_neutral': "neutral", 'integration_neutral': "neutral",
    'immigration_sceptical': "sceptical", 'integration_sceptical': "sceptical",
    'immigration_supportive': "supportive", 'integration_supportive': "supportive",
    'no_topic': "no_topic"
}
df_cl["label_text"] = df_cl.label_text.map(task_label_text_map)
df_cl["labels"] = df_cl.label_text.factorize(sort=True)[0]


## train-test split
# important: with this train-test split I cannot use preceding+following sentences, otherwise leakage
df_train, df_test = train_test_split(df_cl, test_size=0.2, random_state=SEED_GLOBAL, stratify=df_cl["label_text"])

# reduce no-topic to N
sample_no_topic = 5000
df_test = df_test.groupby(by="label_text", as_index=False, group_keys=False).apply(
    lambda x: x.sample(n=min(sample_no_topic, len(x)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] == "no_topic" else x)
print("df_test.label_text.value_counts:\n", df_test.label_text.value_counts())


# save to disk
df_cl.to_csv(f"./data-clean/df_pimpo_samp_all.zip",
                compression={"method": "zip", "archive_name": f"df_pimpo_samp_all.csv"}, index=False)
df_train.to_csv(f"./data-clean/df_pimpo_samp_train.zip",
                compression={"method": "zip", "archive_name": f"df_pimpo_samp_train.csv"}, index=False)
df_test.to_csv(f"./data-clean/df_pimpo_samp_test.zip",
                compression={"method": "zip", "archive_name": f"df_pimpo_samp_test.csv"}, index=False)


# test label distribution and imbalance
label_distribution_per_group_member = df_train.groupby("parfam_text").apply(lambda x: x.label_text.value_counts())
label_distribution_per_group_member = df_train.groupby("parfam_text_aggreg").apply(lambda x: x.label_text.value_counts())
label_distribution_per_group_member = df_train.groupby("decade").apply(lambda x: x.label_text.value_counts())
label_distribution_per_group_member = df_train.groupby("country_iso").apply(lambda x: x.label_text.value_counts())
print("Overall label distribution per group member:\n", label_distribution_per_group_member)
print("Overall label distribution:\n", df_train.label_text.value_counts())

# => does not really work with 1k+ and single group member due to class imbalance and lack of per-group data
