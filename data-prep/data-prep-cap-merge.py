

import pandas as pd

df_court = pd.read_csv(f"./data-clean/df_cap_court_all.zip")
df_sotu = pd.read_csv(f"./data-clean/df_cap_sotu_for_merge.zip")

df_court["domain"] = "legal"
df_sotu["domain"] = "speech"

# concatenate datasets
df_concat = pd.concat([df_court, df_sotu], axis=0)

# remove label_text that is not in both datasets
set1 = set(df_court.label_text.unique())
set2 = set(df_sotu.label_text.unique())
union = set1.union(set2)
label_text_common = set1.intersection(set2)
label_text_not_common = union - label_text_common
df_concat = df_concat[~df_concat.label_text.isin(label_text_not_common)]

# make sure that numeric labels are aligned with factorized labels
df_concat["labels"] = pd.factorize(df_concat["label_text"], sort=True)[0]


# only use top N classes to avoid label imbalance issues
# take top N from legal domain, because these are also in speeches; while overall top N are underrepresented in legal. (e.g. International Affairs only 41 times in lega, but 3k in speeches)
# top in legal: Law and Crime, Civil Rights, Domestic Commerce, Labor, Government Operations
# enables me to draw 1k balanced sample
top_n_classes = 5
label_text_top_n = [label_text for label_text in df_concat[df_concat["domain"] == "legal"].label_text.value_counts()[:top_n_classes].index]
print("classes used: ", label_text_top_n)
df_concat = df_concat[df_concat.label_text.isin(label_text_top_n)]
df_concat.loc[:, "labels"] = df_concat.label_text.factorize(sort=True)[0]


## train-test split
from sklearn.model_selection import train_test_split

#TODO: remember that with this train-test split I cannot use preceding+following sentences, otherwise leakage
df_train, df_test = train_test_split(df_concat, test_size=0.2, random_state=42, stratify=df_concat["label_text"])

# down sample test set
# no need to downsample, already only 1928 texts

# save to disk
df_concat.to_csv(f"./data-clean/df_cap_merge_all.zip",
                compression={"method": "zip", "archive_name": f"df_cap_merge_all.csv"}, index=False)
df_train.to_csv(f"./data-clean/df_cap_merge_train.zip",
                compression={"method": "zip", "archive_name": f"df_cap_merge_train.csv"}, index=False)
df_test.to_csv(f"./data-clean/df_cap_merge_test.zip",
                compression={"method": "zip", "archive_name": f"df_cap_merge_test.csv"}, index=False)





## check per group distribution
label_distribution_per_group_member = df_concat.groupby("domain").apply(lambda x: x.label_text.value_counts())
print("Overall label distribution per group member:\n", label_distribution_per_group_member)
print("Overall label distribution:\n", df_concat.label_text.value_counts())
