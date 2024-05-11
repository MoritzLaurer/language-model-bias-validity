# Analyse per-group data distribution in test data
# to show degree of group imbalance

import pandas as pd

datataset_names = [
    "df_pimpo_samp_test", "df_coronanet_test", "df_cap_merge_test", "df_cap_sotu_test"
    # "df_pimpo_samp_train", "df_coronanet_train", "df_cap_merge_train", "df_cap_sotu_train"
]

group_dic = {
    "df_pimpo_samp_test": ["country_iso", "parfam_text", "decade"],
    "df_coronanet_test": ["year", "continent", "ISO_A3"],
    "df_cap_merge_test": ["domain"],
    "df_cap_sotu_test": ["pres_party", "phase"]
}


# Load data
df_dic = {}
for dataset in datataset_names:
    df_dic.update({dataset: pd.read_csv(f"./data-clean/{dataset}.zip", engine="python")})

# Create different dfs per group
df_groups_dic = {}
for key_dataset_name, value_group_types in group_dic.items():
    for group_type in value_group_types:
        # get specific dataset
        df_group = df_dic[key_dataset_name]
        # count groups per group_type
        df_group_count = pd.DataFrame(df_group[group_type].value_counts()).reset_index()
        df_group_count.columns = [group_type, "n_texts"]

        # limit n rows for appendix tables
        df_group_count = df_group_count.head(20)

        # to dic
        df_groups_dic.update({f"{key_dataset_name}_{group_type}": df_group_count})
        # to csv
        df_group_count.to_csv(f"./results/group_distributions/{key_dataset_name}_{group_type}.csv", index=False)






