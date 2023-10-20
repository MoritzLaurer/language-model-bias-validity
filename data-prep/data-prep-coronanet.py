### This scripts downloads and cleans the data for the CoronaNet dataset

# load packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


### Load & prepare data

## load data
# Overview of CoronaNet - https://www.coronanet-project.org/
# using event data format
# ! dataset is updated regularly. Working with github version commit 24.01.22
# last data update seems to be from Jan 2022 for the compatible data format https://github.com/CoronaNetDataScience/corona_tscs/tree/64b0ef942aea98057e41fed16794284847e34cd6/data/CoronaNet/data_bulk
df = pd.read_csv("https://github.com/CoronaNetDataScience/corona_tscs/raw/64b0ef942aea98057e41fed16794284847e34cd6/data/CoronaNet/data_bulk/coronanet_release.csv.gz")
print(df.columns)
print(len(df))

## Data Cleaning
# Select relevant columns
df_cl = df[['record_id', 'policy_id', 'entry_type', 'update_type', #'update_level', 'update_level_var',
       'description', #'date_announced', 'date_start', 'date_end', 'date_end_spec', 
       'ISO_A3', "ISO_A2", 'recorded_date', # 'country',  'ISO_A2',
       #'init_country_level', 'domestic_policy', 'province', 'ISO_L2', 'city',
       'type', 'type_sub_cat', 
       #'type_new_admin_coop', 'type_vac_cat', 'type_vac_mix', 'type_vac_reg', 'type_vac_purchase', 'type_vac_group',
       #'type_vac_group_rank', 'type_vac_who_pays', 'type_vac_dist_admin',
       #'type_vac_loc', 'type_vac_cost_num', 'type_vac_cost_scale',
       #'type_vac_cost_unit', 'type_vac_cost_gov_perc', 'type_vac_amt_num',
       #'type_vac_amt_scale', 'type_vac_amt_unit', 'type_vac_amt_gov_perc',
       #'type_text', 'institution_cat', 'institution_status',
       #'institution_conditions', #'target_init_same', 'target_country',
       #'target_geog_level', 'target_region', 'target_province', 'target_city',
       #'target_other', 'target_who_what', 'target_who_gen', 'target_direction',
       #'travel_mechanism', 'compliance', 'enforcer', 'dist_index_high_est',
       #'dist_index_med_est', 'dist_index_low_est', 'dist_index_country_rank',
       'pdf_link', 'link', #'date_updated', 'recorded_date'
       ]].copy(deep=True)

## add new meta-data columns
# continents
from pycountry_convert import country_alpha2_to_continent_code
df_cl["ISO_A2"].replace("-", np.nan, inplace=True)
df_cl["ISO_A3"].replace("-", np.nan, inplace=True)
df_cl["continent"] = [country_alpha2_to_continent_code(iso2) if ((not pd.isna(iso2)) and (iso2 not in ['TL', 'VA'])) else np.nan for iso2 in df_cl["ISO_A2"]]
# special case for country not covered by package: if in ISO_A2 column [TL, VA] then convert value in continent column to continent
df_cl.loc[df_cl["ISO_A2"] == "VA", "continent"] = "EU"
df_cl.loc[df_cl["ISO_A2"] == "TL", "continent"] = "AS"
# time
df_cl["year"] = df_cl["recorded_date"].apply(lambda x: str(x)[:4])
df_cl["year"].value_counts()

## data cleaning
print(len(df_cl))

# remove very short and long strings - too much noise
df_cl["description"] = df_cl["description"].str.replace("\n", " ")
df_cl = df_cl[df_cl.description.str.len().ge(30)]  # removes  67
print(len(df_cl))
df_cl = df_cl[~df_cl.description.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
print(len(df_cl))

# are there unique texts which are annotated with more than one type? Yes. removes around 10k
df_cl = df_cl.groupby(by="description").filter(lambda x: len(x.value_counts("type")) == 1)
print(len(df_cl))

# duplicates
# remove updates/duplicate policy ids - only work with unique policy_id. each update to a policy measure is a new row, with some slight updates at the end of the description text.
# best to only work with unique policy measures, not unique description strings (otherwise small description updates multiply one policy). ! deduplication on policy_id removes 27k rows (updates) of same policy measure
df_cl = df_cl[~df_cl.policy_id.duplicated(keep="first")]
print(len(df_cl))
# also remove duplicate texts
df_cl = df_cl[~df_cl.description.duplicated(keep="first")]  # removes around 8.5k
print(len(df_cl))

# remove very low n types
df_cl = df_cl[df_cl.type != "Missing/Not Applicable"]  # type only has 6 entries
print(len(df_cl))

# maintain and rename only key columns & rename colums so that trainer recognises them
df_cl = df_cl.rename(columns={"description": "text", "type": "label_text"})
# add numeric labels column based on alphabetical label_text order
df_cl["labels"] = pd.factorize(df_cl["label_text"], sort=True)[0]

# final update
df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 

# final clean
df_cl = df_cl[~df_cl.continent.isna()]
df_cl = df_cl[~df_cl.year.isna()]

# reorder columns
print(df_cl.columns)
df_cl = df_cl[["labels", "label_text", "text", "type_sub_cat", "record_id", "policy_id", "ISO_A3", "continent", "recorded_date", "year", "pdf_link", "link"]]



### inspect labels distribution
print("\n")
df_cl.label_text.value_counts()
df_cl.ISO_A3.value_counts()
df_cl.year.value_counts()
df_cl.continent.value_counts()

# Challenge: too many classes; very imbalanced
# pragmatically only use top 4 classes. Same n_class as PimPo
top_n_classes = 4
df_cl = df_cl[df_cl.label_text.isin(df_cl.label_text.value_counts()[:top_n_classes].index)]
df_cl.loc[:, "labels"] = df_cl.label_text.factorize(sort=True)[0]

## train-test split
df_train, df_test = train_test_split(df_cl, test_size=0.2, random_state=SEED_GLOBAL, stratify=df_cl["label_text"])

# save to disk
df_cl.to_csv(f"./data-clean/df_coronanet_all.zip",
                compression={"method": "zip", "archive_name": f"df_coronanet_all.csv"}, index=False)
df_train.to_csv(f"./data-clean/df_coronanet_train.zip",
                compression={"method": "zip", "archive_name": f"df_coronanet_train.csv"}, index=False)
df_test.to_csv(f"./data-clean/df_coronanet_test.zip",
                compression={"method": "zip", "archive_name": f"df_coronanet_test.csv"}, index=False)


# for testing, only use top 4 classes. Same n_class as PimPo
label_distribution_per_group_member_country = df_train.groupby("ISO_A3", group_keys=True, as_index=True).apply(lambda x: x.label_text.value_counts())
label_distribution_per_group_member_continent = df_train.groupby("continent", group_keys=True, as_index=True).apply(lambda x: x.label_text.value_counts())
label_distribution_per_group_member_year = df_train.groupby("year", group_keys=True, as_index=True).apply(lambda x: x.label_text.value_counts())
print("Overall label distribution per group member:\n", label_distribution_per_group_member_country)
print("Overall label distribution per group member:\n", label_distribution_per_group_member_continent)
print("Overall label distribution per group member:\n", label_distribution_per_group_member_year)
print("Overall label distribution:\n", df_train.label_text.value_counts())

# number of members per group
len(df_cl["year"].unique())

# => distribution works for 1k sample

