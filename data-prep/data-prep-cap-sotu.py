
### This scripts downloads and cleans the data for the CAP- SotU dataset

# load packages
import pandas as pd
import numpy as np
import os

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


### Load & prepare data

## load data
# overview of CAP data: https://www.comparativeagendas.net/datasets_codebooks
# overall CAP master codebook: https://www.comparativeagendas.net/pages/master-codebook
# SOTU codebook 2015: https://comparativeagendas.s3.amazonaws.com/codebookfiles/State_of_the_Union_Address_Codebook.pdf

#df = pd.read_csv("https://comparativeagendas.s3.amazonaws.com/datasetfiles/US-Executive_State_of_the_Union_Speeches-22.csv")
df = pd.read_csv("https://comparativeagendas.s3.amazonaws.com/datasetfiles/US-Exec_SOTU_2023.csv")

print(df.columns)
print(len(df))



#### Data Cleaning


## data cleaning
# contains two types of CAP topics
# based on codebook, seems like PAP is older policy agendas project code from US, while CAP is newer, more international project code
# ! in CAP-us-courts it made more sense to use pap_majortopic
df_cl = df[["description", 'majortopic', 'subtopic', "year", "president", "pres_party", "id"]].copy(deep=True)
print(len(df_cl))

# remove NAs
df_cl = df_cl[~df_cl.description.isna()]
print(len(df_cl))
# remove very short strings
df_cl = df_cl[df_cl.description.str.len().ge(30)]  # removes X. mostly noise, some content like "Amen.	"
print(len(df_cl))
df_cl = df_cl[~df_cl.description.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
print(len(df_cl))
# are there unique texts which are annotated with more than one type? Yes, 105. String like " ", ".", "Thank you very much", "#NAME?", "It's the right thing to do."
#df_cl = df_cl.groupby(by="description").filter(lambda x: len(x.value_counts("majortopic")) == 1)
#print(len(df_cl))
# remove duplicates
# maintain duplicates to maintain sequentiality of texts
#df_cl = df_cl[~df_cl.description.duplicated(keep="first")]  # 170 duplicates
#print(len(df_cl))

# renumber "Other" cateogry label from -555 to 99
df_cl.majortopic = df_cl.majortopic.replace(-555, 99)
df_cl.subtopic = df_cl.subtopic.replace(-555, 99)

# new version has 0 in this column for some reason
df_cl.majortopic = df_cl.majortopic.replace(0, 99)


# rename columns
df_cl = df_cl.rename(columns={"majortopic": "label_cap2", "subtopic": "label_cap4", "description": "text", "id": "id_original"})

df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 


### adding label_text to label ids
# label names from master codebook as of Oct. 2021, https://www.comparativeagendas.net/pages/master-codebook
label_text_map_cap2 = {  
    1: "Macroeconomics",
    2: "Civil Rights", 
    3: "Health",
    4: "Agriculture",
    5: "Labor",
    6: "Education",
    7: "Environment",
    8: "Energy",
    9: "Immigration",
    10: "Transportation",
    12: "Law and Crime",  
    13: "Social Welfare",
    14: "Housing",  
    15: "Domestic Commerce", 
    16: "Defense",
    17: "Technology",  
    18: "Foreign Trade",
    19: "International Affairs",  
    20: "Government Operations",
    21: "Public Lands",  
    23: "Culture", 
    99: "Other",  
}

df_cl["label_cap2_text"] = df_cl.label_cap2.map(label_text_map_cap2)
print(f"Maybe label_cap4 later too. Very fine-grained number of classes: {len(df_cl.label_cap4.unique())}. Makes for interesting data")

# labels numbers in alphabetical order of text
df_cl["labels"] = pd.factorize(df_cl["label_cap2_text"], sort=True)[0]
df_cl["label_text"] = df_cl["label_cap2_text"]

# add decade column
df_cl.loc[:, "decade"] = df_cl.loc[:, "year"] // 10 * 10
# add time phase column
df_cl.loc[:, "phase"] = ["cold_war" if year <= 1991 else "post_cold_war" for year in df_cl["year"]]

# reorder and select columns
df_cl = df_cl[["labels", "label_text", "text", 'label_cap2', "label_cap2_text", 'label_cap4',  "year", "decade", "phase", "president", "pres_party", "id_original"]]

# test that label_cap2 and label_cap2_text correspond
assert len(df_cl[df_cl.label_cap2_text.isna()]) == 0  # each label_cap2 could be mapped to a labels text. no labels text is missing.
print(np.sort(df_cl["label_cap2_text"].value_counts().tolist()) == np.sort(df_cl["label_cap2"].value_counts().tolist()))

df_cl.label_cap2_text.value_counts()



### augmenting text column

## new column where every sentence is merged with previous sentence
n_unique_doc_lst = []
n_unique_doc = 0
text_preceding = []
text_following = []
for name_group, df_group in df_cl.groupby(by=["president", "year"], sort=False):  # over each speech to avoid merging sentences accross manifestos
    n_unique_doc += 1
    df_group = df_group.reset_index(drop=True)  # reset index to enable iterating over index
    #text_ext = []
    for i in range(len(df_group["text"])):
        if i > 0 and i < len(df_group["text"]) - 1:
            text_preceding.append(df_group["text"][i-1])
            text_following.append(df_group["text"][i+1])
        elif i == 0:  # for very first sentence of each manifesto
            text_preceding.append("")
            text_following.append(df_group["text"][i+1])
        elif i == len(df_group["text"]) - 1:  # for last sentence
            text_preceding.append(df_group["text"][i-1])
            text_following.append("")
        else:
          raise Exception("Issue: condition not in code")
    #text_ext2_all.append(text_ext)
    n_unique_doc_lst.append([n_unique_doc] * len(df_group["text"]))
n_unique_doc_lst = [item for sublist in n_unique_doc_lst for item in sublist]

# create new columns
df_cl["text_original"] = df_cl["text"]
df_cl = df_cl.drop(columns=["text"])
df_cl["text_preceding"] = text_preceding
df_cl["text_following"] = text_following
df_cl["doc_id"] = n_unique_doc_lst  # column with unique doc identifier


## write to disk before sampling classes; version for merging with cap_court
df_cl_merged = df_cl.copy()
df_cl_merged.to_csv(f"./data-clean/df_cap_sotu_for_merge.zip",
                compression={"method": "zip", "archive_name": f"df_cap_sotu_for_merge.csv"}, index=False)


## sample only top N classes
top_n_classes = 6
# removing other because its unclean
# Other, Macroeconomics, International Affairs, Defense, Health, Government Operations
label_text_top_n = [label_text for label_text in df_cl.label_text.value_counts()[:top_n_classes].index if label_text != "Other"]
print("classes used: ", label_text_top_n)
df_cl = df_cl[df_cl.label_text.isin(label_text_top_n)]
df_cl.loc[:, "labels"] = df_cl.label_text.factorize(sort=True)[0]

# rename parties:
df_cl.loc[:, "pres_party"] = df_cl.loc[:, "pres_party"].replace({100: "dem", 200: "rep"})

## train-test split
from sklearn.model_selection import train_test_split

#TODO: remember that with this train-test split I cannot use preceding+following sentences, otherwise leakage
df_train, df_test = train_test_split(df_cl, test_size=0.2, random_state=42, stratify=df_cl["label_text"])


# down sample test set
# no need to downsample, already only 2314 texts


# save to disk
df_cl.to_csv(f"./data-clean/df_cap_sotu_all.zip",
                compression={"method": "zip", "archive_name": f"df_cap_sotu_all.csv"}, index=False)
df_train.to_csv(f"./data-clean/df_cap_sotu_train.zip",
                compression={"method": "zip", "archive_name": f"df_cap_sotu_train.csv"}, index=False)
df_test.to_csv(f"./data-clean/df_cap_sotu_test.zip",
                compression={"method": "zip", "archive_name": f"df_cap_sotu_test.csv"}, index=False)



## check per group distribution
label_distribution_per_group_member = df_train.groupby("pres_party").apply(lambda x: x.label_text.value_counts())
label_distribution_per_group_member = df_train.groupby("phase").apply(lambda x: x.label_text.value_counts())
print("Overall label distribution per group member:\n", label_distribution_per_group_member)
print("Overall label distribution:\n", df_train.label_text.value_counts())




#### Train-Test-Split
"""
### simplified dataset
# sample based on docs - to make test set composed of entirely different docs - avoid data leakage when including surrounding sentences
doc_id_train = pd.Series(df_cl.doc_id.unique()).sample(frac=0.80, random_state=SEED_GLOBAL).tolist()
doc_id_test = df_cl[~df_cl.doc_id.isin(doc_id_train)].doc_id.unique().tolist()
print(len(doc_id_train))
print(len(doc_id_test))
assert sum([doc_id in doc_id_train for doc_id in doc_id_test]) == 0, "should be 0 if doc_id_train and doc_id_test don't overlap"
df_train = df_cl[df_cl.doc_id.isin(doc_id_train)]
df_test = df_cl[~df_cl.doc_id.isin(doc_id_train)]

# ## Save data

# dataset statistics
text_length = [len(text) for text in df_cl.text_original]
text_context_length = [len(text) + len(preceding) + len(following) for text, preceding, following in zip(df_cl.text_original, df_cl.text_preceding, df_cl.text_following)]
print("Average number of characters in text: ", int(np.mean(text_length)))
print("Average number of characters in text with context: ", int(np.mean(text_context_length)))

print(os.getcwd())

df_cl.to_csv("./data_clean/df_cap_sotu_all.csv")
df_train.to_csv("./data_clean/df_cap_sotu_train.csv")
df_test.to_csv("./data_clean/df_cap_sotu_test.csv")

"""