
### This scripts downloads and cleans the data for the CAP-US-Court dataset

# load packages
import pandas as pd
import numpy as np

SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)


### Load & prepare data

# raw load data
# codebook: https://comparativeagendas.s3.amazonaws.com/codebookfiles/Supreme_Court_Codebook.pdf
df = pd.read_csv("https://comparativeagendas.s3.amazonaws.com/datasetfiles/US-Judicial-supreme_court_cases_20.1.csv", sep=",", encoding='utf-8') #encoding='utf-8',  # low_memory=False  #lineterminator='\t',
print(df.columns)
print(len(df))


## Data Cleaning

df_cl = df[['summary', 'ruling', #'type', 'id',
            #'filter_vac', 'filter_dis', 'filter_gd',  # vacated, dismissed, granted/denied
            #'oral', 'decision', 'term_a', 'term_b', 'year', 
            'pap_majortopic', 'pap_subtopic', #'majortopic', 'subtopic', 
            'case_name'
            #'source', 'description', 'congress'
            ]].copy(deep=True)
print(len(df_cl), " - original length")

# marginal different between pap_majortopic and majortopic - going with pap_majortopic, because majortopic has only 2 texts in category 23, while pap_ has them in category 6
# ! in CAP-sotu it made more sense to use majortopic
# pd.DataFrame([df_cl.pap_majortopic.value_counts(), df_cl.majortopic.value_counts()])
# Summary seems to be main basis for coding: "A comprehensive description or summary of the case was acquired in order to code each case according to the Policy Agendas Project coding scheme" ... "For the purposes of public policy research, we felt the “Summary” section would suffice. "
# not entirely clear if "ruling" text was also used. "in most instances, this data was found in the second paragraph of the “Summary” or in the more detailed “Opinion” sections for a given case in LexisNexis. For the purposes of public policy research, the ruling paragraph of the “Summary” section is often sufficient."
# => seems like coders mostly looked at summary, but might also have looked at more from the NexiLexis (or other) website

# remove NAs and very short texts
df_cl = df_cl[~df_cl.summary.isna()]  # 1 NA
df_cl = df_cl[~df_cl.ruling.isna()]  # 0 NA
print(len(df_cl), "after removing NA")
df_cl = df_cl[df_cl.summary.str.len().ge(30)]  # removes 1
df_cl = df_cl[df_cl.ruling.str.len().ge(30)]  # removes 0
print(len(df_cl), "after removing very short texts")
# keep very long texts, is part of the characteristics of this dataset
#df_cl = df_cl[~df_cl.summary.str.len().ge(1000)]  # remove very long descriptions, assuming that they contain too much noise from other types and unrelated language. 1000 characters removes around 9k
#print(len(df_cl))

# remove those where same text received different label
df_cl = df_cl.groupby(by="summary").filter(lambda x: len(x.pap_majortopic.unique()) == 1)  # removes 60
print(len(df_cl), "after removing same texts with different label")

# deduplicated 
df_cl = df_cl[~df_cl.summary.duplicated(keep='first')]
print(len(df_cl), "after deduplicating summaries")
df_cl = df_cl[~df_cl.ruling.duplicated(keep='first')]
print(len(df_cl), "after deduplicating rulings")


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
# ! label Culture and Other not availabel in dataset

df_cl = df_cl.rename(columns={'pap_majortopic': "label_cap2", 'pap_subtopic': "label_cap4"})

df_cl["label_cap2_text"] = df_cl.label_cap2.map(label_text_map_cap2)
print(f"Maybe include label_cap4 later too. Very fine-grained number of classes: {len(df_cl.label_cap4.unique())}. Makes for interesting data")

# labels numbers in alphabetical order of text
df_cl["labels"] = pd.factorize(df_cl["label_cap2_text"], sort=True)[0]
df_cl["label_text"] = df_cl["label_cap2_text"]

# test that label_cap2 and label_cap2_text correspond
assert len(df_cl[df_cl.label_cap2_text.isna()]) == 0  # each label_cap2 could be mapped to a labels text. no labels text is missing.
print(np.sort(df_cl["label_cap2_text"].value_counts().tolist()) == np.sort(df_cl["label_cap2"].value_counts().tolist()))



## chose main text for analysis
# can also try concatenating summary and ruling - makes it longer and more complicated
# or try adding the course case name at the beginning
df_cl["text"] = df_cl["summary"] + df_cl["ruling"]

## finalisation
df_cl = df_cl[["labels", "label_text", "text", "summary", "ruling", "case_name", "label_cap2", "label_cap2_text", "label_cap4"]]  # "summary", "ruling", "label_cap2", "label_cap4",

df_cl = df_cl.reset_index(drop=True)
df_cl.index = df_cl.index.rename("idx")  # name index. provides proper column name in dataset object downstream 

df_cl.label_text.value_counts()



### length of texts
# ! some texts are quite long. processing with transformers max_len=512 will probably lead to truncation for a few texts
text_length = [len(text) for text in df_cl.text]
print("Average number of characters in text: ", int(np.mean(text_length)), "\n")
pd.Series(text_length).value_counts(bins=10).plot.bar()


### save file to disk

# zipped csv
df_cl.to_csv(
       f"./data-clean/df_cap_court_all.zip",
       compression={"method": "zip", "archive_name": f"df_cap_court_all.csv"},
       index=False
)

