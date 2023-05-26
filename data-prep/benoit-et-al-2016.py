

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import requests, zipfile, io


##### 1. Analysis: Left/right Econ/Social UK manifestos 1987 - 2010
### load crowd annotated data
# https://github.com/kbenoit/CSTA-APSR/tree/master/Data%20-%20Created
# wide dataset includes already aggregated scores, one row per sentence per 'scale'/domain
# long
r = requests.get('https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20Created/coding_all_long.zip')
# wide
#r = requests.get('https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20Created/coding_all_wide.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
print(zipfile.ZipFile.namelist(z))

# ? not sure what difference the different dates make
# github indicates using "coding_all_long_2013-10-31.dta" https://github.com/kbenoit/CSTA-APSR/blob/master/Data%20-%20Created/MS%20tables%20replication.do
# but does not exist and "coding_all_long_2013-10-31.csv" misses annotations e.g. for 2010 manifestos
# long - each row represents one annotation, not aggregated
df_lr_all = pd.read_stata(z.open("coding_all_long_2014-03-14.dta"))
# wide - data aggregated by authors on sentence level
#df_lr_all = pd.read_stata(z.open("coding_all_wide_2014-03-14.dta"))

# data was collected in different stages, but unclear what they mean
df_lr_all.stage.value_counts(dropna=False)
# clean weird manifesto ID
df_lr_all = df_lr_all[df_lr_all.manifestoid != 99]
# I do not exclude annotations from their tests of providing sequential/non-sequential data. (not clear how to identify them in data)
#df_lr_all.sequence.value_counts()
# remove annotations where titles where shown (94262)
df_lr_all = df_lr_all[df_lr_all.title_shown != "Yes"]
# exclude sentences where context was not shown (only 990)
df_lr_all = df_lr_all[df_lr_all.context != 0]
# remove screener annotations
df_lr_all = df_lr_all[df_lr_all.manifestoid != "screener"]
# remove annotations from semi-experts, because very few annotations and hardly used in paper and speeds up groupby code below
df_lr_all = df_lr_all[df_lr_all.source != "SemiExperts"]


### link CMP meta-data to dataset?
"""# get all un-annotated sentences
df_lr_sents = pd.read_csv("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20Created/master.sentences.csv").drop(columns=["Unnamed: 0"])
# ? which additional meta-data is relevant?
# already have: date, country (always UK), party, language (always en)
# additions: CMP labels
# how to merge? no common key. merge via date and party, but then sentence level merge difficult (maybe through order and string matching)
df_manif = pd.read_csv("/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/data-raw/manifesto_all_2021a.zip", index_col="idx")
"""


### calculate aggregate per-manifesto score for crowd coders
## aggregate scores per sentence, social/economic/None ("scale") and type of annotator (source)
# sample max X annotations per sentence per type of annotator
df_lr_all_samp = df_lr_all.groupby(by=["sentenceid", "source"]).apply(lambda x: x.sample(n=min(len(x), 10), random_state=42)).reset_index(drop=True)
# aggregate
df_lr_all_aggreg_sent = df_lr_all_samp.groupby(by=["sentenceid", "scale", "source"], as_index=False, group_keys=False).apply(lambda x: x.code.mean())
df_lr_all_aggreg_sent = df_lr_all_aggreg_sent.rename(columns={None: "score_sent_mean"})
df_lr_all_aggreg_sent["n_annotations"] = df_lr_all_samp.groupby(by=["sentenceid", "scale", "source"]).apply(lambda x: len(x)).reset_index(drop=True)

# what to do if same sentences was annotated 1x as social and 4x as economic and 2x as None by different annotators?
# options: (1) drop rows where annotators categorised sentence as both economic and social and None. only keep highest N category; (2) or just leave the rows and it averages out in aggregate
# choosing option 1, because it produces more similar value to paper; option 2 produces low 0.84~ correlation for social policy
df_lr_all_aggreg_sent = df_lr_all_aggreg_sent.groupby(["sentenceid", "source"], as_index=False, group_keys=False).apply(lambda x: x.nlargest(1, "n_annotations", keep="all"))

# add meta-data again
assert len(df_lr_all_aggreg_sent) == len(pd.merge(df_lr_all_aggreg_sent, df_lr_all_samp[["sentenceid", "manifestoid", "party", "year"]].drop_duplicates(), on="sentenceid", how="inner")), "Make sure that adding meta-data to manifesto-level aggretated data does not remove any data"
df_lr_all_aggreg_sent = pd.merge(df_lr_all_aggreg_sent, df_lr_all_samp[["sentenceid", "manifestoid", "party", "year"]].drop_duplicates(), on="sentenceid", how="inner")

## aggregate per manifesto, social/economic ("scale") and type of annotator (source)
df_lr_all_aggreg_manif = df_lr_all_aggreg_sent.groupby(by=["manifestoid", "scale", "source"], as_index=False, group_keys=False).apply(lambda x: x.score_sent_mean.mean())
df_lr_all_aggreg_manif = df_lr_all_aggreg_manif.rename(columns={None: "score_manif_mean"})
# remove annotations for None (no topic)
df_lr_all_aggreg_manif = df_lr_all_aggreg_manif[df_lr_all_aggreg_manif.scale != "None"]
# disaggregate manifesto id to party and year in separate columns
df_lr_all_aggreg_manif[["party", "year"]] = df_lr_all_aggreg_manif["manifestoid"].str.split(' ', expand=True)

# sort to make sure that values are properly aligned for correlations
df_lr_all_aggreg_manif = df_lr_all_aggreg_manif.sort_values(by=["scale", "source", "party", "year"])

# separate crowd annotations and expert annotations
df_lr_aggreg_crowd = df_lr_all_aggreg_manif[df_lr_all_aggreg_manif.source == "Crowd"]
df_lr_aggreg_experts = df_lr_all_aggreg_manif[df_lr_all_aggreg_manif.source == "Experts"]


### correlation of crowd vs. expert annotations
corr_dic_lr = {"dimension": [], "correlation": [], "p-value": []}
for domain in ["Economic", "Social"]:
    pearson_cor = pearsonr(df_lr_aggreg_crowd[df_lr_aggreg_crowd.scale == domain].score_manif_mean, df_lr_aggreg_experts[df_lr_aggreg_experts.scale == domain].score_manif_mean)
    corr_dic_lr["dimension"].append(domain)
    corr_dic_lr["correlation"].append(pearson_cor[0])
    corr_dic_lr["p-value"].append(pearson_cor[1])
    #corr_dic_lr.update({f"corr_{domain.lower()}": pearson_cor[0], f"p-value-{domain.lower()}": pearson_cor[1]})  # f"p-value-labels_all": pearson_cor[1]

# correlation values are probably different, because crowd data includes more annotations per sentence than table in paper
print(corr_dic_lr)
df_corr_lr = pd.DataFrame(corr_dic_lr)


### load external expert surveys
# seems to be in appendix section 3 in .pdf, but don't find it in repo
# ...


## try  external CHES survey data
# Benoit does not use CHES, but these: "Laver and Hunt (1992); Laver (1998) for 1997; Benoit and Laver (2006) for 2001; Benoit (2005, 2010) for 2005 and 2010."
# cannot find the correct data on github. Below some test loading code for CHES.
"""# aggregate CHES data for 1999 - 2019
df_ches = pd.read_csv("https://www.chesdata.eu/s/1999-2019_CHES_dataset_meansv3.csv")
df_ches.columns
df_ches_cl = df_ches.copy(deep=True)

# only UK manifestos:
# country code map p.2 in codebook https://static1.squarespace.com/static/5975c9bfdb29d6a05c65209b/t/62e9612a9090ba609c4b8693/1659461930880/1999-2019_CHES_codebook.pdf
df_ches_cl = df_ches_cl[df_ches_cl.country == 11]
# harmonise party ids:
ches_to_benoit_party_map = {
    "CONS": "Con", "LibDem": "LD", "LAB": "Lab"
}
df_ches_cl["party_name_harmonised"] = df_ches_cl.party.map(ches_to_benoit_party_map)
# remove parties that are not included in crowd data
df_ches_cl = df_ches_cl[~df_ches_cl.party_name_harmonised.isna()]
# remove years where no crowd data is available
df_ches_cl.year.unique()  #array([1999, 2002, 2006, 2010, 2014, 2019])
#df_lr_aggreg_crowd.year.unique()  #array(['1987', '1992', '1997', '2001', '2005', '2010'], dtype=object)
ches_to_benoit_year_map = {
    2002: 2001, 2006: 2005, 2010: 2010
}
df_ches_cl["year_harmonised"] = df_ches_cl.year.map(ches_to_benoit_year_map)
df_ches_cl = df_ches_cl[~df_ches_cl.year_harmonised.isna()]


# add older Ray-Marks-Steenbergen surveys from 1984 - 1996
# https://www.chesdata.eu/ches-europe
df_ray = pd.read_csv("https://www.chesdata.eu/s/1984-1999_dataset_means.csv")
df_ray_cl = df_ray.copy(deep=True)
# only UK manifestos
df_ray_cl = df_ray_cl[df_ray_cl.country == 11]
# only selected parties from crowd data
ray_to_benoit_party_map = {
    "TORY": "Con", "LDP": "LD", "LAB": "Lab"
}
df_ray_cl["party_name_harmonised"] = df_ray_cl.party.map(ray_to_benoit_party_map)
# remove parties that are not included in crowd data
df_ray_cl = df_ray_cl[~df_ray_cl.party_name_harmonised.isna()]
# harmonise years and remove years, where no crowd data is available
df_ray_cl.year.unique()  # #array([1984, 1988, 1992, 1996, 1999])
#df_lr_aggreg_crowd.year.unique()  #array(['1987', '1992', '1997', '2001', '2005', '2010'], dtype=object)
ray_to_benoit_year_map = {
    1988: 1987, 1992: 1992, 1996: 1997
}
df_ray_cl["year_harmonised"] = df_ray_cl.year.map(ray_to_benoit_year_map)
df_ray_cl = df_ray_cl[~df_ray_cl.year_harmonised.isna()]

df_ches_ray = pd.concat([df_ches_cl, df_ray_cl])

## how does ray data before 1999 map to ches data after (including) 1999?
# !! before 1999, ray only has general left-right question on scale from 0-1 "lr_gen"
# Benoit took other data, not Ray/CHES: "Laver and Hunt (1992); Laver (1998) for 1997; Benoit and Laver (2006) for 2001; Benoit (2005, 2010) for 2005 and 2010."
#df_ches["lrgen", "lrecon", "lrecon_salience", "galtan", "galtan_salience"]
#df_ches["eu_asylum", "immigrate_policy", "immigrate_salience"]
"""


### write cleaned data to disc
## add sentence texts
df_lr_sents = pd.read_csv("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20Created/master.sentences.csv").drop(columns=["Unnamed: 0"])
df_lr_all_aggreg_sent_full = df_lr_all_aggreg_sent.merge(df_lr_sents[["sentenceid", "manifestoid", "sentence_text", "pre_sentence", "post_sentence", "policy_area_gold", "econ_scale_gold", "soc_scale_gold", "truncated"]], on=["sentenceid", "manifestoid"], how="left")
df_lr_all_aggreg_sent_full = df_lr_all_aggreg_sent_full.rename(columns={"sentence_text": "text_original", "pre_sentence": "text_preceding", "post_sentence": "text_following"})
assert len(df_lr_all_aggreg_sent) == len(df_lr_all_aggreg_sent_full)

## sentence level train/test data
df_lr_all_aggreg_sent_full.to_csv(f"./data-clean/benoit_leftright_sentences.zip",
                             compression={"method": "zip", "archive_name": f"benoit_leftright_sentences.csv"},
                             index=False)
df_lr_all_aggreg_manif.to_csv(f"./data-clean/benoit_leftright_manifestos_gold.zip",
                             compression={"method": "zip", "archive_name": f"benoit_leftright_manifestos_gold.csv"},
                             index=False)



##### 2. Analysis: Immigration policy positions
# !! issue: too few sentences on immigration that train-test split for getting predictions is impossible. way too imbalanced
# => dataset not really usable for sml analysis (only active learning could make sense, can solve sampling/imbalance issue)

r = requests.get('https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20CF%20jobs/immigration/CFjobsresultsImmigration.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
print(zipfile.ZipFile.namelist(z))

# two files probably on same data/sentences, only in different year with different coders to demonstrate reproducibility
df_immig = pd.read_csv(z.open("f354277_immigration1.csv"))
#df_immig = pd.read_csv(z.open("f389381_immigration2.csv"))

df_immig_cl = df_immig.copy(deep=True)

# remove sentences that are not about immigration ?
df_immig_cl.policy_area.value_counts()
#df_immig_cl = df_immig_cl[df_immig_cl.policy_area == 4]
# remove screeners
df_immig_cl = df_immig_cl[df_immig_cl.manifestoid != "screener"]
# not clear from paper that/why they also annotate manifestos from other countries. but they do.
df_immig_cl = df_immig_cl[df_immig_cl._country == "GBR"]

# coding in data: -1 (left, pro-immig), 1 (right, anti-imig), 0 (neutral); this is also in the HTML in appendix p26
# weird: coding instruction appendix p26 mention "5-point scale", scale in paper figure 6 is -4 to 4
# ! also: very few sentences per manifesto about migration (96% about immigration not accoring to paper, p.290)
df_immig_cl.immigr_scale.value_counts()
df_immig_cl.immigr_scale_gold.value_counts()
#df_immig_cl.policy_area.value_counts()
#df_immig_cl.manifestoid.value_counts()

## aggregate
# aggregate annotations on sentence level
# not necessary, no sentence has been annotated more than once.
#test = df_immig_cl.groupby(by=["sentenceid"]).apply(lambda x: x.sample(n=min(len(x), 10), random_state=42)).reset_index(drop=True)
assert 0 == sum(df_immig_cl.sentenceid.duplicated()), "Careful, a sentence has been annotated more than once"
# aggregate annotations on manifesto level
df_immig_aggreg = df_immig_cl.groupby("manifestoid", as_index=False, group_keys=False).apply(lambda x: x.immigr_scale.mean(skipna=True))
# remove coalition manifesto which is not in CHES
df_immig_aggreg = df_immig_aggreg[df_immig_aggreg.manifestoid != "Coalition 2010"]

# harmonise party names between crowd data and external data
uk_2010_map = {
    "BNP 2010": "BNP", "Con 2010": "CON", "Greens 2010": "GREEN",
    "LD 2010": "LIB", "Lab 2010": "LAB", "PC 2010": "PLAID",
    "SNP 2010": "SNP", "UKIP 2010": "UKIP"
}

df_immig_aggreg["party"] = df_immig_aggreg.manifestoid.map(uk_2010_map)
df_immig_aggreg = df_immig_aggreg.sort_values(by="party")
df_immig_aggreg = df_immig_aggreg.rename(columns={None: "score_manif_mean"})



#### external data

### benoits survey on migration, figure 6
# I get the same results as in the paper here. Benoits unpublished expert survey seems to be the following.
# the relevant values seems to be "Immigration" and "Position"
df_benoit2 = pd.read_stata("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20External/GBexpertscores2.dta")
#df_benoit1 = pd.read_stata("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20External/GBexpertscores.dta")

# only use specific year from paper
df_benoit2 = df_benoit2[df_benoit2.year == 2010]

# align party labels from crowd data and external data
uk_2010_map_benoit = {
    "British National Party": "BNP", "Conservative Party": "CON", "Green Party of England and Wales": "GREEN",
    "Liberal Democrats": "LIB", "Labour Party": "LAB", "Plaid Cymru": "PLAID",
    "Scottish National Party": "SNP", "UK Independence Party": "UKIP",
    "Scottish Socialist Party": np.nan,
}
df_benoit2["party"] = df_benoit2.party_name_english.map(uk_2010_map_benoit)
df_benoit2 = df_benoit2[~df_benoit2.party.isna()]

# sort values to align value order to correlations
df_benoit2 = df_benoit2.sort_values(by=["party", "dimension", "scale"])


## correlate crowd vs. benoit expert survey
corr_dic_immig_2 = {"dimension": [], "correlation": [], "p-value": []}
for dimension in df_benoit2.dimension.unique():
    for concept in ["Importance", "Position"]:
        pearson_cor = pearsonr(df_immig_aggreg.score_manif_mean, df_benoit2[(df_benoit2.scale == concept) & (df_benoit2.dimension == dimension)]["mean"])
        corr_dic_immig_2["dimension"].append(dimension + "-" + concept)
        corr_dic_immig_2["correlation"].append(pearson_cor[0])
        corr_dic_immig_2["p-value"].append(pearson_cor[1])
        #corr_dic_immig_2.update({f"corr_{dimension + '_' + concept}": pearson_cor[0], f"p-value_{dimension + '_' + concept}": pearson_cor[1]})  # f"p-value-labels_all": pearson_cor[1]

print(corr_dic_immig_2)
df_corr_immig_2 = pd.DataFrame(corr_dic_immig_2)


### CHES
# data not well documented. Have to assume that "Mean" column represents the mean score from all experts
# potential message to Benoit:
"""To calculate the exert survey score for the correlation with the crowd score, I tried to take the average of the 
position towards asylum and the position towards immigration (taking the scores from the "Mean" column), but then I do 
not get the same correlation values as in table 4. (I'm following footnote 33: "CHES included two highly correlated measures, 
one aimed at “closed or open” immigration policy another aimed at policy toward asylum seekers and whether immigrants should 
be integrated into British society. Our measure averages the two. Full numerical results are given in the Online Appendix, 
Section 3."  (section 3 in the appendix seem to contain information on the first analysis))"""
#df_ches = pd.read_stata("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20External/CHES_2010_expert_data_public.dta")
df_ches_aggreg_manif = pd.read_stata("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20External/CHexpertsurveyscores.dta")

# adding rows with mean of immigration and asylum score from experts as described in footnote 33
df_ches_immig_asyl = df_ches_aggreg_manif[(df_ches_aggreg_manif.dimension.isin(["Immigration", "Asylum"])) & (df_ches_aggreg_manif.scale == "Position")]
df_ches_immig_asyl = df_ches_immig_asyl.groupby("party", as_index=False, group_keys=False).apply(lambda x: x.Mean.mean())
df_ches_immig_asyl = df_ches_immig_asyl.rename(columns={None: "Mean"})
df_ches_immig_asyl["dimension"] = "immig_asyl"
df_ches_immig_asyl["scale"] = "Position"

df_ches_aggreg_manif = pd.concat([df_ches_aggreg_manif, df_ches_immig_asyl])

# only analyse specific type of questions?
#df_ches_aggreg_manif = df_ches_aggreg_manif[(df_ches_aggreg_manif.dimension == "Immigration")]

# sort to ensure correct sequence of observations for correlation
df_ches_aggreg_manif = df_ches_aggreg_manif.sort_values(by=["party", "dimension", "scale"])

## correlate crowd vs. expert survey
corr_dic_immig_1 = {"dimension": [], "correlation": [], "p-value": []}
for dimension in df_ches_aggreg_manif.dimension.unique():
    for concept in ["Importance", "Position"]:
        if not ((concept == "Importance") and (dimension == "immig_asyl")):  # because importance score not calculated for new immig_asyl rows
            pearson_cor = pearsonr(df_immig_aggreg.score_manif_mean, df_ches_aggreg_manif[(df_ches_aggreg_manif.scale == concept) & (df_ches_aggreg_manif.dimension == dimension)].Mean)
            corr_dic_immig_1["dimension"].append(dimension + "-" + concept)
            corr_dic_immig_1["correlation"].append(pearson_cor[0])
            corr_dic_immig_1["p-value"].append(pearson_cor[1])
            #corr_dic_immig_1.update({f"corr_{dimension + '_' + concept}": pearson_cor[0], f"p-value_{dimension + '_' + concept}": pearson_cor[1]})  # f"p-value-labels_all": pearson_cor[1]

print(corr_dic_immig_1)
df_corr_immig_1 = pd.DataFrame(corr_dic_immig_1)



### write cleaned training and gold data to disc
# create cleaned, sentence level df with annotations
df_immig_sents = df_immig_cl[["immigr_scale", "policy_area", "sentenceid", "manifestoid", "immigr_scale_gold", "policy_area_gold", "sentence_text", "pre_sentence", "post_sentence"]]
df_immig_sents = df_immig_sents.rename(columns={"sentence_text": "text_original", "pre_sentence": "text_preceding", "post_sentence": "text_following"})
# !! issue: too few sentences on immigration that train-test split for getting predictions is impossible
df_immig_sents.immigr_scale.value_counts(dropna=False)
# sentence level train/test data
df_immig_sents.to_csv(f"./data-clean/benoit_immigr_sentences.zip",
                      compression={"method": "zip", "archive_name": f"benoit_immigr_sentences.csv"},
                      index=False)
## gold data on manifesto level
df_benoit2.to_csv(f"./data-clean/benoit_immigr_external_benoit.zip",
                             compression={"method": "zip", "archive_name": f"benoit_immigr_external_benoit.csv"},
                             index=False)
df_ches_aggreg_manif.to_csv(f"./data-clean/benoit_immigr_external_ches.zip",
                             compression={"method": "zip", "archive_name": f"benoit_immigr_external_ches.csv"},
                             index=False)





#####  Analysis 3:  EP debate votes
# !! probably also way too imbalanced and too little data for train-test split to produce predictions

## files that do not seem to contain the crowd data:
#r = requests.get('https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20Created/coalsentences.zip')
#z = zipfile.ZipFile(io.BytesIO(r.content))
#print(zipfile.ZipFile.namelist(z))
# do not understand this disaggregated data. .csv files unclearly named, content unclear (no crowd annotations?)
#df_coal_gold = pd.read_csv(z.open("coalsentences_EN_GOLD.csv"))
#df_coal_crowd = pd.read_csv(z.open("coalsentences_EN_CF.csv"))

## files that probably contain crowd data:
# based on analysis script  https://github.com/kbenoit/CSTA-APSR/blob/master/06%20Produce%20Paper%20Tables%20and%20Figures/Replicate_Figure_7.R
# the raw data seems to be hidden in this generic .zip file: https://github.com/kbenoit/CSTA-APSR/blob/master/Data%20-%20CF%20jobs/CFjobresults.zip
# ...

## somewhat cleaned excel file with interpretable aggregate data
df_coal = pd.read_excel("https://github.com/kbenoit/CSTA-APSR/raw/master/Data%20-%20External/coal-debate.xlsx")
df_coal = df_coal.rename(columns={"Unnamed: 14": "other", "Unnamed: 15": "interpretation_results"})

# 1137 is probably the correct column with final vote values based on https://github.com/kbenoit/CSTA-APSR/blob/master/07%20Additional%20Texts/coalmines_02_Analyze_Results.R
# remove nan for column that probably represents final votes
df_coal = df_coal[~df_coal["VoteID-1137"].isna()]

# !! unclear how to transform the data, align categorical vote values to continuous crowd scores

corr_dic_coal = {"correlation": [], "p-value": []}
for language in ["?"]:
    pearson_cor = pearsonr(df_coal.subsidyMean, df_coal["VoteID-1137"])
    corr_dic_coal["correlation"].append(pearson_cor[0])
    corr_dic_coal["p-value"].append(pearson_cor[1])
    #corr_dic_immig_2.update({f"corr_{dimension + '_' + concept}": pearson_cor[0], f"p-value_{dimension + '_' + concept}": pearson_cor[1]})  # f"p-value-labels_all": pearson_cor[1]

print(corr_dic_coal)

