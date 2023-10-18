

import sys
if sys.stdin.isatty():
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.stdin.isatty(): ", sys.stdin.isatty(), ")")


# ## Load packages
import transformers
import datasets
import torch
import optuna

import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
import random
import os
import tqdm
from collections import OrderedDict
from datetime import date
import time
import joblib
import ast

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import svm, naive_bayes, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer

import spacy


## set global seed for reproducibility and against seed hacking
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

print(os.getcwd())
if (EXECUTION_TERMINAL==False) and ("multilingual-repo" not in os.getcwd()):
    os.chdir("./multilingual-repo")
print(os.getcwd())



# ## Main parameters

### argparse for command line execution
import argparse
# https://realpython.com/command-line-interfaces-python-argparse/
# https://docs.python.org/3/library/argparse.html

# Create the parser
parser = argparse.ArgumentParser(description='Do final run with best hyperparameters (on different languages, datasets, algorithms)')

## Add the arguments
# arguments only for test script
parser.add_argument('-cvf', '--n_cross_val_final', type=int, default=3,
                    help='For how many different random samples should the algorithm be tested at a given sample size?')
parser.add_argument('-zeroshot', '--zeroshot', action='store_true',
                    help='Start training run with a zero-shot run')

# arguments for both hyperparam and test script
parser.add_argument('-lang', '--languages', type=str, nargs='+',
                    help='List of languages to iterate over. e.g. "en", "de", "es", "fr", "ko", "tr", "ru" ')
parser.add_argument('-anchor', '--language_anchor', type=str,
                    help='Anchor language to translate all texts to if using anchor. Default is "en"')
parser.add_argument('-language_train', '--language_train', type=str,
                    help='What language should the training set be in?. Default is "en"')
parser.add_argument('-augment', '--augmentation_nmt', type=str,
                    help='Whether and how to augment the data with machine translation (MT).')

parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "manifesto-8" ')
parser.add_argument('-samp', '--sample_interval', type=int, nargs='+',
                    help='Interval of sample sizes to test.')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "classical_ml"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings-en" or "embeddings-multi"')
parser.add_argument('-hp_date', '--hyperparam_study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')




### choose arguments depending on execution in terminal or in script for testing
if EXECUTION_TERMINAL == True:
  print("Arguments passed via the terminal:")
  # Execute the parse_args() method
  args = parser.parse_args()
  # To show the results of the given option to screen.
  print("")
  for key, value in parser.parse_args()._get_kwargs():
      #if value is not None:
          print(value, "  ", key)

elif EXECUTION_TERMINAL == False:
  # parse args if not in terminal, but in script
  args = parser.parse_args(["--n_cross_val_final", "2",  #--zeroshot
                            "--dataset", "manifesto-8",
                            "--languages", "en", "de", "es", "fr", "tr", "ru", "ko",
                            "--language_anchor", "en", "--language_train", "en",  # in multiling scenario --language_train needs to be list of lang (?)
                            "--augmentation_nmt", "no-nmt-single",  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
                            "--sample_interval", "300",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "logistic",  # SVM, logistic
                            "--vectorizer", "embeddings-multi",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--hyperparam_study_date", "20221026"])


### args only for test runs
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final
ZEROSHOT = args.zeroshot

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET_NAME = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
LANGUAGES = args.languages
LANGUAGES = LANGUAGES[:3]
LANGUAGE_TRAIN = args.language_train
LANGUAGE_ANCHOR = args.language_anchor
AUGMENTATION = args.augmentation_nmt

N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "classical_ml"
MODEL_NAME = args.model  # "SVM"

HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"





# ## Load data
if "manifesto-8" in DATASET_NAME:
  df_cl = pd.read_csv("./data-clean/df_manifesto_all.csv")
  df_train = pd.read_csv("./data-clean/df_manifesto_train_trans_embed_tfidf.csv")
  df_test = pd.read_csv("./data-clean/df_manifesto_test_trans_embed_tfidf.csv")
else:
  raise Exception(f"Dataset name not found: {DATASET_NAME}")

## special preparation of manifesto simple dataset - chose 8 or 57 labels
if DATASET_NAME == "manifesto-8":
  df_cl["label_text"] = df_cl["label_domain_text"]
  df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]
  df_train["label_text"] = df_train["label_domain_text"]
  df_train["label"] = pd.factorize(df_train["label_text"], sort=True)[0]
  df_test["label_text"] = df_test["label_domain_text"]
  df_test["label"] = pd.factorize(df_test["label_text"], sort=True)[0]
else:
    raise Exception(f"Dataset not defined: {DATASET_NAME}")

print(DATASET_NAME)




## reduce max sample size interval list to fit to max df_train length
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len(df_train)]
if len(df_train) < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len(df_train)]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)

"""
# tests for code above
N_SAMPLE_DEV = [1000]
len_df_train = 500
n_sample_dev_filt = [sample for sample in N_SAMPLE_DEV if sample <= len_df_train]
if len_df_train < N_SAMPLE_DEV[-1]:
  n_sample_dev_filt = n_sample_dev_filt + [len_df_train]
  if len(n_sample_dev_filt) > 1:
    if n_sample_dev_filt[-1] == n_sample_dev_filt[-2]:  # if last two sample sizes are duplicates, delete the last one
      n_sample_dev_filt = n_sample_dev_filt[:-1]
N_SAMPLE_DEV = n_sample_dev_filt
print("Final sample size intervals: ", N_SAMPLE_DEV)"""



LABEL_TEXT_ALPHABETICAL = np.sort(df_cl.label_text.unique())
TRAINING_DIRECTORY = f"results/{DATASET_NAME}"


## data checks
print(DATASET_NAME, "\n")
# verify that numeric label is in alphabetical order of label_text
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)


# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())

import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from helpers import compute_metrics_classical_ml, clean_memory
## functions for scenario data selection and augmentation
from helpers import select_data_for_scenario_hp_search, select_data_for_scenario_final_test, data_augmentation





# ## Final test with best hyperparameters


## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_SVM_tfidf_01000samp_20221006.pkl")

# select best hp based on hp-search
hp_study_dic = {}
for n_sample in N_SAMPLE_DEV:
  while len(str(n_sample)) <= 4:
    n_sample = "0" + str(n_sample)

  if EXECUTION_TERMINAL == True:
        hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
  elif EXECUTION_TERMINAL == False:
        #hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_many2many_embeddings-en_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")
        hp_study_dic_step = joblib.load(f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample}samp_{HYPERPARAM_STUDY_DATE}.pkl")

  hp_study_dic.update(hp_study_dic_step)




if ZEROSHOT == False:
  N_SAMPLE_TEST = N_SAMPLE_DEV * len(LANGUAGES)
  print(N_SAMPLE_TEST)

  HYPER_PARAMS_LST = [study_value['optuna_study'].best_trial.user_attrs["hyperparameters_all"] for study_key, study_value in hp_study_dic.items()]

  # hypothesis template: always simple without context and without NLI for this paper
  #HYPOTHESIS_TEMPLATE_LST = [hyperparams_dic["hypothesis_template"] for hyperparams_dic in HYPER_PARAMS_LST]  #if ("context" in hyperparams_dic["hypothesis_template"])
  #HYPOTHESIS_TEMPLATE_LST = HYPOTHESIS_TEMPLATE_LST * len(LANGUAGES)
  HYPOTHESIS_TEMPLATE_LST = ["template_not_nli"] * len(LANGUAGES)
  print(HYPOTHESIS_TEMPLATE_LST)

  HYPER_PARAMS_LST = [{key: dic[key] for key in dic if key!="hypothesis_template"} for dic in HYPER_PARAMS_LST]  # return dic with all elements, except hypothesis template
  HYPER_PARAMS_LST_TEST = HYPER_PARAMS_LST
  HYPER_PARAMS_LST_TEST = HYPER_PARAMS_LST_TEST * len(LANGUAGES)
  print(HYPER_PARAMS_LST_TEST)

else:
    raise Exception("zero-shot classification not implemented")


## intermediate text formatting function for testing code on manifesto-8 without NLI and without context
"""def format_text(df=None, text_format=None, embeddings=VECTORIZER, translated_text=True):
    # ! translated_text is for df_test (false) or not df_train (true)
    # ! review 'translated_text' - still up to date with updated code? can be removed?
    if (text_format == 'template_not_nli') and (translated_text == False) and (embeddings == "tfidf"):
        df["text_prepared"] = df.text_original
    elif (text_format == 'template_not_nli') and (translated_text == True) and (embeddings == "tfidf"):
        df["text_prepared"] = df.text_original_trans
    elif (text_format == 'template_not_nli') and (embeddings == "embeddings-en"):
        df["text_prepared"] = df.text_original_trans_embed_en
    elif (text_format == 'template_not_nli') and (embeddings == "embeddings-multi"):
        df["text_prepared"] = df.text_original_trans_embed_multi
    else:
        raise Exception(f'format_text did not work for text_format == {text_format}, vectorizer == {embeddings}, translated_text == {translated_text}')

    # ! special case for no-nmt-multi, don't have monolingual models - taking multilingual embeddings on single languages as proxy
    if (text_format == 'template_not_nli') and (embeddings == "embeddings-en") and (AUGMENTATION == "no-nmt-many"):
        df["text_prepared"] = df.text_original_trans_embed_multi

    return df.copy(deep=True)"""


#### tests with fairlearn
### correlation remover
# https://fairlearn.org/main/user_guide/mitigation.html#correlation-remover
"""from fairlearn.preprocessing import CorrelationRemover

## augment relevant meta-data
mapping_parfam = {10: "ECO: Ecological parties", 20: "LEF: Socialist or other left parties",
                  30: "SOC: Social democratic parties", 40: "LIB: Liberal parties", 50: "CHR: Christian democratic parties (in Isreal also Jewish parties)",
                  60: "CON: Conservative parties", 70: "NAT: Nationalist parties", 80: "AGR: Agrarian parties",
                  90: "ETH: Ethnic and regional parties", 95: "SIP: Special issue parties", 98: "DIV: Electoral alliances of diverse origin without dominant party",
                  999: "MI: Missing information"
}
df_train["parfam_name"] = df_train.parfam.map(mapping_parfam)
df_train["parfam_rile"] = ["left" if parfam in [10, 20, 30] else "right" if parfam in [50, 60, 70, 80, 90] else "other" for parfam in df_train["parfam"]]
cmp_code_left = [103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701]
cmp_code_right = [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606]
cmp_code_other = np.unique([cmp_code for cmp_code in df_train["cmp_code"] if cmp_code not in cmp_code_left + cmp_code_right])
df_train["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df_train["cmp_code"]]
df_train["decade"] = [str(date)[:3]+"0" for date in df_train.date]

df_cl["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df_test["cmp_code"]]
df_cl["label_rile"].value_counts()
df_train["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df_train["cmp_code"]]
df_train["label_rile"].value_counts()
df_test["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df_test["cmp_code"]]
df_test["label_rile"].value_counts()

df_embed_lst = pd.DataFrame([ast.literal_eval(lst) for lst in df_train.text_original_trans_embed_multi.astype('object')])
df_embed_lst["label_rile"] = pd.factorize(df_train.label_rile.tolist())[0]

X = df_embed_lst

cr = CorrelationRemover(sensitive_feature_ids=['label_rile'], alpha=1)
cr.fit(X)
X_transform = cr.transform(X)

df_train["text_original_trans_embed_multi"] = X_transform.tolist()
"""


### run random cross-validation for hyperparameter search without a dev set
np.random.seed(SEED_GLOBAL)

### K example intervals loop
experiment_details_dic = {}
for lang, n_max_sample, hyperparams, hypothesis_template in tqdm.tqdm(zip(LANGUAGES, N_SAMPLE_TEST, HYPER_PARAMS_LST_TEST, HYPOTHESIS_TEMPLATE_LST), desc="Iterations for different number of texts", leave=True):
  np.random.seed(SEED_GLOBAL)
  t_start = time.time()  # log how long training of model takes

  # ! put in function
  ### select correct language for train sampling and test
  ## Two different language scenarios
  if "no-nmt-single" in AUGMENTATION:
      df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
      df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
      if "multi" not in VECTORIZER:
          raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
  elif "one2anchor" in AUGMENTATION:
      df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
      # for test set - for non-multi, test on translated text, for multi algos test on original lang text
      if "multi" not in VECTORIZER:
          df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
      elif "multi" in VECTORIZER:
          df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
  elif "one2many" in AUGMENTATION:
      # augmenting this with other translations further down after sampling
      df_train_lang = df_train.query("language_iso == @LANGUAGE_TRAIN & language_iso_trans == @LANGUAGE_TRAIN").copy(deep=True)
      # for test set - for multi algos test on original lang text
      if "multi" in VECTORIZER:
          df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
      elif "multi" not in VECTORIZER:
          raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")

  ## many2X scenarios
  elif "no-nmt-many" in AUGMENTATION:
      # separate analysis per lang if not multi
      if "multi" not in VECTORIZER:
          df_train_lang = df_train.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
          df_test_lang = df_test.query("language_iso == @lang & language_iso_trans == @lang").copy(deep=True)
      # multilingual models can analyse all original texts here
      elif "multi" in VECTORIZER:
          df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
          df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
  elif "many2anchor" in AUGMENTATION:
      if "multi" not in VECTORIZER:
          df_train_lang = df_train.query("language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
          df_test_lang = df_test.query("language_iso_trans == @LANGUAGE_ANCHOR").copy(deep=True)
      # multilingual models can analyse all original texts here. augmented below with anchor lang
      elif "multi" in VECTORIZER:
          df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
          df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
  elif "many2many" in AUGMENTATION:
      # multilingual models can analyse all original texts here. can be augmented below with all other translations
      if "multi" in VECTORIZER:
          df_train_lang = df_train.query("language_iso == language_iso_trans").copy(deep=True)
          df_test_lang = df_test.query("language_iso == language_iso_trans").copy(deep=True)
      elif "multi" not in VECTORIZER:
          raise Exception(f"Cannot use {VECTORIZER} if augmentation is {AUGMENTATION}")
  else:
      raise Exception("Issue with AUGMENTATION")


  # prepare loop
  k_samples_experiment_dic = {"method": METHOD, "language_source": lang, "language_anchor": LANGUAGE_ANCHOR, "n_max_sample": n_max_sample, "model": MODEL_NAME, "vectorizer": VECTORIZER, "augmentation": AUGMENTATION, "hyperparams": hyperparams}  # "trainer_args": train_args, "hypotheses": HYPOTHESIS_TYPE, "dataset_stats": dataset_stats_dic
  f1_macro_lst = []
  f1_micro_lst = []
  accuracy_balanced_lst = []
  # randomness stability loop. Objective: calculate F1 across N samples to test for influence of different (random) samples
  np.random.seed(SEED_GLOBAL)
  for random_seed_sample in tqdm.tqdm(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_FINAL), desc="iterations for std", leave=True):

    ## sampling
    if n_max_sample == 999_999:  # all data, no sampling
      df_train_samp = df_train_lang.copy(deep=True)
    # same sample size per language for multiple language data scenarios
    elif any(augmentation in AUGMENTATION for augmentation in ["no-nmt-many", "many2anchor", "many2many"]):
      df_train_samp = df_train_lang.groupby(by="language_iso").apply(lambda x: x.sample(n=min(n_max_sample, len(x)), random_state=random_seed_sample).copy(deep=True))
    # one sample size for single language data scenario
    else:
      df_train_samp = df_train_lang.sample(n=min(n_max_sample, len(df_train_lang)), random_state=random_seed_sample).copy(deep=True)

    if n_max_sample == 0:  # only one inference necessary on same test set in case of zero-shot
      metric_step = {'accuracy_balanced': 0, 'accuracy_not_b': 0, 'f1_macro': 0, 'f1_micro': 0}
      f1_macro_lst.append(0)
      f1_micro_lst.append(0)
      accuracy_balanced_lst.append(0)
      k_samples_experiment_dic.update({"n_train_total": len(df_train_samp), f"metrics_seed_{random_seed_sample}": metric_step})
      break


    ### data augmentation on sample for multiling models + translation scenarios
    # general function - common with hp-search script
    df_train_samp_augment = data_augmentation(df_train_scenario_samp=df_train_samp, df_train=df_train)
    """## single language text scenarios
    sample_sent_id = df_train_samp.sentence_id.unique()
    if AUGMENTATION == "no-nmt-single":
        df_train_samp_augment = df_train_samp.copy(deep=True)
    elif AUGMENTATION == "one2anchor":
        if "multi" not in VECTORIZER:
            df_train_samp_augment = df_train_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # augment by combining texts from train language with texts from train translated to target language
            df_train_augment = df_train[(((df_train.language_iso == LANGUAGE_TRAIN) & (df_train.language_iso_trans == LANGUAGE_TRAIN)) | ((df_train.language_iso == LANGUAGE_TRAIN) & (df_train.language_iso_trans == lang)))].copy(deep=True)
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)
        else:
            raise Exception("Issue with VECTORIZER")
    elif AUGMENTATION == "one2many":
        if "multi" in VECTORIZER:
            df_train_augment = df_train.query("language_iso == @LANGUAGE_TRAIN").copy(deep=True)  # can use all translations and original text (for single train lang) here
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        else:
            raise Exception("AUGMENTATION == 'X2many' only works for multilingual vectorization")
    ## multiple language text scenarios
    elif AUGMENTATION == "no-nmt-many":
        df_train_samp_augment = df_train_samp.copy(deep=True)
    elif AUGMENTATION == "many2anchor":
        if "multi" not in VECTORIZER:
            df_train_samp_augment = df_train_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # already have all original languages in the scenario. augmenting it with translated (to anchor) texts here. e.g. for 7*6=3500 original texts, adding 6*6=3000 texts, all translated to anchor (except anchor texts)
            df_train_augment = df_train[(df_train.language_iso == df_train.language_iso_trans) | (df_train.language_iso_trans == LANGUAGE_ANCHOR)].copy(deep=True)
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
    elif AUGMENTATION == "many2many":
        if "multi" not in VECTORIZER:
            df_train_samp_augment = df_train_samp.copy(deep=True)
        elif "multi" in VECTORIZER:
            # already have all original languages in the scenario. augmenting it with all other translated texts
            df_train_augment = df_train.copy(deep=True)
            df_train_samp_augment = df_train_augment[df_train_augment.sentence_id.isin(sample_sent_id)].copy(deep=True)  # training on both original text and anchor
        #else:
        #    raise Exception(f"AUGMENTATION == {AUGMENTATION} only works for multilingual vectorization")"""


    # chose the text format depending on hyperparams
    # returns "text_prepared" column, e.g. with concatenated sentences
    df_train_samp_augment = format_text(df=df_train_samp_augment, text_format=hypothesis_template, embeddings=VECTORIZER, translated_text=False)
    df_train_samp_augment = df_train_samp_augment.sample(frac=1.0, random_state=random_seed_sample)  # shuffle df_train
    df_test_formatted = format_text(df=df_test_lang, text_format=hypothesis_template, embeddings=VECTORIZER, translated_text=True)

    # separate hyperparams for vectorizer and classifier.
    hyperparams_vectorizer = {key: value for key, value in hyperparams.items() if key in ["ngram_range", "max_df", "min_df"]}
    hyperparams_clf = {key: value for key, value in hyperparams.items() if key not in ["ngram_range", "max_df", "min_df"]}

    # Vectorization
    if VECTORIZER == "tfidf":
        # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', norm="l2", use_idf=True, smooth_idf=True, analyzer="word", **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=1.0, min_df=10
        vectorizer.fit(pd.concat([df_train_samp_augment.text_prepared, df_test_formatted.text_prepared]))
        X_train = vectorizer.transform(df_train_samp_augment.text_prepared)
        X_test = vectorizer.transform(df_test_formatted.text_prepared)
    if "embeddings" in VECTORIZER:
        X_train = np.array([ast.literal_eval(lst) for lst in df_train_samp_augment.text_prepared.astype('object')])
        X_test = np.array([ast.literal_eval(lst) for lst in df_test_formatted.text_prepared.astype('object')])


    y_train = df_train_samp_augment.label
    y_test = df_test_formatted.label

    # training on train set sample
    if MODEL_NAME == "SVM":
        clf = svm.SVC(**hyperparams_clf)
    elif MODEL_NAME == "logistic":
        clf = linear_model.LogisticRegression(**hyperparams_clf)
    clf.fit(X_train, y_train)


    # prediction on test set
    label_gold = y_test
    label_pred = clf.predict(X_test)

    # metrics
    metric_step = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df_cl.label_text.unique()))

    k_samples_experiment_dic.update({"n_train_total": len(df_train_samp), f"metrics_seed_{random_seed_sample}": metric_step})
    f1_macro_lst.append(metric_step["eval_f1_macro"])
    f1_micro_lst.append(metric_step["eval_f1_micro"])
    accuracy_balanced_lst.append(metric_step["eval_accuracy_balanced"])

    if (n_max_sample == 0) and (n_max_sample == 999_999):  # only one inference necessary on same test set in case of zero-shot or full dataset
      break

  # timer
  t_end = time.time()
  t_total = round(t_end - t_start, 2)
  t_total = t_total / CROSS_VALIDATION_REPETITIONS_FINAL  # average of all random seed runs

  ## calculate aggregate metrics across random runs
  f1_macro_mean = np.mean(f1_macro_lst)
  f1_micro_mean = np.mean(f1_micro_lst)
  accuracy_balanced_mean = np.mean(accuracy_balanced_lst)
  f1_macro_std = np.std(f1_macro_lst)
  f1_micro_std = np.std(f1_micro_lst)
  accuracy_balanced_std = np.std(accuracy_balanced_lst)
  # add aggregate metrics to overall experiment dict
  metrics_mean = {"f1_macro_mean": f1_macro_mean, "f1_micro_mean": f1_micro_mean, "accuracy_balanced_mean": accuracy_balanced_mean, "f1_macro_std": f1_macro_std, "f1_micro_std": f1_micro_std, "accuracy_balanced_std": accuracy_balanced_std}
  k_samples_experiment_dic.update({"metrics_mean": metrics_mean, "dataset": DATASET_NAME, "n_classes": len(df_cl.label_text.unique()), "train_eval_time_per_model": t_total})

  # harmonise n_sample file title
  while len(str(n_max_sample)) <= 4:
    n_max_sample = "0" + str(n_max_sample)

  # update of overall experiment dic
  experiment_details_dic_step = {f"experiment_sample_{n_max_sample}_{METHOD}_{MODEL_NAME}_{lang}": k_samples_experiment_dic}
  experiment_details_dic.update(experiment_details_dic_step)


  ## stop loop for multiple language case - no separate iterations per language necessary
  if "many2anchor" in AUGMENTATION:
    break
  if ("multi" in VECTORIZER) and (any(augmentation in AUGMENTATION for augmentation in ["no-nmt-many", "many2many"])):  # "no-nmt-single", "one2anchor", "one2many", "no-nmt-many", "many2anchor", "many2many"
    break



### summary dictionary across all languages
experiment_summary_dic = {"experiment_summary":
     {"dataset": DATASET_NAME, "sample_size": N_SAMPLE_DEV, "method": METHOD, "model_name": MODEL_NAME, "vectorizer": VECTORIZER, "lang_anchor": LANGUAGE_ANCHOR, "lang_train": LANGUAGE_TRAIN, "lang_all": LANGUAGES, "augmentation": AUGMENTATION}
 }
# calculate averages across all languages
f1_macro_lst_mean = []
f1_micro_lst_mean = []
accuracy_balanced_lst_mean = []
f1_macro_lst_mean_std = []
f1_micro_lst_mean_std = []
accuracy_balanced_lst_mean_std = []
for experiment_key in experiment_details_dic:
    f1_macro_lst_mean.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_mean'])
    f1_micro_lst_mean.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_mean'])
    accuracy_balanced_lst_mean.append(experiment_details_dic[experiment_key]['metrics_mean']['accuracy_balanced_mean'])
    f1_macro_lst_mean_std.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_std'])
    f1_micro_lst_mean_std.append(experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_std'])
    accuracy_balanced_lst_mean_std.append(experiment_details_dic[experiment_key]['metrics_mean']['accuracy_balanced_std'])
    #print(f"{experiment_key}: f1_macro: {experiment_details_dic[experiment_key]['metrics_mean']['f1_macro_mean']} , f1_micro: {experiment_details_dic[experiment_key]['metrics_mean']['f1_micro_mean']} , accuracy_balanced: {experiment_details_dic[experiment_key]['metrics_mean']['accuracy_balanced_mean']}")
f1_macro_lst_mean = np.mean(f1_macro_lst_mean)
f1_micro_lst_mean = np.mean(f1_micro_lst_mean)
accuracy_balanced_lst_mean = np.mean(accuracy_balanced_lst_mean)
f1_macro_lst_mean_std = np.mean(f1_macro_lst_mean_std)
f1_micro_lst_mean_std = np.mean(f1_micro_lst_mean_std)
accuracy_balanced_lst_mean_std = np.mean(accuracy_balanced_lst_mean_std)

experiment_summary_dic["experiment_summary"].update({"f1_macro_mean": f1_macro_lst_mean, "f1_micro_mean": f1_micro_lst_mean, "accuracy_balanced_mean": accuracy_balanced_lst_mean,
                               "f1_macro_mean_std": f1_macro_lst_mean_std, "f1_micro_mean_std": f1_micro_lst_mean_std, "accuracy_balanced_mean_std": accuracy_balanced_lst_mean_std})

print(experiment_summary_dic)


### save full experiment dic
# merge individual languages experiments with summary dic
experiment_details_dic = {**experiment_details_dic, **experiment_summary_dic}


if EXECUTION_TERMINAL == True:
  joblib.dump(experiment_details_dic, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_max_sample}samp_{AUGMENTATION}_{HYPERPARAM_STUDY_DATE}.pkl")
elif EXECUTION_TERMINAL == False:
  joblib.dump(experiment_details_dic, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_max_sample}samp_{AUGMENTATION}_{HYPERPARAM_STUDY_DATE}_t.pkl")


print("Run done.")





"""
##### single lang experiments
### classical_ml
## no-NMT-single, sentence-embed-multi
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-single', 
'f1_macro_mean': 0.3534645474830353, 'f1_micro_mean': 0.4572575832546144, 'accuracy_balanced_mean': 0.35691265644956, 'f1_macro_mean_std': 0.012277780071643304, 'f1_micro_mean_std': 0.014746932856258629, 'accuracy_balanced_mean_std': 0.010978804500949638}}

## anchor 'en', tfidf, one2anchor
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.2118354405320051, 'f1_micro_mean': 0.3179280368464983, 'accuracy_balanced_mean': 0.2163801405620471, 'f1_macro_mean_std': 0.009229978878317351, 'f1_micro_mean_std': 0.011363037915812729, 'accuracy_balanced_mean_std': 0.008312693863799151}}

## anchor 'en', sentence-embeddings-en, one2anchor
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.33984091289212187, 'f1_micro_mean': 0.43704185004817836, 'accuracy_balanced_mean': 0.3455634379787608, 'f1_macro_mean_std': 0.012872255743693268, 'f1_micro_mean_std': 0.011860833391862844, 'accuracy_balanced_mean_std': 0.01098504906461055}}

## anchor 'en', sentence-embeddings-multi, trained on one2anchor (EN-anchor+anchor2test-lang), tested on test-lang 
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.34753720700622903, 'f1_micro_mean': 0.4462708519608439, 'accuracy_balanced_mean': 0.3514446040496088, 'f1_macro_mean_std': 0.01092396332476347, 'f1_micro_mean_std': 0.0140087481234386, 'accuracy_balanced_mean_std': 0.010391447194617548}}
# ! seems embed-multi seems actually better when no additional embeddings for translations. probably adds unnecessary noise from lower quality texts through nmt  # simple no-NMT-single seems better

## sentence-embeddings-multi, trained on one2many, tested on test-lang 
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2many', 
'f1_macro_mean': 0.334976068703692, 'f1_micro_mean': 0.43052156503444294, 'accuracy_balanced_mean': 0.3373919447972773, 'f1_macro_mean_std': 0.015057241526948756, 'f1_micro_mean_std': 0.010832965715384535, 'accuracy_balanced_mean_std': 0.011477658939804574}}


### standard_dl
## no-NMT-single, minilm-multi, 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-single', 
'f1_macro_mean': 0.32465740077048505, 'f1_micro_mean': 0.4350816756328619, 'accuracy_balanced_mean': 0.33964981863190635, 'f1_macro_mean_std': 0.017021589438980484, 'f1_micro_mean_std': 0.008622679567886487, 'accuracy_balanced_mean_std': 0.015569686061854549}}

## one2anchor, minilm-en, 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/MiniLM-L12-H384-uncased', 'vectorizer': 'transformer-mono', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.3526785087677711, 'f1_micro_mean': 0.4449542193991269, 'accuracy_balanced_mean': 0.3624410341419119, 'f1_macro_mean_std': 0.0182065168918998, 'f1_micro_mean_std': 0.02047364185883419, 'accuracy_balanced_mean_std': 0.01922773673919911}}

## one2anchor, minilm-multi, 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2anchor', 
'f1_macro_mean': 0.3310419058216454, 'f1_micro_mean': 0.4276589497148626, 'accuracy_balanced_mean': 0.3431239237844909, 'f1_macro_mean_std': 0.012778547620174932, 'f1_micro_mean_std': 0.011620658524093516, 'accuracy_balanced_mean_std': 0.00975351471176684}}

## one2many, minilm-multi, 15 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'one2many', 
'f1_macro_mean': 0.34504896603553187, 'f1_micro_mean': 0.45934509552937114, 'accuracy_balanced_mean': 0.3520232568809181, 'f1_macro_mean_std': 0.016290885919853306, 'f1_micro_mean_std': 0.011691613391657663, 'accuracy_balanced_mean_std': 0.01616228799569507}}



#### many lang experiments
### classical_ml

# no-nmt-many, tfidf
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many',
'f1_macro_mean': 0.19921951661655699, 'f1_micro_mean': 0.3009558921594391, 'accuracy_balanced_mean': 0.2019695591798122, 'f1_macro_mean_std': 0.009561234583870556, 'f1_micro_mean_std': 0.00615240006513217, 'accuracy_balanced_mean_std': 0.008381838229723854}}
# ! problematic because no custom stopwords & lang-specific feature engineering

# no-nmt-many, embeddings-en (embeddings-multi separately per lang as proxy)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many', 
'f1_macro_mean': 0.3685606391115852, 'f1_micro_mean': 0.47229330204043096, 'accuracy_balanced_mean': 0.36848250108324637, 'f1_macro_mean_std': 0.0132849060127712, 'f1_micro_mean_std': 0.01544597252506312, 'accuracy_balanced_mean_std': 0.014226180844586689}}

# no-nmt-many, embeddings-multi (not separately)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many', 
'f1_macro_mean': 0.40123910112737377, 'f1_micro_mean': 0.4962471131639723, 'accuracy_balanced_mean': 0.39881525293622316, 'f1_macro_mean_std': 0.012050144455204859, 'f1_micro_mean_std': 0.007794457274826777, 'accuracy_balanced_mean_std': 0.010892572269832757}}

# many2anchor, tfidf
# !!! performance with 500 samp is better than 2000 samp => there must be an issue with my code ... (or hps are really bad for larger sample)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.14096287931460702, 'f1_micro_mean': 0.29330254041570436, 'accuracy_balanced_mean': 0.16227912164844183, 'f1_macro_mean_std': 0.010291036560354988, 'f1_micro_mean_std': 0.001732101616628151, 'accuracy_balanced_mean_std': 0.005258498354857241}}
# ! not sure why this is worse than no-nmt-many. bug in code? or too much translation noise? should be better because more originally different texts. df_train_samp seems correct. maybe because test set more diverse and larger with NMT noise? Or unsuitable hp!
# with train shuffle
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'tfidf', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.14072795148138845, 'f1_micro_mean': 0.29272517321016167, 'accuracy_balanced_mean': 0.16202710551940958, 'f1_macro_mean_std': 0.010305313432689367, 'f1_micro_mean_std': 0.0017321016166281789, 'accuracy_balanced_mean_std': 0.005258498354857241}}

# many2anchor, embeddings-en
# !!! performance with 500 samp is better than 2000 samp => there must be an issue with my code ... (or hps are really bad for larger sample)
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.35697876826617514, 'f1_micro_mean': 0.4532332563510393, 'accuracy_balanced_mean': 0.3564410059193819, 'f1_macro_mean_std': 0.008993920383241288, 'f1_micro_mean_std': 0.00808314087759815, 'accuracy_balanced_mean_std': 0.008827801936200363}}
# ! not sure why this is worse than no-nmt-many. bug in code? or too much translation noise? should be better because more originally different texts. df_train_samp seems correct. maybe because test set more diverse and larger with NMT noise? Or unsuitable hp!
# with train shuffle
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-en', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.36208952684838036, 'f1_micro_mean': 0.4636258660508083, 'accuracy_balanced_mean': 0.3636840115696522, 'f1_macro_mean_std': 0.010150913399416511, 'f1_micro_mean_std': 0.006351039260969971, 'accuracy_balanced_mean_std': 0.012020711360166303}}

# many2anchor, embeddings-multi
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor',
'f1_macro_mean': 0.35997759792774364, 'f1_micro_mean': 0.4416859122401848, 'accuracy_balanced_mean': 0.35838894874988086, 'f1_macro_mean_std': 0.008749341694553575, 'f1_micro_mean_std': 0.0017321016166281789, 'accuracy_balanced_mean_std': 0.009085730632123923}}
# ?! translating to anchor and mixing seems to hurt performance quite a bit. Despite mixing original texts with trans-anchor. maybe embeddings cannot represent additional info in single vectors properly? Or unsuitable hps!

# many2many, embeddings-multi
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2many', 
'f1_macro_mean': 0.30696868425830426, 'f1_micro_mean': 0.3767321016166282, 'accuracy_balanced_mean': 0.3076885088264402, 'f1_macro_mean_std': 0.006938317325874105, 'f1_micro_mean_std': 0.0025981524249422683, 'accuracy_balanced_mean_std': 0.007123764810302247}}
# more mixing hurts even more (or less suitable hps)


### standard_dl
## no-NMT-many, minilm-multi, 10 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'no-nmt-many', 
'f1_macro_mean': 0.37265943750685937, 'f1_micro_mean': 0.49855658198614317, 'accuracy_balanced_mean': 0.38138113840610777, 'f1_macro_mean_std': 0.029512500372632644, 'f1_micro_mean_std': 0.025115473441108538, 'accuracy_balanced_mean_std': 0.019593473356427776}}

## many2anchor, minilm-en, 10 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/MiniLM-L12-H384-uncased', 'vectorizer': 'transformer-mono', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.12292358678515314, 'f1_micro_mean': 0.3443995381062356, 'accuracy_balanced_mean': 0.1865493321634295, 'f1_macro_mean_std': 0.06726829055535061, 'f1_micro_mean_std': 0.05802540415704388, 'accuracy_balanced_mean_std': 0.061549332163429504}}
# 30 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/MiniLM-L12-H384-uncased', 'vectorizer': 'transformer-mono', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.29671129778832206, 'f1_micro_mean': 0.398094688221709, 'accuracy_balanced_mean': 0.3096610023639281, 'f1_macro_mean_std': 0.00959969473724312, 'f1_micro_mean_std': 0.012990762124711314, 'accuracy_balanced_mean_std': 0.005635760287067082}}

## many2anchor, minilm-multi, 10 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.40696748699064167, 'f1_micro_mean': 0.5046189376443417, 'accuracy_balanced_mean': 0.4081168572038839, 'f1_macro_mean_std': 0.0013554627823026688, 'f1_micro_mean_std': 0.006351039260969971, 'accuracy_balanced_mean_std': 0.00019480069060628935}}

## many2many, minilm-multi, 4 epochs
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'standard_dl', 'model_name': 'microsoft/Multilingual-MiniLM-L12-H384', 'vectorizer': 'transformer-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2many', 
'f1_macro_mean': 0.4060399798702805, 'f1_micro_mean': 0.4942263279445728, 'accuracy_balanced_mean': 0.41114805502934704, 'f1_macro_mean_std': 0.01256610814450368, 'f1_micro_mean_std': 0.026558891454965372, 'accuracy_balanced_mean_std': 0.01592137067509322}}



"""


### deletable extract of prediction for accuracy-meta paper
#df_test_predictions = df_test_lang.copy(deep=True)
#df_test_predictions["prediction"] = label_pred
#df_test_predictions.to_csv("df_test_many2anchor_svm_500_embed_predictions.csv", index=False)


##### tests with fairlearn
"""
import sklearn.metrics as skm

df_test_predictions = df_test_lang.copy(deep=True)
df_test_predictions["prediction"] = label_pred

y_true = df_test_predictions.label
y_pred = df_test_predictions.prediction

## create new meta-data variables
# https://manifesto-project.wzb.eu/down/data/2020a/codebooks/codebook_MPDataset_MPDS2020a.pdf
mapping_parfam = {10: "ECO: Ecological parties", 20: "LEF: Socialist or other left parties",
                  30: "SOC: Social democratic parties", 40: "LIB: Liberal parties", 50: "CHR: Christian democratic parties (in Isreal also Jewish parties)",
                  60: "CON: Conservative parties", 70: "NAT: Nationalist parties", 80: "AGR: Agrarian parties",
                  90: "ETH: Ethnic and regional parties", 95: "SIP: Special issue parties", 98: "DIV: Electoral alliances of diverse origin without dominant party",
                  999: "MI: Missing information"
}
df_test_predictions["parfam_name"] = df_test_predictions.parfam.map(mapping_parfam)
df_test_predictions["parfam_rile"] = ["left" if parfam in [10, 20, 30] else "right" if parfam in [50, 60, 70, 80, 90] else "other" for parfam in df_test_predictions["parfam"]]
cmp_code_left = [103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701]
cmp_code_right = [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606]
cmp_code_other = np.unique([cmp_code for cmp_code in df_test_predictions["cmp_code"] if cmp_code not in cmp_code_left + cmp_code_right])
df_test_predictions["label_rile"] = ["left" if cmp_code in cmp_code_left else "right" if cmp_code in cmp_code_right else "other" for cmp_code in df_test_predictions["cmp_code"]]
df_test_predictions["decade"] = [str(date)[:3]+"0" for date in df_test_predictions.date]

df_test_predictions["country_iso"].value_counts()

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
                             # can look at intersection between features by passing df_test_predictions with multiple columns
                             sensitive_features=df_test_predictions[["label_rile"]],  # df_test_predictions[["language_iso", "parfam_name", "parfam_rile", "label_rile", "decade"]]
                             #control_features=df_test_predictions[["parfam_name"]]
)

print("## Metrics overall:\n", grouped_metric.overall, "\n")
print("## Metrics by group:\n", grouped_metric.by_group, "\n")  #.to_dict()
#print("## Metrics min:\n", grouped_metric.group_min(), "\n")
#print("## Metrics max:\n", grouped_metric.group_max(), "\n")
print("## Metrics difference min-max:\n", grouped_metric.difference(method='between_groups'), "\n")  # to_overall, between_groups  # difference or ratio of the metric values between the best and the worst slice
#print(grouped_metric.ratio(method='between_groups')) # to_overall, between_group  # difference or ratio of the metric values between the best and the worst slice
"""


"""
#### test with fairlearn transformation to remove correlation with rile

### many2anchor, embeddings-multi
## without preprocessing transformation or postprocessing
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.3792114424709246, 'f1_micro_mean': 0.46709006928406466, 'accuracy_balanced_mean': 0.3769977849366013, 'f1_macro_mean_std': 0.009789851688477513, 'f1_micro_mean_std': 0.008660508083140894, 'accuracy_balanced_mean_std': 0.01110978162005441}}
## Metrics overall:
 accuracy    0.475751
f1_macro    0.389001
count           1732
dtype: object 
## Metrics by group:
             accuracy  f1_macro count
label_rile                          
left        0.608796  0.370428   432
other        0.42158  0.367644   899
right       0.453865  0.276199   401 
## Metrics difference min-max:
 accuracy    0.187217
f1_macro    0.094228
count            498
dtype: object 


## with transformation alpha=1
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor',
'f1_macro_mean': 0.37990347718731216, 'f1_micro_mean': 0.46766743648960746, 'accuracy_balanced_mean': 0.3769393235252684, 'f1_macro_mean_std': 0.0040161485348022, 'f1_micro_mean_std': 0.0034642032332563577, 'accuracy_balanced_mean_std': 0.002439633348997078}}
## Metrics overall:
 accuracy    0.464203
f1_macro     0.38392
count           1732
dtype: object 
## Metrics by group:
             accuracy  f1_macro count
label_rile                          
left        0.543981  0.305981   432
other       0.420467  0.362778   899
right       0.476309  0.312158   401 
## Metrics difference min-max:
 accuracy    0.123514
f1_macro    0.056797
count            498
dtype: object 


## with transformation alpha=0.5
{'experiment_summary': {'dataset': 'manifesto-8', 'sample_size': [500], 'method': 'classical_ml', 'model_name': 'SVM', 'vectorizer': 'embeddings-multi', 'lang_anchor': 'en', 'lang_train': 'en', 'lang_all': ['en', 'de', 'es', 'fr', 'tr', 'ru'], 'augmentation': 'many2anchor', 
'f1_macro_mean': 0.353936169487248, 'f1_micro_mean': 0.43187066974595845, 'accuracy_balanced_mean': 0.35311475802284503, 'f1_macro_mean_std': 0.0007956664772877375, 'f1_micro_mean_std': 0.002309468822170896, 'accuracy_balanced_mean_std': 0.00041150944584439353}}
Run done.
## Metrics overall:
 accuracy     0.43418
f1_macro    0.353141
count           1732
dtype: object 
## Metrics by group:
             accuracy  f1_macro count
label_rile                          
left        0.527778  0.317143   432
other       0.382647  0.328759   899
right       0.448878  0.284718   401 
## Metrics difference min-max:
 accuracy     0.14513
f1_macro    0.044041
count            498
dtype: object 



"""

