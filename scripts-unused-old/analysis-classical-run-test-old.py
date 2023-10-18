

import sys
if sys.argv[0] != '/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/pydevconsole.py':
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.argv[0]: ", sys.argv[0], ")")


# ## Load packages
import pandas as pd
import numpy as np
import os
import tqdm
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, linear_model




## set global seed for reproducibility and against seed hacking
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

print(os.getcwd())
if (EXECUTION_TERMINAL==False) and ("meta-metrics-repo" not in os.getcwd()):
    os.chdir("./meta-metrics-repo")
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
#parser.add_argument('-lang', '--languages', type=str, nargs='+',
#                    help='List of languages to iterate over. e.g. "en", "de", "es", "fr", "ko", "tr", "ru" ')
#parser.add_argument('-anchor', '--language_anchor', type=str,
#                    help='Anchor language to translate all texts to if using anchor. Default is "en"')
#parser.add_argument('-language_train', '--language_train', type=str,
#                    help='What language should the training set be in?. Default is "en"')
#parser.add_argument('-augment', '--augmentation_nmt', type=str,
#                    help='Whether and how to augment the data with machine translation (MT).')

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
#parser.add_argument('-nmt', '--nmt_model', type=str,
#                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')



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
                            "--dataset", "benoit_lr",
                            "--sample_interval", "100",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "logistic",  # SVM, logistic
                            "--vectorizer", "tfidf",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--hyperparam_study_date", "20221111"])


### args only for test runs
CROSS_VALIDATION_REPETITIONS_FINAL = args.n_cross_val_final
ZEROSHOT = args.zeroshot

### args for both hyperparameter tuning and test runs
# choose dataset
DATASET = args.dataset

N_SAMPLE_DEV = args.sample_interval   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "classical_ml"
MODEL_NAME = args.model  # "SVM"

HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"





# ## Load data
if "manifesto-8" in DATASET:
  df_cl = pd.read_csv("./data-clean/df_manifesto_all.zip")
  df_train = pd.read_csv(f"./data-clean/df_{DATASET}_samp_train_trans_embed_tfidf.zip")
  df_test = pd.read_csv(f"./data-clean/df_{DATASET}_samp_test_trans_embed_tfidf.zip")
else:
  raise Exception(f"Dataset name not found: {DATASET}")

print(DATASET)




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
TRAINING_DIRECTORY = f"results/{DATASET}"


## data checks
print(DATASET, "\n")
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

from helpers import compute_metrics_classical_ml
## functions for scenario data selection and augmentation
#from helpers import select_data_for_scenario_final_test, data_augmentation, choose_preprocessed_text





# ## Final test with best hyperparameters

# select best hp based on hp-search
n_sample = N_SAMPLE_DEV[0]
n_sample_string = N_SAMPLE_DEV[0]
#n_sample_string = 300
while len(str(n_sample_string)) <= 4:
    n_sample_string = "0" + str(n_sample_string)

## hyperparameters for final tests
# selective load one decent set of hps for testing
# seems like optuna 3.1 cannot read study objects from optuna 2.x
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results-raw/manifesto-military/optuna_study_logistic_tfidf_01000samp_20221006.pkl")

# seems like optuna 3.1 cannot read study objects from optuna 2.x
#HYPER_PARAMS_LST = [value_hp_study["optuna_study"].best_trial.user_attrs["hyperparameters_all"] for key_hp_study, value_hp_study in hp_study_dic.items()]

hyperparams = {
    'penalty': 'l2',  # works with all solvers
    'solver': "liblinear",
    #'C': trial.suggest_float("C", 1, 1000, log=False),
    #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    #"intercept_scaling": trial.suggest_float("intercept_scaling", 1, 50, log=False),
    "class_weight": "balanced",
    #"max_iter": trial.suggest_int("max_iter", 50, 1000, log=False),  # 100 default
    "multi_class": "auto",  # {‘auto’, ‘ovr’, ‘multinomial’}
    #"warm_start": trial.suggest_categorical("warm_start", [True, False]),
    #"l1_ratio": None,
    #"n_jobs": -1,
    "random_state": SEED_GLOBAL,
}


### text pre-processing
# separate hyperparams for vectorizer and classifier.
hyperparams_vectorizer = {key: value for key, value in hyperparams.items() if key in ["ngram_range", "max_df", "min_df", "analyzer"]}
hyperparams_clf = {key: value for key, value in hyperparams.items() if key not in ["ngram_range", "max_df", "min_df", "analyzer"]}
# in case I want to add tfidf later
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"


# choose correct pre-processed text column here
import ast
if VECTORIZER == "tfidf":
    # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
    vectorizer_sklearn.fit(pd.concat([df_train_format.text_trans_concat_tfidf, df_test_format.text_trans_concat_tfidf]))
    X_train = vectorizer_sklearn.transform(df_train_format.text_trans_concat_tfidf)
    X_test = vectorizer_sklearn.transform(df_test_format.text_trans_concat_tfidf)
elif "en" == VECTORIZER:
    X_train = np.array([ast.literal_eval(lst) for lst in df_train_format.text_trans_concat_embed_en.astype('object')])
    X_test = np.array([ast.literal_eval(lst) for lst in df_test_format.text_trans_concat_embed_en.astype('object')])
elif "multi" == VECTORIZER:
    X_train = np.array([ast.literal_eval(lst) for lst in df_train_format.text_concat_embed_multi.astype('object')])
    X_test = np.array([ast.literal_eval(lst) for lst in df_test_format.text_concat_embed_multi.astype('object')])

y_train = df_train_format.label
y_test = df_test_format.label

## initialise and train classifier
from sklearn import svm, linear_model
if MODEL_NAME == "SVM":
    clf = svm.SVC(**hyperparams_clf)
elif MODEL_NAME == "logistic":
    clf = linear_model.LogisticRegression(**hyperparams_clf)
clf.fit(X_train, y_train)



### Evaluate
# test on test set
label_gold = y_test
label_pred = clf.predict(X_test)

### metrics
from helpers import compute_metrics_classical_ml
#results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
results_test = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df_cl.label_text.unique()))

print(results_test)

# do prediction on entire corpus? No.
# could do because annotations also contribute to estimate of distribution
#dataset["all"] = datasets.concatenate_datasets([dataset["test"], dataset["train"]])
#results_corpus = trainer.evaluate(eval_dataset=datasets.concatenate_datasets([dataset["train"], dataset["test"]]))  # eval_dataset=encoded_dataset["test"]
# with NLI, cannot run inference also on train set, because augmented train set can have different length than original train-set






### save final results
# save pickel or df? probably df to enable meta-data analyses
"""if EXECUTION_TERMINAL == True:
  joblib.dump(experiment_details_dic_summary, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}.pkl")
elif EXECUTION_TERMINAL == False:
  joblib.dump(experiment_details_dic_summary, f"./{TRAINING_DIRECTORY}/experiment_results_{MODEL_NAME.split('/')[-1]}_{AUGMENTATION}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{NMT_MODEL}_{HYPERPARAM_STUDY_DATE}_t.pkl")


df_cl_concat.to_csv(f"./results/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.zip",
                    compression={"method": "zip", "archive_name": f"df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv"}, index=False)
"""




print("\n\nRun done.")


