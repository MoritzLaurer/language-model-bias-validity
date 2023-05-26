

import sys
if sys.argv[0] != '/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/pydevconsole.py':
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.argv[0]: ", sys.argv[0], ")")


# ## Load packages
import optuna

import pandas as pd
import numpy as np
import os
import tqdm
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, linear_model
from sklearn.model_selection import train_test_split

# adjust directory to machine the code is running on
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
parser = argparse.ArgumentParser(description='Run hyperparameter tuning with different languages, algorithms, datasets')

## Add the arguments
# arguments for hyperparameter search
parser.add_argument('-t', '--n_trials', type=int,
                    help='How many optuna trials should be run?')
parser.add_argument('-ts', '--n_trials_sampling', type=int,
                    help='After how many trials should optuna start sampling?')
parser.add_argument('-tp', '--n_trials_pruning', type=int,
                    help='After how many trials should optuna start pruning?')
parser.add_argument('-cvh', '--n_cross_val_hyperparam', type=int, default=2,
                    help='How many times should optuna cross validate in a single trial?')
parser.add_argument('-hp_date', '--hyperparam_study_date', type=str,
                    help='Date string to specifiy which hyperparameter run should be selected. e.g. "20220304"')
# general hyperparameters
parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "uk-leftright", "pimpo" ')
parser.add_argument('-samp', '--sample_size', type=int, #nargs='+',
                    help='Sample size')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "classical_ml"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings"')





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
  args = parser.parse_args(["--n_trials", "40", "--n_trials_sampling", "25", "--n_trials_pruning", "30", "--n_cross_val_hyperparam", "3",  #"--context",
                            "--dataset", "pimpo",  #  uk-leftright-econ, pimpo
                            "--sample_size", "500",  #"100", "500", "1000", #"2500", "5000", #"10000",
                            "--method", "classical_ml", "--model", "logistic",  # SVM, logistic
                            "--vectorizer", "tfidf",  # "tfidf", "embeddings-en", "embeddings-multi"
                            "--hyperparam_study_date", "20230207"])


### args only for hyperparameter tuning
N_TRIALS = args.n_trials
N_STARTUP_TRIALS_SAMPLING = args.n_trials_sampling
N_STARTUP_TRIALS_PRUNING = args.n_trials_pruning
CROSS_VALIDATION_REPETITIONS_HYPERPARAM = args.n_cross_val_hyperparam
#CONTEXT = False  #args.context   # ! do not use context, because some languages have too little data and makes train-test split problematic

### args for both hyperparameter tuning and test runs
DATASET = args.dataset  # "sentiment-news-econ" "coronanet" "cap-us-court" "cap-sotu" "manifesto-8" "manifesto-military" "manifesto-protectionism" "manifesto-morality" "manifesto-nationalway" "manifesto-44" "manifesto-complex"
TRAINING_DIRECTORY = f"results/{DATASET}"

N_SAMPLE_DEV = args.sample_size   # [100, 500, 1000, 2500, 5000, 10_000]  999_999 = full dataset  # cannot include 0 here to find best hypothesis template for zero-shot, because 0-shot assumes no dev set
VECTORIZER = args.vectorizer

# decide on model to run
METHOD = args.method  # "standard_dl", "nli", "classical_ml"
MODEL_NAME = args.model  # "logistic", "SVM"

HYPERPARAM_STUDY_DATE = args.hyperparam_study_date  #"20220304"

# set global seed for reproducibility and against seed hacking
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# special intermediate sample for PimPo
SAMPLE_NO_TOPIC = 50_000  #100_000
TRAIN_NOTOPIC_PROPORTION = 0.4

### Load data
if "uk-leftright" in DATASET:
    df = pd.read_csv(f"./data-clean/benoit_leftright_sentences.zip", engine='python')
    df_cl = df.copy(deep=True)
    #df_train = pd.read_csv(f"./data-clean/df_{DATASET}_samp_train_trans_{NMT_MODEL}_embed_tfidf.zip")
    #df_test = pd.read_csv(f"./data-clean/df_{DATASET}_samp_test_trans_{NMT_MODEL}_embed_tfidf.zip")
elif "pimpo" in DATASET:
    #df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/data-clean/df_pimpo_samp_trans_m2m_100_1.2B_embed_tfidf.zip", engine='python')
    df = pd.read_csv("./data-clean/df_pimpo_samp_trans_lemmatized_stopwords.zip", engine="python")
    df_cl = df.copy(deep=True)
else:
    raise Exception(f"Dataset name not found: {DATASET}")

print(DATASET)


### preprocessing data

## uk-leftright
# select relevant subset for training
if "uk-leftright-econ" in DATASET:
    # select to work with crowd annotations and expert annotations
    df_cl = df_cl[df_cl.source == "Crowd"]
    # select task on either economy or social policy
    df_cl = df_cl[df_cl.scale == "Economic"]
    # transform continuous float data to categorical classes
    df_cl["label_scale"] = df_cl.score_sent_mean.fillna(0).round().astype(int)
    print(df_cl["label_scale"].value_counts())
elif "uk-leftright-soc" in DATASET:
    raise NotImplementedError

# translate scale to classes
if "uk-leftright" in DATASET:
    ## simplify scale to three classes for label text
    task_label_text_map = {0: "neutral", 1: "right", 2: "right", -1: "left", -2: "left"}
    # could also test scale as 5 classes
    #task_label_text_map = {0: "neutral", 1: "right", 2: "very_right", -1: "left", -2: "very_left"}
    df_cl["label_text"] = df_cl.label_scale.map(task_label_text_map)
    print(df_cl["label_text"].value_counts())
    ## adapt numeric label
    task_label_text_map_factorized = {"neutral": 1, "right": 2, "right": 2, "left": 0, "left": 0}
    #task_label_text_map_factorized = {"neutral": 2, "right": 3, "very_right": 4, "left": 1, "very_left": 0}
    df_cl["label"] = df_cl["label_text"].map(task_label_text_map_factorized)
    print(df_cl["label"].value_counts())

## pimpo task-specific pre-processing
# testing all tasks at once for now
"""if TASK == "integration":
    task_label_text = ["integration_supportive", "integration_sceptical", "integration_neutral", "no_topic"]
elif TASK == "immigration":
    task_label_text = ["immigration_supportive", "immigration_sceptical", "immigration_neutral", "no_topic"]
    #df_cl = df[df.label_text.isin(immigration_label_text)]
    # replace labels for other task with "no_topic"
    df_cl["label_text"] = [label if label in task_label_text else "no_topic" for label in df.label_text]"""

# remove x% no_topic for faster testing
if "pimpo" in DATASET:
    df_cl = df_cl.groupby(by="label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] == "no_topic" else x)


print(df_cl["label"].value_counts())
print(df_cl["label_text"].value_counts())



## prepare input text data
if VECTORIZER == "tfidf":
    if "uk-leftright" in DATASET:
        df_cl["text_prepared"] = df_cl["text_preceding"].fillna('') + " " + df_cl["text_original"] + " " + df_cl["text_following"].fillna('')
    elif "pimpo" in DATASET:
        #df_cl["text_prepared"] = df_cl["text_preceding_trans"].fillna('') + " " + df_cl["text_original_trans"] + " " + df_cl["text_following_trans"].fillna('')
        next
else:
    raise Exception(f"Vectorizer {VECTORIZER} not implemented.")

## data checks
print("Dataset: ", DATASET, "\n")
# verify that numeric label is in alphabetical order of label_text (can avoid issues for NLI)
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)



## lemmatize prepared text
# TODO: put in different script upstream (with embeddings probably)
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.batch_size

def lemmatize_and_stopwordremoval(text_lst):
    texts_lemma = []
    for doc in nlp.pipe(text_lst, n_process=1, disable=["ner"]):  # disable=["tok2vec", "ner"] "tagger", "attribute_ruler", "parser",
        doc_lemmas = [token.lemma_ for token in doc if not token.is_stop]
        # if else in case all tokens are deleted due to stop word removal
        if not any(pd.isna(doc_lemmas)):
            doc_lemmas = " ".join(doc_lemmas)
            texts_lemma.append(doc_lemmas)
        else:
            print(doc)
            texts_lemma.append(doc.text)
    return texts_lemma

if "uk-rightleft" in DATASET:
    df_cl["text_prepared"] = lemmatize_and_stopwordremoval(df_cl.text_prepared)
    print("Spacy lemmatization done")
elif "pimpo" in DATASET:
    # re-using translated, concatenated, lemmatized, stopword-cleaned column from multilingual paper
    df_cl["text_prepared"] = df_cl["text_trans_concat_tfidf"]


## train test split?
# not doing a train-test split for hp-search. Reason:
# want to do tests on multiple different train-test splits to control for randomness in sampling. this enables easier robustness tests. (data-test is always everything except data-train)
# searching for one set of general hps should not have negative data leakage effects, since I will use these hps for multiple train-test splits
# and is more practical to have one general set of hps instead of N hps for N train-test splits
# and expecting classical methods to underperform anyways and even if there is indirect data leakage, good performing classical algos would go against my planned argument ('steel-manning')

df_train = df_cl.copy(deep=True)

df_train.label_text.value_counts()





## Load helper functions
import sys
sys.path.insert(0, os.getcwd())

import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)

from helpers import compute_metrics_classical_ml, clean_memory




##### hyperparameter tuning


def optuna_objective(trial, n_sample=None, df_train=None, df=None):  # hypothesis_hyperparams_dic=None
  clean_memory()
  np.random.seed(SEED_GLOBAL)  # setting seed again for safety. not sure why this needs to be run here at each iteration. it should stay constant once set globally?! explanation could be this https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f

  # for testing
  global df_train_samp_split
  global df_dev

  if VECTORIZER == "tfidf":
      # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
      hyperparams_vectorizer = {
          'ngram_range': trial.suggest_categorical("ngram_range", [(1, 2), (1, 3), (1, 6)]),
          'max_df': trial.suggest_categorical("max_df", [1.0, 0.9, 0.8]),  # new runs for 'no-nmt-many many2many' + tfidf - others [0.95, 0.9, 0.8]  # can lead to error "ValueError: After pruning, no terms remain. Try a lower min_df or a higher max_df."
          'min_df': trial.suggest_categorical("min_df", [1, 0.01, 0.03]),  # new runs for 'no-nmt-many many2many' + tfidf - others [0.01, 0.03, 0.05]
          'analyzer': trial.suggest_categorical("analyzer", ["word", "char_wb"]),  # could be good for languages like Korean where longer sequences of characters without a space seem to represent compound words
      }
      vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"
  elif "embeddings" in VECTORIZER:
      vectorizer_sklearn = ["somerandomstringtopreventerrorsfromoptuna"]
      hyperparams_vectorizer = {}

  # SVM  # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  if MODEL_NAME == "SVM":
      hyperparams_clf = {'kernel': trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                   'C': trial.suggest_float("C", 1, 1000, log=True),
                   "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                   "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                   "coef0": trial.suggest_float("coef0", 1, 100, log=True),  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
                   "degree": trial.suggest_int("degree", 1, 50, log=False),  # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
                   #"decision_function_shape": trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"]),  # "However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification."
                   #"tol": trial.suggest_categorical("tol", [1e-3, 1e-4]),
                   "random_state": SEED_GLOBAL,
                    # 10k max_iter had 1-10% performance drop for high n sample (possibly overfitting to majority class)
                    #MAX_ITER_LOW, MAX_ITER_HIGH = 1_000, 7_000  # tried 10k, but led to worse performance on larger, imbalanced dataset (?)
                    "max_iter": trial.suggest_int("num_train_epochs", 1_000, 7_000, log=False, step=1000),  #MAX_ITER,
                   }
  # Logistic Regression # ! disadvantage: several parameters only work with certain other parameters  # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn-linear-model-logisticregression
  elif MODEL_NAME == "logistic":
      hyperparams_clf = {#'penalty': trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"]),
                        'penalty': 'l2',  # works with all solvers
                        'solver': trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
                        'C': trial.suggest_float("C", 1, 1000, log=False),
                        #"fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
                        #"intercept_scaling": trial.suggest_float("intercept_scaling", 1, 50, log=False),
                        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                        "max_iter": trial.suggest_int("max_iter", 50, 1000, log=False),  # 100 default
                        "multi_class": "auto",  # {‘auto’, ‘ovr’, ‘multinomial’}
                        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
                        #"l1_ratio": None,
                        "n_jobs": -1,
                        "random_state": SEED_GLOBAL,
                        }
  else:
      raise Exception("Model not available: ", MODEL_NAME)


  hyperparams_optuna = {**hyperparams_clf, **hyperparams_vectorizer}
  trial.set_user_attr("hyperparameters_vectorizer", hyperparams_vectorizer)
  trial.set_user_attr("hyperparameters_classifier", hyperparams_clf)
  print("Hyperparameters for this run: ", hyperparams_optuna)


  ## cross-validation loop.
  # Objective: determine F1 for specific sample for specific hyperparams, without a test set
  run_info_dic_lst = []
  for step_i, random_seed_cross_val in enumerate(np.random.choice(range(1000), size=CROSS_VALIDATION_REPETITIONS_HYPERPARAM)):

    # TODO: final reflection: does sampling for hp-search like this make sense?

    ## sample overall maximum training size
    if "uk-leftright" in DATASET:
        df_train_samp = df_train.sample(n=N_SAMPLE_DEV, random_state=random_seed_cross_val).copy(deep=True)
    elif "pimpo" in DATASET:
        # sample x% of training data for no topic, then share the remainder equally across classes
        n_sample_notopic = int(N_SAMPLE_DEV * TRAIN_NOTOPIC_PROPORTION)
        n_sample_perclass = int((N_SAMPLE_DEV - n_sample_notopic) / (len(df_cl.label_text.unique())-1))
        df_train_samp1 = df_train.groupby("label_text", as_index=False, group_keys=False).apply(
            lambda x: x.sample(min(n_sample_perclass, len(x)), random_state=random_seed_cross_val) if x.label_text.unique()[0] != "no_topic" else None)
        df_train_samp2 = df_train[df_train.label_text == "no_topic"].sample(n_sample_notopic, random_state=random_seed_cross_val)
        df_train_samp = pd.concat([df_train_samp1, df_train_samp2])

    ## train-validation split
    # ~50% split cross-val as recommended by https://arxiv.org/pdf/2109.12742.pdf
    test_size = 0.4
    # test unique splitting with unique sentence_id for manifesto or unique id "rn" column for pimpo, or sentenceid for uk-leftright
    if "uk-leftright" in DATASET:
        df_train_samp_split_ids, df_dev_ids = train_test_split(df_train_samp.sentenceid.unique(), test_size=test_size, shuffle=True, random_state=random_seed_cross_val)
        df_train_samp_split = df_train_samp[df_train_samp.sentenceid.isin(df_train_samp_split_ids)]
        df_dev = df_train_samp[df_train_samp.sentenceid.isin(df_dev_ids)]
    elif "pimpo" in DATASET:
        df_train_samp_split_ids, df_dev_ids = train_test_split(df_train_samp.rn.unique(), test_size=test_size, shuffle=True, random_state=random_seed_cross_val)
        df_train_samp_split = df_train_samp[df_train_samp.rn.isin(df_train_samp_split_ids)]
        df_dev = df_train_samp[df_train_samp.rn.isin(df_dev_ids)]
    print(f"Final train test length after cross-val split: len(df_train_samp_lang_samp) = {len(df_train_samp_split)}, len(df_dev_lang_samp) {len(df_dev)}.")

    print("Number of training examples after sampling: ", len(df_train_samp_split))
    print("Label distribution for df_train_samp_split:\n", df_train_samp_split.label_text.value_counts())
    print("Number of validation examples after sampling: ", len(df_dev))
    print("Label distribution for df_dev:\n", df_dev.label_text.value_counts())
    print("\n")

    clean_memory()

    ## ! choose correct pre-processed text column here
    # possible vectorizers: "tfidf", "embeddings-en", "embeddings-multi"
    # TODO: need to implement spacy lemmatization + probably embeddings later
    if VECTORIZER == "tfidf":
        vectorizer_sklearn.fit(pd.concat([df_train_samp_split.text_prepared, df_dev.text_prepared]))
        X_train = vectorizer_sklearn.transform(df_train_samp_split.text_prepared)
        X_test = vectorizer_sklearn.transform(df_dev.text_prepared)
    else:
        raise NotImplementedError

    y_train = df_train_samp_split.label
    y_test = df_dev.label

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
    metric_step = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df.label_text.unique()))

    run_info_dic = {"method": METHOD, "vectorizer": VECTORIZER, "n_sample": n_sample, "model": MODEL_NAME, "results": metric_step, "hyper_params": hyperparams_optuna}
    run_info_dic_lst.append(run_info_dic)
    
    # Report intermediate objective value.
    intermediate_value = (metric_step["eval_f1_macro"] + metric_step["eval_f1_micro"]) / 2
    trial.report(intermediate_value, step_i)
    # Handle trial pruning based on the intermediate value.
    if trial.should_prune() and (CROSS_VALIDATION_REPETITIONS_HYPERPARAM > 1):
      raise optuna.TrialPruned()
    if n_sample == 999_999:  # no cross-validation necessary for full dataset
      break


  ## aggregation over cross-val loop
  f1_macro_crossval_lst = [dic["results"]["eval_f1_macro"] for dic in run_info_dic_lst]
  f1_micro_crossval_lst = [dic["results"]["eval_f1_micro"] for dic in run_info_dic_lst]
  accuracy_balanced_crossval_lst = [dic["results"]["eval_accuracy_balanced"] for dic in run_info_dic_lst]
  metric_details = {
      "F1_macro_mean": np.mean(f1_macro_crossval_lst), "F1_micro_mean": np.mean(f1_micro_crossval_lst), "accuracy_balanced_mean": np.mean(accuracy_balanced_crossval_lst),
      "F1_macro_std": np.std(f1_macro_crossval_lst), "F1_micro_std": np.std(f1_micro_crossval_lst), "accuracy_balanced_std": np.std(accuracy_balanced_crossval_lst)
  }
  trial.set_user_attr("metric_details", metric_details)

  results_lst = [dic["results"] for dic in run_info_dic_lst]
  trial.set_user_attr("results_trainer", results_lst)

  # objective: maximise mean of f1-macro & f1-micro. HP should be good for imbalanced data, but also important/big classes
  metric = (np.mean(f1_macro_crossval_lst) + np.mean(f1_micro_crossval_lst)) / 2
  std = (np.std(f1_macro_crossval_lst) + np.std(f1_micro_crossval_lst)) / 2

  print(f"\nFinal metrics for run: {metric_details}. With hyperparameters: {hyperparams_optuna}\n")

  return metric



### run study
def run_study(n_sample=None):
  np.random.seed(SEED_GLOBAL)

  optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS_PRUNING, n_warmup_steps=0, interval_steps=1, n_min_trials=1)  # https://optuna.readthedocs.io/en/stable/reference/pruners.html
  optuna_sampler = optuna.samplers.TPESampler(seed=SEED_GLOBAL, consider_prior=True, prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False,
                                              n_startup_trials=N_STARTUP_TRIALS_SAMPLING, n_ei_candidates=24, multivariate=False, group=False, warn_independent_sampling=True, constant_liar=False)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler
  study = optuna.create_study(direction="maximize", study_name=None, pruner=optuna_pruner, sampler=optuna_sampler)  # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html

  study.optimize(lambda trial: optuna_objective(trial, n_sample=n_sample, df_train=df_train.copy(deep=True), df=df_cl.copy(deep=True)),  # hypothesis_hyperparams_dic=hypothesis_hyperparams_dic
                n_trials=N_TRIALS, show_progress_bar=True)  # Objective function with additional arguments https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
  return study


study = run_study(n_sample=N_SAMPLE_DEV)

print(f"\nBest hyperparameters: {study.best_params} \n")
print(f"\nBest values (F1-micro+macro mean): {study.best_value} \n")


hp_study_dic = {"vectorizer": VECTORIZER, "method": METHOD, "n_sample": N_SAMPLE_DEV, "dataset": DATASET, "algorithm": MODEL_NAME, "optuna_study": study}




### save studies
# harmonise length of n_sample string (always 5 characters)
n_sample_str = N_SAMPLE_DEV
while len(str(n_sample_str)) <= 3:
    n_sample_str = "0" + str(n_sample_str)

if EXECUTION_TERMINAL == True:
    joblib.dump(hp_study_dic, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample_str}samp_{DATASET}_{HYPERPARAM_STUDY_DATE}.pkl")
elif EXECUTION_TERMINAL == False:
    joblib.dump(hp_study_dic, f"./{TRAINING_DIRECTORY}/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample_str}samp_{DATASET}_{HYPERPARAM_STUDY_DATE}_t.pkl")



print("Run done.")





