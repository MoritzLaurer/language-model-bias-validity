


import sys
if sys.argv[0] != '/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/pydevconsole.py':
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.argv[0]: ", sys.argv[0], ")")


## load relevant packages
import pandas as pd
import numpy as np
import os
import torch
#from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
#from transformers import TrainingArguments
import time

# Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import (
    compute_metrics_standard, clean_memory, compute_metrics_nli_binary, compute_metrics_generation,
    load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset
)


# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

# main args
parser.add_argument('-ds', '--dataset', type=str,
                    help='Name of dataset. Can be one of: "uk-leftright", "pimpo" ')
parser.add_argument('-samp_train', '--sample_size_train', type=int, #nargs='+',
                    help='Sample size')
parser.add_argument('-samp_corpus', '--sample_size_corpus', type=int, #nargs='+',
                    help='Sample size for corpus to to al on')
parser.add_argument('-samp_no_topic', '--sample_size_no_topic', type=int, #nargs='+',
                    help='Sample size for no-topic class')
parser.add_argument('-samp_test', '--sample_size_test', type=int, #nargs='+',
                    help='Sample size for full test set')
parser.add_argument('-m', '--method', type=str,
                    help='Method. One of "classical_ml"')
parser.add_argument('-model', '--model', type=str,
                    help='Model name. String must lead to any Hugging Face model or "SVM" or "logistic". Must fit to "method" argument.')
parser.add_argument('-vectorizer', '--vectorizer', type=str,
                    help='How to vectorize text. Options: "tfidf" or "embeddings"')
parser.add_argument('-date', '--study_date', type=str,
                    help='Date')

#parser.add_argument('-size', '--model_size', type=str,
#                    help='base or large')
#parser.add_argument('-hypo', '--hypothesis', type=str,
#                    help='which hypothesis?')
parser.add_argument('-t', '--task', type=str,
                    help='task about integration or immigration?')
parser.add_argument('-n_run', '--n_run', type=int, #nargs='+',
                    help='The number of the respective random iteration')
parser.add_argument('-n_rand_runs', '--n_random_runs_total', type=int, #nargs='+',
                    help='The total number of random iterations')
parser.add_argument('-g_samp', '--group_sample', type=str,
                    help='group to filter training data by')
#parser.add_argument('-n_tok_rm', '--n_tokens_remove', type=int, #nargs='+',
#                    help='number of group-specific tokens to remove from test data')
parser.add_argument('-save', '--save_outputs', action="store_true",
                    help='boolean whether to save outputs to disk')
parser.add_argument('-g_col', '--group_column', type=str,
                    help='group column to filter training df by')

parser.add_argument('-max_l', '--max_length', type=int, #nargs='+',
                    help='max n tokens')


## choose arguments depending on execution in terminal or in script for testing
if EXECUTION_TERMINAL == True:
    print("Arguments passed via the terminal:")
    args = parser.parse_args()
    # To show the results of the given option to screen.
    print("")
    for key, value in parser.parse_args()._get_kwargs():
        print(value, "  ", key)
else:
    # parse args if not in terminal, but in script
    args = parser.parse_args([
        "--task", "cap-merge",  # cap-merge pimpo-simple, uk-leftright-simple, uk-leftright
        "--dataset", "cap-merge",  # cap-merge uk-leftright-econ, pimpo
        "--vectorizer", "tfidf",
        "--model", "log_reg",  #"google/flan-t5-small",  #"google/electra-small-discriminator",  #"MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",  #"google/flan-t5-small",
        "--method", "classical_ml",  #"nli_short",  #"generation",
        "--sample_size_train", "50", "--sample_size_no_topic", "5000",
        "--study_date", "20230601",
        "--n_run", "1", "--n_random_runs_total", "3",
        "--group_sample", "randomall", "--max_length", "256",  #"--n_tokens_remove", "0",
        "--sample_size_test", "500",  #"--sample_size_corpus", "500",
        "--group_column", "domain",  # "domain", "country_iso", "parfam_text", "parfam_text_aggreg", "decade"
        #"--save_outputs"
    ])

#LANGUAGE_LST = args.languages.split("-")
DATASET = args.dataset
TRAINING_DIRECTORY = f"results/{DATASET}"

SAMPLE_SIZE_TRAIN = args.sample_size_train
DATE = args.study_date
TASK = args.task
METHOD = args.method
MODEL_NAME = args.model
VECTORIZER = args.vectorizer
#N_TOKENS_REMOVE = args.n_tokens_remove
SAVE_OUTPUTS = args.save_outputs
GROUP_SAMPLE = args.group_sample
GROUP_COL = args.group_column

MODEL_MAX_LENGTH = args.max_length


# set global seed for reproducibility and against seed hacking
N_RUN = args.n_run - 1
N_RANDOM_RUNS_TOTAL = args.n_random_runs_total
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# special variables for pimpo
SAMPLE_NO_TOPIC = args.sample_size_no_topic  # for number in test set
if "pimpo" in DATASET:
    TRAIN_NOTOPIC_PROPORTION_TRAIN = 0.4
else:
    TRAIN_NOTOPIC_PROPORTION_TRAIN = 0
#SAMPLE_SIZE_CORPUS = args.sample_size_corpus
SAMPLE_SIZE_TEST = args.sample_size_test

# randomly assign different seeds for each run
seed_runs_all = np.random.choice(range(1000), size=N_RANDOM_RUNS_TOTAL)
SEED_RUN = int(seed_runs_all[N_RUN])
print("Iteration number: ", N_RUN)
print("All random seeds: ", seed_runs_all)
print("Random seed for this run: ", SEED_RUN)

# not sure if I should keep TASK variable
assert DATASET.split("-")[0] in TASK, f"Mismatch between dataset {DATASET} and task {TASK}"

# shorten model_name, can use in file names later
MODEL_NAME_SHORT = MODEL_NAME.split("/")[-1]
MODEL_NAME_SHORT = MODEL_NAME_SHORT[:26]  # longer names lead to file name length bugs before
print(MODEL_NAME_SHORT)

if ("nli" in MODEL_NAME.lower()) and ("generation" in METHOD.lower()):
    raise Exception(f"MODEL_NAME {MODEL_NAME} not implemented for METHOD {METHOD}")
elif ("t5" in MODEL_NAME.lower()) and ("nli" in METHOD.lower() or "standard_dl" in METHOD.lower() or "disc" in METHOD.lower()):
    raise Exception(f"MODEL_NAME {MODEL_NAME} not implemented for METHOD {METHOD}")
elif ("electra" in MODEL_NAME.lower()) and ("generation" in METHOD.lower()):
    raise Exception(f"MODEL_NAME {MODEL_NAME} not implemented for METHOD {METHOD}")

if "small" in MODEL_NAME.lower():
    MODEL_SIZE = "small"
elif "base" in MODEL_NAME.lower():
    MODEL_SIZE = "base"
elif "large" in MODEL_NAME.lower():
    MODEL_SIZE = "large"
elif "xl" in MODEL_NAME.lower():
    MODEL_SIZE = "xl"
else:
    MODEL_SIZE = "unspecified"
    #raise NotImplementedError(f"Model size for {MODEL_NAME} not implemented.")





##### load dataset
if "pimpo" in DATASET:
    df = pd.read_csv("./data-clean/df_pimpo_samp_train.zip", engine="python")
    df_test = pd.read_csv("./data-clean/df_pimpo_samp_test.zip", engine="python")
    df_cl = df.copy(deep=True)
elif "coronanet" in DATASET:
    df = pd.read_csv("./data-clean/df_coronanet_train.zip", engine="python")
    df_test = pd.read_csv("./data-clean/df_coronanet_test.zip", engine="python")
    df_cl = df.copy(deep=True)
elif "cap-merge" in DATASET:
    df = pd.read_csv("./data-clean/df_cap_merge_train.zip", engine="python")
    df_test = pd.read_csv("./data-clean/df_cap_merge_test.zip", engine="python")
    df_cl = df.copy(deep=True)
    if GROUP_COL != "domain":
        raise Exception(f"Group column {GROUP_COL} for dataset {DATASET} should always be 'domain'.")
elif "cap-sotu" in DATASET:
    df = pd.read_csv("./data-clean/df_cap_sotu_train.zip", engine="python")
    df_test = pd.read_csv("./data-clean/df_cap_sotu_test.zip", engine="python")
    df_cl = df.copy(deep=True)
else:
    raise Exception(f"Dataset name not found: {DATASET}")


### preprocessing data

## prepare input text data
#if VECTORIZER == "tfidf":
#    if METHOD == "classical_ml":
#        df_cl["text_prepared"] = df_cl["text_preceding"].fillna('') + " " + df_cl["text_original"] + " " + df_cl["text_following"].fillna('')
if "pimpo" in DATASET:
    if any(substring in METHOD for substring in ["nli", "disc"]) or (METHOD == "generation"):
        df_cl["text_prepared"] = 'The quote: "' + df_cl["text_original_trans"] + '"'
        df_test["text_prepared"] = 'The quote: "' + df_test["text_original_trans"] + '"'
    elif "standard" in METHOD:
        df_cl["text_prepared"] = df_cl["text_original_trans"]
        df_test["text_prepared"] = df_test["text_original_trans"]
    elif "classical_ml" in METHOD:
        df_cl["text_prepared"] = df_cl["text_original_trans"]
        df_test["text_prepared"] = df_test["text_original_trans"]
    else:
        raise NotImplementedError
elif "coronanet" in DATASET:
    df_cl["text_prepared"] = df_cl["text"]
    df_test["text_prepared"] = df_test["text"]
elif "cap-merge" in DATASET:
    # cap sotu and court datasets have different text columns. harmonising them here
    df_cl["text_prepared"] = df_cl["text"]
    df_test["text_prepared"] = df_test["text"]
    df_cap_sotu = df_cl[df_cl.domain == "speech"]
    df_cap_sotu_test = df_test[df_test.domain == "speech"]
    if any(substring in METHOD for substring in ["nli", "disc"]) or (METHOD == "generation"):
        df_cl.loc[df_cl.domain == "speech", "text_prepared"] = 'The quote: "' + df_cap_sotu["text_original"] + '"'
        df_test.loc[df_test.domain == "speech", "text_prepared"] = 'The quote: "' + df_cap_sotu_test["text_original"] + '"'
    elif "standard" in METHOD:
        df_cl.loc[df_cl.domain == "speech", "text_prepared"] = df_cap_sotu["text_original"]
        df_test.loc[df_test.domain == "speech", "text_prepared"] = df_cap_sotu_test["text_original"]
    elif "classical_ml" in METHOD:
        df_cl.loc[df_cl.domain == "speech", "text_prepared"] = df_cap_sotu["text_original"]
        df_test.loc[df_test.domain == "speech", "text_prepared"] = df_cap_sotu_test["text_original"]
    else:
        raise NotImplementedError
elif "cap-sotu" in DATASET:
    if any(substring in METHOD for substring in ["nli", "disc"]) or (METHOD == "generation"):
        df_cl.loc[:, "text_prepared"] = 'The quote: "' + df_cl["text_original"] + '"'
        df_test.loc[:, "text_prepared"] = 'The quote: "' + df_test["text_original"] + '"'
    elif "standard" in METHOD:
        df_cl.loc[:, "text_prepared"] = df_cl["text_original"]
        df_test.loc[:, "text_prepared"] = df_test["text_original"]
    elif "classical_ml" in METHOD:
        df_cl.loc[:, "text_prepared"] = df_cl["text_original"]
        df_test.loc[:, "text_prepared"] = df_test["text_original"]
    else:
        raise NotImplementedError
else:
    raise Exception(f"Vectorizer {VECTORIZER} or METHOD {METHOD} not implemented.")


## data checks
#print("Dataset: ", DATASET, "\n")
# verify that numeric labels is in alphabetical order of label_text (can avoid issues for NLI)
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").labels.tolist()  # labels num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)




### select training data
import random
random.seed(SEED_RUN)

def select_group_members_randomly(df=None, group_col=None, n_members_str=None, seed=None):
    random.seed(int(seed))
    n_members = int(n_members_str[-1])
    group_join = random.sample(list(df[group_col].unique().astype(str)), n_members)
    print(f"Group selected: {group_join}  for seed {seed}")
    group_join = r'\b' + r'\b|\b'.join(group_join) + r'\b'
    #group_join = f"^({group_join})$"  # to only look at exact matches
    return group_join, seed+1

#group_join = random.sample(list(df[GROUP_COL].unique().astype(str)), int(GROUP_SAMPLE[-1]))
#r'\b' + r'\b|\b'.join(group_join) + r'\b'

label_distribution_per_group_member = df_cl.groupby(GROUP_COL).apply(lambda x: x.label_text.value_counts())
print("Overall label distribution per group member:\n", label_distribution_per_group_member)

# sample training data
# unrealistic balanced sample without AL
# redo sampling for different groups until get a fully balanced sample (or 20 iter)
# balanced samples are important to remove data imbalanced as intervening variable for performance differences
imbalanced_sample = True
counter = 0
seed_run_update = SEED_RUN
while imbalanced_sample and (counter <= 10):
    # select data based on group. redo sampling with different random seed if necessary
    if "randomall" in GROUP_SAMPLE:
        df_cl_group = df_cl.copy(deep=True)
        group_join = "randomall"
        print("GROUP_SAMPLE is randomall, so just sampling from entire corpus without group selection")
    elif "random" in GROUP_SAMPLE:
        group_join, seed_run_update = select_group_members_randomly(df=df_cl, group_col=GROUP_COL, n_members_str=GROUP_SAMPLE, seed=seed_run_update)
        df_cl_group = df_cl[df_cl[GROUP_COL].astype(str).str.contains(group_join)].copy(deep=True)
    else:
        raise NotImplementedError

    # sample x% of training data for no topic, then share the remainder equally across classes
    n_sample_notopic = int(SAMPLE_SIZE_TRAIN * TRAIN_NOTOPIC_PROPORTION_TRAIN)
    n_sample_perclass = int((SAMPLE_SIZE_TRAIN - n_sample_notopic) / (len(df_cl.label_text.unique()) - 1))
    df_train_samp1 = df_cl_group.groupby("label_text", as_index=False, group_keys=False).apply(
        lambda x: x.sample(min(n_sample_perclass, len(x)), random_state=SEED_RUN) if x.label_text.unique()[0] != "no_topic" else None)
    if "pimpo" in DATASET:
        df_train_samp2 = df_cl_group[df_cl_group.label_text == "no_topic"].sample(n_sample_notopic, random_state=SEED_RUN)
    else:
        df_train_samp2 = pd.DataFrame()
    df_train = pd.concat([df_train_samp1, df_train_samp2])
    print("Sample that might be imbalanced: df_train.label_text.value_counts:\n", df_train.label_text.value_counts())

    # check if n_samples per class correspond to harmonized n_sample_perclass
    df_train_label_distribution = df_train.label_text.value_counts()
    train_label_distribution_not_standard = df_train_label_distribution[~(df_train_label_distribution == n_sample_perclass)]
    all_labels_have_standard_n_samples = len(train_label_distribution_not_standard) == 0
    # check if labels that do not have length of standard labels ("no_topic") have exactly the length of n_sample_notopic and are called "no_topic"
    if TRAIN_NOTOPIC_PROPORTION_TRAIN == 0:
        special_label_correct = False
    elif not all_labels_have_standard_n_samples:
        special_label_has_correct_length = (train_label_distribution_not_standard == n_sample_notopic).tolist()
        special_label_has_correct_name = (train_label_distribution_not_standard.index == "no_topic").tolist()
        special_label_correct = all(special_label_has_correct_length + special_label_has_correct_name)
    else:
        special_label_correct = False
    if (all_labels_have_standard_n_samples) or (not all_labels_have_standard_n_samples and special_label_correct):
        imbalanced_sample = False
    counter += 1

if counter == 11:
    raise ValueError("could not sample balanced training data after 10 iterations")
print(f"\nFINAL DF_TRAIN SAMPLE (BALANCED) for group {group_join}:\ndf_train.label_text.value_counts:\n", df_train.label_text.value_counts())




# remove N no_topic & downsample for faster testing
#if "pimpo" in DATASET:
if EXECUTION_TERMINAL == False:
    # reduce no-topic to N
    df_test = df_test.groupby(by="label_text", as_index=False, group_keys=False).apply(
        lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_RUN) if x.label_text.iloc[0] == "no_topic" else x)
    # reduce entire test-set to N
    df_test = df_test.sample(n=min(SAMPLE_SIZE_TEST, len(df_test)), random_state=SEED_RUN)
    print("df_test.label_text.value_counts:\n", df_test.label_text.value_counts())





##### code specific to classical_ml

## lemmatize prepared text
# TODO: put in different script upstream (with embeddings probably)
import spacy

nlp = spacy.load("en_core_web_md")
nlp.batch_size

def lemmatize_and_stopwordremoval(text_lst):
    texts_lemma = []
    for doc in nlp.pipe(text_lst, n_process=1, disable=["ner"]):  # disable=["tok2vec", "ner"] "tagger", "attribute_ruler", "parser",
        doc_lemmas = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]
        # if else in case all tokens are deleted due to stop word removal
        if not any(pd.isna(doc_lemmas)):
            doc_lemmas = " ".join(doc_lemmas)
            texts_lemma.append(doc_lemmas)
        else:
            print(doc)
            texts_lemma.append(doc.text)
    return texts_lemma


df_train["text_prepared"] = lemmatize_and_stopwordremoval(df_train.text_prepared)
df_test["text_prepared"] = lemmatize_and_stopwordremoval(df_test.text_prepared)
print("Spacy lemmatization done")


if METHOD in ["standard_dl", "dl_embed", "classical_ml"]:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)







##### train classifier

## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_SVM_tfidf_01000samp_20221006.pkl")
"""import joblib

n_sample_str = MAX_SAMPLE
while len(str(n_sample_str)) <= 3:
    n_sample_str = "0" + str(n_sample_str)

hp_study_dic = joblib.load(f"./results/{DATASET}/optuna_study_logistic_tfidf_{n_sample_str}samp_{DATASET}_20230207_t.pkl")


hyperparams_vectorizer = hp_study_dic['optuna_study'].best_trial.user_attrs["hyperparameters_vectorizer"]
hyperparams_clf = hp_study_dic['optuna_study'].best_trial.user_attrs["hyperparameters_classifier"]
print("Hyperparameters vectorizer: ", hyperparams_vectorizer)
print("Hyperparameters classifier: ", hyperparams_clf)

# ! restrict to word-level vectorizer for easier interpretability
hyperparams_vectorizer["analyzer"] = "word"
hyperparams_vectorizer["ngram_range"] = (1, 1)
hyperparams_vectorizer["min_df"] = 1
"""

hyperparams_vectorizer = {"analyzer": "word", "ngram_range": (1, 1), "min_df": 1}
hyperparams_clf = {"random_state": SEED_RUN}
HYPER_PARAMS = {**hyperparams_vectorizer, **hyperparams_clf}


### text pre-processing

# choose correct pre-processed text column here
import ast
if VECTORIZER == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"

    # tests: remove group specific tokens
    """if "random" not in GROUP:
        df_coef_diff_group = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/meta-metrics/meta-metrics-repo/data-classified/pimpo/df_tokens_pimpo_{GROUP}_samp1000_20230427.csv")
        tokens_group_specific = df_coef_diff_group["token"].tolist()[:N_TOKENS_REMOVE]
        tokens_group_specific = [r"\b" + token + r"\b" for token in tokens_group_specific]
        if N_TOKENS_REMOVE > 0:
            # exclude rows in df_test_format that contains any of the group specific tokens in text_prepared column
            #df_test_format = df_test_format[~df_test_format.text_prepared.str.contains("|".join(tokens_group_specific))]
            # replace group specific tokens with empty string
            df_test_format.text_prepared = df_test_format.text_prepared.str.replace("|".join(tokens_group_specific), "", regex=True)"""

        #pd.Series("We need to work with the law. workers unite!").str.replace("|".join(tokens_group_specific), "", regex=True)

    # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
    # ! using columns here that are already bag-of-worded
    vectorizer_sklearn.fit(pd.concat([df_train_format.text_prepared, df_test_format.text_prepared]))
    X_train = vectorizer_sklearn.transform(df_train_format.text_prepared)
    X_test = vectorizer_sklearn.transform(df_test_format.text_prepared)

    #X_test_ood = vectorizer_sklearn.transform(df_test_format_ood.text_prepared)
else:
    raise NotImplementedError

y_train = df_train_format.labels
y_test = df_test_format.labels



## initialise and train classifier
from sklearn import svm, linear_model
if MODEL_NAME == "svm":
    clf = svm.SVC(**hyperparams_clf)
elif MODEL_NAME == "log_reg":
    clf = linear_model.LogisticRegression(**hyperparams_clf)

# train
start_time_train = time.time()

clf.fit(X_train, y_train)

end_time_train = time.time()
train_time = end_time_train - start_time_train
print("\nTrain time:", train_time, "\n")


### Evaluate

label_gold = y_test
label_pred = clf.predict(X_test)


### metrics
from helpers import compute_metrics_classical_ml
results_test = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df_cl.label_text.unique()))

print("\nTest results:")
results_test_cl = {key: value for key, value in results_test.items() if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]}
print(results_test_cl)



### save results
import pickle
import gzip

n_sample_str = SAMPLE_SIZE_TRAIN
while len(str(n_sample_str)) <= 3:
    n_sample_str = "0" + str(n_sample_str)


# merge prediction results with df_test to enable possible meta-data calculations etc. later
df_results = pd.DataFrame(
    {"label_pred": results_test["eval_label_predicted_raw"], "label_gold_pred_aligned": results_test["eval_label_gold_raw"]},
    index=df_test.index
)
df_test_results = pd.concat([df_test, df_results], axis=1)

if "generation" in METHOD:
    assert (df_test_results["label_text"] == df_test_results["label_gold_pred_aligned"]).all(), "label_text and label_gold_pred_aligned should be the same"
else:
    assert (df_test_results["labels"] == df_test_results["label_gold_pred_aligned"]).all(), "labels and label_gold_pred_aligned should be the same"


data_dic = {
    "experiment_metadata": {
        "dataset": DATASET, "group_sample_strategy": GROUP_SAMPLE, "group_col": GROUP_COL, "method": METHOD,
        "model_name": MODEL_NAME, "sample_size_train": SAMPLE_SIZE_TRAIN, #"sample_size_test": SAMPLE_SIZE_TEST,
        "group_members": group_join.replace("\\b", ""), "seed_run": SEED_RUN, "n_run": N_RUN, "date": DATE, "hyperparams": HYPER_PARAMS,
        "train_time": train_time, "model_size": MODEL_SIZE, "task": TASK, "model_params": None, "generation_config": None,
    },
    "experiment_results": results_test_cl,
    "df_train": df_train,
    "df_test_results": df_test_results,
}

if SAVE_OUTPUTS:
    filename = f"./results/{DATASET}/results_{DATASET}_{GROUP_SAMPLE}_{GROUP_COL}_{METHOD}_{MODEL_NAME_SHORT}_samp{n_sample_str}_n_run{N_RUN}_seed{SEED_RUN}_{DATE}.pkl.gz"
    # Use 'wb' to write binary data
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data_dic, f)



print("\nScript done.\n\n")



