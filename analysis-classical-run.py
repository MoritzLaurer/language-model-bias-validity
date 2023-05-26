

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
import datasets
import tqdm



# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

# main args
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
parser.add_argument('-date', '--study_date', type=str,
                    help='Date')

#parser.add_argument('-size', '--model_size', type=str,
#                    help='base or large')
#parser.add_argument('-hypo', '--hypothesis', type=str,
#                    help='which hypothesis?')
parser.add_argument('-t', '--task', type=str,
                    help='task about integration or immigration?')
parser.add_argument('-iter', '--n_iteration', type=int, #nargs='+',
                    help='The number of the respective random iteration')
parser.add_argument('-iter_max', '--n_iterations_max', type=int, #nargs='+',
                    help='The total number of random iteratio')
parser.add_argument('-g', '--group', type=str,
                    help='group to filter training data by')
parser.add_argument('-n_tok_rm', '--n_tokens_remove', type=int, #nargs='+',
                    help='number of group-specific tokens to remove from test data')
parser.add_argument('-save', '--save_outputs', action="store_true",
                    help='boolean whether to save outputs to disk')

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
  args = parser.parse_args(["--task", "pimpo-simple",  # pimpo, uk-leftright-simple, uk-leftright
                            "--dataset", "pimpo",  # uk-leftright-econ, pimpo
                            "--vectorizer", "tfidf",
                            "--model", "logistic",
                            "--method", "classical_ml",
                            "--sample_size", "1000", "--study_date", "20230427",
                            "--n_iteration", "1", "--n_iterations_max", "5",
                            "--group", "random2", "--n_tokens_remove", "0", #"--save_outputs"
                            ])

#LANGUAGE_LST = args.languages.split("-")
DATASET = args.dataset
TRAINING_DIRECTORY = f"results/{DATASET}"

MAX_SAMPLE = args.sample_size
DATE = args.study_date
TASK = args.task
METHOD = args.method
MODEL_NAME = args.model
VECTORIZER = args.vectorizer
GROUP = args.group
N_TOKENS_REMOVE = args.n_tokens_remove
SAVE_OUTPUTS = args.save_outputs

# set global seed for reproducibility and against seed hacking
N_ITER = args.n_iteration - 1
N_ITER_MAX = args.n_iterations_max
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# special variables for pimpo
SAMPLE_NO_TOPIC = 5_000
TRAIN_NOTOPIC_PROPORTION = 0.4

# randomly assign different seeds for each run
seed_runs_all = np.random.choice(range(1000), size=N_ITER_MAX)
SEED_RUN = seed_runs_all[N_ITER]
print("Iteration number: ", N_ITER)
print("All random seeds: ", seed_runs_all)
print("Random seed for this run: ", SEED_RUN)

# not sure if I should keep TASK variable
assert DATASET.split("-")[0] in TASK, f"Mismatch between dataset {DATASET} and task {TASK}"




# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import compute_metrics_standard, clean_memory, compute_metrics_nli_binary
#from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset


##### load dataset
if DATASET == "uk-leftright-econ":
    df = pd.read_csv(f"./data-clean/benoit_leftright_sentences.zip", engine='python')
    df_cl = df.copy(deep=True)
elif "pimpo" in DATASET:
    # df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/data-clean/df_pimpo_samp_trans_m2m_100_1.2B_embed_tfidf.zip", engine='python')
    df = pd.read_csv("./data-clean/df_pimpo_samp_trans_lemmatized_stopwords.zip", engine="python")
    df_cl = df.copy(deep=True)
else:
    raise Exception(f"Dataset name not found: {DATASET}")

### preprocessing data

## uk-leftright
if "uk-leftright-econ" in DATASET:
    # select to work with crowd annotations and expert annotations
    df_cl = df_cl[df_cl.source == "Crowd"]
    # select task on either economy or social policy
    df_cl = df_cl[df_cl.scale == "Economic"]
    # transform continuous float data to categorical classes
    df_cl["label_scale"] = df_cl.score_sent_mean.fillna(0).round().astype(int)
    print(df_cl["label_scale"].value_counts())
    # prepare input data
    df_cl["text_prepared"] = df_cl["text_preceding"].fillna('') + " " + df_cl["text_original"] + " " + df_cl["text_following"].fillna('')
elif "uk-leftright-soc" in DATASET:
    raise NotImplementedError

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

if "pimpo-simple" in TASK:
    task_label_text_map = {
        'immigration_neutral': "neutral", 'integration_neutral': "neutral",
        'immigration_sceptical': "sceptical", 'integration_sceptical': "sceptical",
        'immigration_supportive': "supportive", 'integration_supportive': "supportive",
        'no_topic': "no_topic"
    }
    df_cl["label_text"] = df_cl.label_text.map(task_label_text_map)
    df_cl["label"] = df_cl.label_text.factorize(sort=True)[0]

df_cl.label_text.unique()

"""if TASK == "integration":
    #task_label_text = ["integration_supportive", "integration_sceptical", "integration_neutral", "no_topic"]
    raise NotImplementedError
elif TASK == "immigration":
    #task_label_text = ["immigration_supportive", "immigration_sceptical", "immigration_neutral", "no_topic"]
    raise NotImplementedError"""

## prepare input text data
if VECTORIZER == "tfidf":
    if METHOD == "classical_ml":
        df_cl["text_prepared"] = df_cl["text_preceding"].fillna('') + " " + df_cl["text_original"] + " " + df_cl["text_following"].fillna('')
elif VECTORIZER == "transformer":
    if METHOD == "nli":
        df_cl["text_prepared"] = df_cl["text_preceding_trans"].fillna('') + '  || The quote: "' + df_cl["text_original_trans"] + '" End of the quote ||  ' + df_cl["text_following_trans"].fillna('')
else:
    raise Exception(f"Vectorizer {VECTORIZER} or METHOD {METHOD} not implemented.")


## data checks
#print("Dataset: ", DATASET, "\n")
# verify that numeric label is in alphabetical order of label_text (can avoid issues for NLI)
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)


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


if "uk-rightleft" in DATASET:
    df_cl["text_prepared"] = lemmatize_and_stopwordremoval(df_cl.text_prepared)
    print("Spacy lemmatization done")
elif "pimpo" in DATASET:
    # re-using translated, concatenated, lemmatized, stopword-cleaned column from multilingual paper
    df_cl["text_prepared"] = df_cl["text_trans_concat_tfidf"]


## add left/right aggreg parfam to df
parfam_aggreg_map = {"ECO": "left", "LEF": "left", "SOC": "left",
                     "CHR": "right", "CON": "right", "NAT": "right",
                     "LIB": "other", "AGR": "other", "ETH": "other", "SIP": "other"}
df_cl["parfam_text_aggreg"] = df_cl.parfam_text.map(parfam_aggreg_map)



### select training data

df_cl.groupby("parfam_text").apply(lambda x: x.label_text.value_counts())
# parfam with > 100 for each class. CHR, LEF, LIB, NAT, SOC. (less: (ECO) SIP, ETH, CON, AGR)
if "pimpo" in DATASET:
    col_group_map = {}
    col_group_map.update(**{parfam: "parfam_text" for parfam in df_cl.parfam_text.unique()})
    col_group_map.update(**{country: "country_iso" for country in df_cl.country_iso.unique()})
else:
    raise NotImplementedError


if "random3" in GROUP:
    # TODO: implement for more than countries and make more general
    import random
    random.seed(SEED_RUN)
    GROUP_join = random.sample(list(df_cl.country_iso.unique()), 3)
    GROUP_join = '|'.join(GROUP_join)
    print(GROUP_join)
elif "random2" in GROUP:
    import random
    random.seed(SEED_RUN)
    group_enough_data = ["nld", "esp", "deu", "dnk", "aut"]
    GROUP_join = random.sample(group_enough_data, 2)
    GROUP_join = '|'.join(GROUP_join)
    print(GROUP_join)

#print(df_cl.groupby("country_iso").apply(lambda x: x.label_text.value_counts()))

# sample training data
if "uk-leftright" in DATASET:
    df_train = df_cl.sample(n=MAX_SAMPLE, random_state=SEED_RUN)
elif "pimpo" in DATASET:
    # ! test: only sample df_train from one country/group
    if ("random3" in GROUP) or ("random2" in GROUP):
        df_cl_group = df_cl[df_cl.country_iso.str.contains(GROUP_join)].copy(deep=True)
    elif "random" not in GROUP:
        df_cl_group = df_cl[df_cl[col_group_map[GROUP]] == GROUP ].copy(deep=True)
    else:
        df_cl_group = df_cl.copy(deep=True)

    # sample x% of training data for no topic, then share the remainder equally across classes
    n_sample_notopic = int(MAX_SAMPLE * TRAIN_NOTOPIC_PROPORTION)
    n_sample_perclass = int((MAX_SAMPLE - n_sample_notopic) / (len(df_cl.label_text.unique()) - 1))
    df_train_samp1 = df_cl_group.groupby("label_text", as_index=False, group_keys=False).apply(
        lambda x: x.sample(min(n_sample_perclass, len(x)), random_state=SEED_RUN) if x.label_text.unique()[0] != "no_topic" else None)
    df_train_samp2 = df_cl_group[df_cl_group.label_text == "no_topic"].sample(n_sample_notopic, random_state=SEED_RUN)
    df_train = pd.concat([df_train_samp1, df_train_samp2])

    # ! only for feature matrix over entire corpus (deletable)
    #if GROUP == "randomall":
    #    df_train = df_cl.groupby("label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=len(x)-1, random_state=42))

    # ! for testing, remove remaining data from group from df_cl
    #df_cl_group_rest = df_cl_group[~df_cl_group.index.isin(df_train.index)]
    #df_cl = df_cl[~df_cl.index.isin(df_cl_group_rest.index)]

print("df_train.label_text.value_counts:\n", df_train.label_text.value_counts())

# !! make sure that df_train group has same length as df_train random
#df_cl.groupby("country_iso", as_index=False, group_keys=False).apply(lambda x: print(x.country_iso.iloc[0], "\n", x.label_text.value_counts()))
# countries with at least 100 per class: nld, esp, dnk, deu

# create df test
df_test = df_cl[~df_cl.index.isin(df_train.index)]
# also remove all GROUP from df_test?
# TODO make work beyond countries
if ("random3" in GROUP) or ("random2" in GROUP):
    df_test = df_test[~df_test.country_iso.str.contains(GROUP_join)].copy(deep=True)
elif "randomall" in GROUP:
    # TODO: same as above, clean later
    df_test = df_cl[~df_cl.index.isin(df_train.index)]
else:
    df_test = df_test[~df_test.country_iso.str.contains(GROUP)].copy(deep=True)
#assert len(df_train) + len(df_test) == len(df_cl)

# remove N no_topic for faster testing
if "pimpo" in DATASET:
    # reduce no-topic to N
    df_test = df_test.groupby(by="label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_RUN) if x.label_text.iloc[0] == "no_topic" else x)
    # reduce entire test-set to N
    df_test = df_test.sample(n=min(SAMPLE_NO_TOPIC, len(df_test)), random_state=SEED_RUN)

print("df_test.label_text.value_counts:\n", df_test.label_text.value_counts())


if METHOD in ["standard_dl", "dl_embed", "classical_ml"]:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
#elif METHOD == "nli":
#    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=42)
#    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)



### test splitting df_test in in-distribution and OOD split
## random ood split
#df_test_format_ood = df_test_format.sample(frac=0.5, random_state=SEED_RUN)
#df_test_format = df_test_format[~df_test_format.index.isin(df_test_format_ood.index)]
## by-group ood split - unclear if useful (deletable?)
"""groups_all = df_test_format.country_iso.unique()
groups_sample_size = int(len(df_test_format.country_iso.unique()) / 2)
groups_ood = np.random.choice(groups_all, size=groups_sample_size, replace=False)
df_test_format_ood = df_test_format[df_test_format.country_iso.isin(groups_ood)]
df_test_format = df_test_format[~df_test_format.country_iso.isin(groups_ood)]"""
"""groups_majority_size = int(len(df_train_format.country_iso.unique()) / 4)
groups_majority = df_train_format.country_iso.value_counts().nlargest(n=groups_majority_size).index
df_test_format_ood = df_test_format[~df_test_format.country_iso.isin(groups_majority)]
df_test_format = df_test_format[df_test_format.country_iso.isin(groups_majority)]"""
## biased by-group ood split
# specific groups are spuriously linked to specific classes
# downsample left-parties to contain no con-migration texts & downsample right-parties contain no pro-migration texts
"""df_train_left_pro = df_train_format[df_train_format.label_text.str.contains("supportive|neutral|no_topic")
                                     & df_train_format.parfam_text_aggreg.str.contains("left")]
df_train_right_con = df_train_format[df_train_format.label_text.str.contains("sceptical|neutral|no_topic")
                                     & df_train_format.parfam_text_aggreg.str.contains("right")]
df_train_format = pd.concat([df_train_left_pro, df_train_right_con])
# same biased test set
df_test_left_pro = df_test_format[df_test_format.label_text.str.contains("supportive|neutral|no_topic")
                                     & df_test_format.parfam_text_aggreg.str.contains("left")]
df_test_right_con = df_test_format[df_test_format.label_text.str.contains("sceptical|neutral|no_topic")
                                     & df_test_format.parfam_text_aggreg.str.contains("right")]
df_test_format = pd.concat([df_test_left_pro, df_test_right_con])"""
# unbiased test set
#df_test_format_ood


### Test injecting noise/bias tokens

"""def add_noise_token(df, fraction_noise=1, noise_column_name="label_text"):
    df_samp_noise = df.sample(frac=fraction_noise, random_state=SEED_RUN)
    if noise_column_name == "label_text":
        df_samp_noise["text_prepared"] = f"{noise_column_name}_" + str(df_samp_noise["label"].iloc[0]) + " " + df_samp_noise["text_prepared"]
    else:
        df_samp_noise["text_prepared"] = f"{noise_column_name}_" + str(df_samp_noise[noise_column_name].iloc[0]) + " " + df_samp_noise["text_prepared"]
    df_nonoise = df[~df.index.isin(df_samp_noise.index)]
    df_noise = pd.concat([df_samp_noise, df_nonoise])
    return df_noise

## Test injecting meaningless label token for each label
df_train_format = df_train_format.groupby(by="label_text", as_index=False, group_keys=False).apply(
    lambda x: add_noise_token(x, fraction_noise=0.01))

df_test_format = df_test_format.groupby(by="label_text", as_index=False, group_keys=False).apply(
    lambda x: add_noise_token(x, fraction_noise=0.01))
"""
## Test injecting meaningless group token in groups that spuriously correlate with labels
"""df_train_format = df_train_format.groupby(by="parfam_text", as_index=False, group_keys=False).apply(
    lambda x: add_noise_token(x, fraction_noise=0.8, noise_column_name="parfam_text"))

df_test_format = df_test_format.groupby(by="parfam_text", as_index=False, group_keys=False).apply(
    lambda x: add_noise_token(x, fraction_noise=0.8, noise_column_name="parfam_text"))"""
# ! also without injecting meaningless group token, there is difference between biased in-distribution and OOD test data



##### train classifier

# ! potential criticism for UK-left-right data: why use a classifier that is also like scaling, or a regression task ?!
# one argument: only a small fraction of texts was actually classified in extremes by crowd.

## ! do hp-search in separate script. should not take too long
## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_SVM_tfidf_01000samp_20221006.pkl")
import joblib

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

y_train = df_train_format.label
y_test = df_test_format.label
#y_test_ood = df_test_format_ood.label


#### tests with fairlearn
### correlation remover
# https://fairlearn.org/main/user_guide/mitigation.html#correlation-remover
"""from fairlearn.preprocessing import CorrelationRemover

series_tfidf_dense = [pd.Series(matrix.data, index=matrix.indices) for matrix in X_train]
# create dummy series with only 0 values for all indexes in matrix
series_dummy_0 = pd.Series([0] * (X_train.indices.max() + 1), index=range(X_train.indices.max() + 1))
# combine dense tfidf matrix with dummy 0 values matrix to get sparse matrix
series_tfidf_sparse = [series_dense.combine(series_dummy_0, max, fill_value=0).tolist() for series_dense in series_tfidf_dense]
# create df combining matrix and meta-data
#X = pd.DataFrame({"tfidf": series_tfidf_sparse, **df_train_format[["party", "year"]]}, dtype=object)
#X["tfidf"] = X["tfidf"].astype(object)

X = pd.DataFrame(series_tfidf_sparse, index=df_train_format.index)
X["year"] = df_train_format["year"]  # "party", "year"
X["party"] = pd.factorize(df_train_format["party"])[0]  # "party", "year"

# old working code from other script as reference
#df_embed_lst = pd.DataFrame([ast.literal_eval(lst) for lst in df_train.text_original_trans_embed_multi.astype('object')])
#df_embed_lst["label_rile"] = pd.factorize(df_train.label_rile.tolist())[0]
#X = df_embed_lst

cr = CorrelationRemover(sensitive_feature_ids=['year', "party"], alpha=1)
cr.fit(X)
X_train = cr.transform(X)"""



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

#label_gold_ood = y_test_ood
#label_pred_ood = clf.predict(X_test_ood)

### metrics
from helpers import compute_metrics_classical_ml
#results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
print("\nTest results:")
results_test = compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=np.sort(df_cl.label_text.unique()))
#print("Aggregate metrics: ", {key: results_test[key] for key in results_test if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]})  # print metrics but without label lists

## results ood
#print("Out-of-distribution test results:")
#results_test_ood = compute_metrics_classical_ml(label_pred_ood, label_gold_ood, label_text_alphabetical=np.sort(df_cl.label_text.unique()))
#print("Aggregate metrics, out-of-distribution test results: ", {key: results_test_ood[key] for key in results_test_ood if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]})  # print metrics but without label lists

df_results = pd.DataFrame([results_test])
if SAVE_OUTPUTS:
    #df_results.to_csv(f"./data-classified/{DATASET}/df_results_{DATASET}_{GROUP}_samp{n_sample_str}_tokrm{N_TOKENS_REMOVE}_seed{SEED_RUN}_{DATE}.csv", index=False)
    df_results.to_csv(f"./results/{DATASET}/df_results_{DATASET}_{GROUP}_{METHOD}_samp{n_sample_str}_seed{SEED_RUN}_{DATE}.csv", index=False)



# do prediction on entire corpus? No.
# could do because annotations also contribute to estimate of distribution
#dataset["all"] = datasets.concatenate_datasets([dataset["test"], dataset["train"]])
#results_corpus = trainer.evaluate(eval_dataset=datasets.concatenate_datasets([dataset["train"], dataset["test"]]))  # eval_dataset=encoded_dataset["test"]
# with NLI, cannot run inference also on train set, because augmented train set can have different length than original train-set


#### prepare data for redoing figure from paper
"""assert (df_test["label"] == results_test["eval_label_gold_raw"]).all
df_test["label_pred"] = results_test["eval_label_predicted_raw"]
df_test_format["label_pred"] = results_test["eval_label_predicted_raw"]

## add classification probabilities to data to use for scale
probabilities = clf.predict_proba(X_test)
probabilities_max = [np.max(per_text_probability) for per_text_probability in probabilities]
df_test_format["label_pred_probability"] = probabilities_max

df_train["label_pred"] = [np.nan] * len(df_train["label"])

df_cl_concat = pd.concat([df_train, df_test])

# add label text for predictions
label_text_map = {}
for i, row in df_cl_concat[~df_cl_concat.label_text.duplicated(keep='first')].iterrows():
    label_text_map.update({row["label"]: row["label_text"]})
df_cl_concat["label_text_pred"] = df_cl_concat["label_pred"].map(label_text_map)

## translate label pred back to -2 to +2 labels to enable mean calculation for correlation
if TASK == "uk-leftright":
    task_label_text_map_reversed = {value: key for key, value in task_label_text_map.items()}
    df_cl_concat["label_scale_pred"] = df_cl_concat.label_text_pred.map(task_label_text_map_reversed)
# in case of simplified -1 to +1 scale
elif TASK == "uk-leftright-simple":
    task_label_text_map_reversed = {value: key for key, value in task_label_text_map.items() if key not in [-2, 2]}
    df_cl_concat["label_scale_pred"] = df_cl_concat.label_text_pred.map(task_label_text_map_reversed)
"""

## trim df to save storage
"""df_cl_concat = df_cl_concat[['label', 'label_text', 'country_iso', 'language_iso', 'doc_id',
                           'text_original', 'text_original_trans', 'text_preceding_trans', 'text_following_trans',
                           # 'text_preceding', 'text_following', 'selection', 'certainty_selection', 'topic', 'certainty_topic', 'direction', 'certainty_direction',
                           'rn', 'cmp_code', 'partyname', 'partyabbrev',
                           'parfam', 'parfam_text', 'date', #'language_iso_fasttext', 'language_iso_trans',
                           #'text_concat', 'text_concat_embed_multi', 'text_trans_concat',
                           #'text_trans_concat_embed_en', 'text_trans_concat_tfidf', 'text_prepared',
                           'label_pred', 'label_text_pred']]"""



##### cleanlab tests
# https://docs.cleanlab.ai/stable/index.html
"""import cleanlab
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from cleanlab.classification import CleanLearning

# display potential issues in data based on classifier uncertainty
X_all = vectorizer_sklearn.transform(df_cl.text_prepared)
y_all = df_cl.label
#X_all = vectorizer_sklearn.transform(pd.concat([df_train_format.text_prepared, df_test_format.text_prepared]))
#y_all = pd.concat([df_train_format.label, df_test_format.label])
cl = CleanLearning(clf=LogisticRegression(**hyperparams_clf))  # any sklearn-compatible classifier
df_issues  = cl.find_label_issues(X_all, y_all)
#df_issues = CleanLearning(clf=clf).find_label_issues(X_test, y_test)

df_issues_merge = pd.concat([df_test[["text_prepared", "text_original", "label_text", "party"]].reset_index(), df_issues], axis=1)
df_issues_merge = df_issues_merge.sort_values("label_quality", ascending=False)

### training robust clf with noisy labels
cl = CleanLearning(clf=LogisticRegression(**hyperparams_clf))  # any sklearn-compatible classifier
# train on train set
cl.fit(X_train, y_train)
# fit on entire corpus (with cross-val)
#cl.fit(X_all, y_all)
#predictions = cl.predict(X_test)

## compare vanilla to cleanlab model
clf.score(X_test, y_test)
cl.score(X_test, y_test)

### finding overlapping classes
df_class_overlap = cleanlab.dataset.find_overlapping_classes(
    labels=y_test,
    confident_joint=cl.confident_joint,  # cleanlab uses the confident_joint internally to quantify label noise (see cleanlab.count.compute_confident_joint)
    class_names=np.sort(df_cl.label_text.unique()),
)

## dataset curation
from cleanlab.dataset import health_summary
health = cleanlab.dataset.overall_label_health_score(y_test, confident_joint=cl.confident_joint)
#label_acc = sum(predictions != y_test) / len(predictions)
#health_summary(X_test, y_test.to_numpy(), class_names=np.sort(df_cl.label_text.unique()))
"""




##### save data
#langs_concat = "_".join(LANGUAGE_LST)
#df_cl_concat.to_csv(f"./data-classified/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE}samp_{langs_concat}_{DATE}.csv", index=False)
#df_cl_concat.to_csv(f"./data-classified/{DATASET}/df_{DATASET}_{n_sample_str}_{MODEL}_{DATE}_{N_ITER}.zip",
#                    compression={"method": "zip", "archive_name": f"df_{DATASET}_{n_sample_str}_{MODEL}_{DATE}_{N_ITER}.csv"}, index=False)











###### Interpretability tests

# feature importance
#clf._feature_importances
#clf.coef_

# vocabulary in order of feature importance scores (in order of index)
vectorizer_sklearn.vocabulary_
vocab_sorted = {k: v for k, v in sorted(vectorizer_sklearn.vocabulary_.items(), key=lambda item: item[1])}

if "uk-leftright" in DATASET:
    task_label_text_map_factorized = {"neutral": 1, "right": 2, "right": 2, "left": 0, "left": 0}
elif "pimpo" in DATASET:
    task_label_text_map_factorized = {label_text: i for i, label_text in enumerate(df_cl.label_text.factorize(sort=True)[1])}

coef_dic = {}
for key, value in task_label_text_map_factorized.items():
    coef_dic.update({f"coef_{key}": clf.coef_[value]})
# factorize provides label_text in same order as numerical labels
#for i, label_text in enumerate(pd.factorize(df_cl["label_text"], sort=True)[1]):
#    coef_dic.update({f"coef_{label_text}": clf.coef_[i]})

df_feat_importance = pd.DataFrame({"token": vocab_sorted.keys(), **coef_dic})
# high idf means low frequency, low idf means high frequency
#df_test_format.text_prepared.str.contains(r"\bpeople\b", case=False).sum()
df_feat_importance["inverse_doc_freq"] = df_feat_importance["token"].apply(lambda x: vectorizer_sklearn.idf_[vectorizer_sklearn.vocabulary_[x]])
df_feat_importance = df_feat_importance.sort_values(by=list(coef_dic.keys())[0], ascending=False)


# save df_feat_importance for further comparative tests
if SAVE_OUTPUTS:
    df_feat_importance.to_csv(f"./data-classified/{DATASET}/df_feat_importance_{DATASET}_{GROUP}_samp{n_sample_str}_seed{SEED_RUN}_{DATE}.csv", index=False)

## get probabilities for each class   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# integer prediction
clf.predict(X_test)
# confidence scores
clf.predict_proba(X_test)
clf.decision_function(X_test)
clf.predict_log_proba(X_test)



#### explain individual classification decisions
# https://github.com/TeamHG-Memex/eli5
import eli5

### eli5 tests (only for individual decision)
# only works in ipython notebook
"""eli5.show_weights(clf)
#eli5.show_prediction(clf)

eli5.explain_prediction(clf, X_test[0])

# confert outputs to df https://eli5.readthedocs.io/en/latest/autodocs/formatters.html
eli5.formatters.as_dataframe.explain_weights_df(clf)
eli5.formatters.as_dataframe.explain_prediction_df(clf, X_test[0])

## explain individual classification decision
# create eli df
df_eli = eli5.formatters.as_dataframe.explain_prediction_df(clf, X_test[0])
# map token indices to tokens
df_eli["feature"] = df_eli.feature.apply(lambda x: x.replace("x", ""))
vocab_dic_inverted = inv_map = {str(v): k for k, v in vocab_sorted.items()}
df_eli["feature_text"] = df_eli["feature"].map(vocab_dic_inverted)
# map labels to label text
#label_label_text_map = {i: text for i, text in enumerate(pd.factorize(df_cl["label_text"], sort=True)[1])}
df_eli["target_text"] = df_eli["target"].map({value: key for key, value in task_label_text_map_factorized.items()})

# "weight" displays the importance of respective token for respective class
df_eli = df_eli.sort_values(["target", "weight"], ascending=False)
"""



### key features for classification decisions by meta-data groups (aggregating individual decisions)

# ! do analysis only on wrongly classified texts
#df_test_format = df_test_format[df_test_format.label != df_test_format.label_pred]
"""
#n_iter = 0
df_feat_importance_group_lst_lst = []
for key_group, df_group in tqdm.tqdm(df_test_format.reset_index(drop=True).groupby(by=["party"])):  # country_iso, parfam_text
    # get vectorized test data for each group
    X_test_group = X_test[df_group.index]

    df_feat_importance_group_lst = []
    for text_matrix, df_group_row in zip(X_test_group, df_group.iterrows()):
        #df_group_row[0] is index of row, df_group_row[1] row as series
        #df_eli_group = eli5.explain_prediction(clf, text_matrix)
        df_eli_group = eli5.formatters.as_dataframe.explain_prediction_df(clf, text_matrix)
        # select only feature scores for predicted class (could also select for true class)
        df_eli_group = df_eli_group[df_eli_group.target == df_group_row[1].label_pred]
        df_eli_group["text_id"] = df_group_row[0]
        # select only X features with largest weight
        df_eli_group = df_eli_group.nlargest(min(len(df_eli_group), 5), "weight")
        df_feat_importance_group_lst.append(df_eli_group)

    df_feat_importance_group = pd.concat(df_feat_importance_group_lst)
    df_feat_importance_group["group"] = key_group
    df_feat_importance_group_lst_lst.append(df_feat_importance_group)
    #df_feat_importance_group_dic.update({key_group: df_feat_importance_group})

    #n_iter += 1
    #if n_iter == 2:
    #    break

df_feat_importance_group_all = pd.concat(df_feat_importance_group_lst_lst)
df_feat_importance_group_all = df_feat_importance_group_all[df_feat_importance_group_all.feature != "<BIAS>"]

# map token indices to tokens
df_feat_importance_group_all["feature"] = df_feat_importance_group_all.feature.apply(lambda x: x.replace("x", ""))
vocab_dic_inverted = inv_map = {str(v): k for k, v in vocab_sorted.items()}
df_feat_importance_group_all["feature_text"] = df_feat_importance_group_all["feature"].map(vocab_dic_inverted)
# map labels to label text
#label_label_text_map = {i: text for i, text in enumerate(pd.factorize(df_cl["label_text"], sort=True)[1])}
task_label_text_map_factorized_reverse = {value: key for key, value in task_label_text_map_factorized.items()}
df_feat_importance_group_all["label_text"] = df_feat_importance_group_all["target"].map(task_label_text_map_factorized_reverse)

# sort
df_feat_importance_group_all = df_feat_importance_group_all.sort_values(["group", "text_id", "weight"], ascending=False)

## create aggregate statistics by group
df_feat_importance_group_all_aggreg = df_feat_importance_group_all.groupby(by=["group", "label_text", "feature_text"], as_index=False, group_keys=False).apply(lambda x: x.weight.mean())
df_feat_importance_group_all_aggreg = df_feat_importance_group_all_aggreg.rename(columns={None: "weight"})
df_feat_importance_group_all_aggreg = df_feat_importance_group_all_aggreg.groupby(by=["group", "label_text"]).apply(lambda x: x.nlargest(5, "weight")).reset_index(drop=True)
df_feat_importance_group_all_aggreg = df_feat_importance_group_all_aggreg.sort_values(["label_text", "group", "weight"], ascending=False)



### intermediate conclusion (from pimpo)
# important features seem to be similar across countries and parfam per class
#, both for correct and false predictions
# the key features from individual decisions seem to correspond
# to overall model coefficients per class from training data
# => in the incorrect cases, the algorithm followed the general patterns
# it missclassifies when it finds keywords that are generally associated to a class (no new insight, but shows issue of classical models)

# assumption: this is less the case with Transformers (?). They can rely more on synonyms etc. and are less bound to specific words from data_train.
# prompt-based models depend even less on data_train words, because anchored through prompt.
# not sure how feature-importance analyes can help illustrate this though
# the best way of discovering biases is test-set disaggregation and meta-metrics;
# feature-importance analysis can only reveal grave issues by highlighting clearly problematic features, more nuanced insights cannot come from word lists
# => good (disaggregated) test-set design is the key to testing content validity
# => interpretability aspect (as in feature importances) is of limited use for assessing validity

# could try to do general feature count for texts that were wrongly predicted and see which class-important-features they mostly overlap
# what would this tell me?
# maybe something like: empirically show, that if a sub-group has high overlap with the feature frequencies in data_train, then meta-metric performance is higher, otherwise lower
#   this is an issue, therefore we need methods that depend less on (transfer learning, prompting)
"""



#### Test meta-metrics for spurious token scenarios
import sklearn.metrics as skm
from fairlearn.metrics import MetricFrame, count, selection_rate, false_positive_rate
import functools

f1_macro = functools.partial(skm.f1_score, average='macro')
precision_micro = functools.partial(skm.precision_score, average='micro')
recall_micro = functools.partial(skm.recall_score, average='micro')
precision_macro = functools.partial(skm.precision_score, average='macro')
recall_macro = functools.partial(skm.recall_score, average='macro')

# ! don't understand why balanced_accuracy != recall_macro. They are the same in overall metrics, but not when disagg

metrics_dic = {
    "f1_macro": f1_macro,
    'accuracy': skm.accuracy_score,
    'accuracy_balanced': skm.balanced_accuracy_score,  # "precision_micro": precision_micro, #"recall_micro": recall_micro,
    "precision_macro": precision_macro, "recall_macro": recall_macro,
    # "selection_rate": selection_rate,  # does not seem to work on multi-class
    # "false_positive_rate": false_positive_rate,  # does not seem to work on multi-class
    'count': count,
}

## create metric frame for each group
"""mf = MetricFrame(
    metrics=metrics_dic,
    y_true=label_gold,
    y_pred=label_pred,
    # can look at intersection between features by passing df with multiple columns
    sensitive_features=df_test_format[["parfam_text"]],  # df_cl[["country_iso", "language_iso", "parfam_name", "parfam_rile", "label_rile", "decade"]]
    # Control features: "When the data are split into subgroups, control features (if provided) act similarly to sensitive features. However, the ‘overall’ value for the metric is now computed for each subgroup of the control feature(s). Similarly, the aggregation functions (such as MetricFrame.group_max()) are performed for each subgroup in the conditional feature(s), rather than across them (as happens with the sensitive features)."
    # https://fairlearn.org/v0.8/user_guide/assessment/intersecting_groups.html
    # control_features=df_cl[["parfam_text"]]
)
df_metrics_group = mf.by_group"""

# OOD metrics
"""mf_ood = MetricFrame(
    metrics=metrics_dic,
    y_true=label_gold_ood,
    y_pred=label_pred_ood,
    # can look at intersection between features by passing df with multiple columns
    sensitive_features=df_test_format_ood[["parfam_text"]],  # df_cl[["country_iso", "language_iso", "parfam_name", "parfam_rile", "label_rile", "decade"]]
    # Control features: "When the data are split into subgroups, control features (if provided) act similarly to sensitive features. However, the ‘overall’ value for the metric is now computed for each subgroup of the control feature(s). Similarly, the aggregation functions (such as MetricFrame.group_max()) are performed for each subgroup in the conditional feature(s), rather than across them (as happens with the sensitive features)."
    # https://fairlearn.org/v0.8/user_guide/assessment/intersecting_groups.html
    # control_features=df_cl[["parfam_text"]]
)
df_metrics_group_ood = mf_ood.by_group"""






print("\nScript done.\n\n")





