

import sys
if sys.argv[0] != '/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/pydevconsole.py':
    EXECUTION_TERMINAL = True
else:
    EXECUTION_TERMINAL = False
print("Terminal execution: ", EXECUTION_TERMINAL, "  (sys.argv[0]: ", sys.argv[0], ")")



# Create the argparse to pass arguments via terminal
import argparse
parser = argparse.ArgumentParser(description='Pass arguments via terminal')

# main args
parser.add_argument('-lang', '--languages', type=str, #nargs='+',
                    help='List of languages to iterate over. one string separated with separator and split in code.')
parser.add_argument('-samp_lang', '--max_sample_lang', type=int, #nargs='+',
                    help='Sample')
parser.add_argument('-max_e', '--max_epochs', type=int, #nargs='+',
                    help='number of epochs')
parser.add_argument('-date', '--study_date', type=str,
                    help='Date')
parser.add_argument('-t', '--task', type=str,
                    help='task about integration or immigration?')
parser.add_argument('-meth', '--method', type=str,
                    help='NLI or standard dl?')
parser.add_argument('-v', '--vectorizer', type=str,
                    help='en or multi?')
parser.add_argument('-hypo', '--hypothesis', type=str,
                    help='which hypothesis?')
parser.add_argument('-nmt', '--nmt_model', type=str,
                    help='Which neural machine translation model to use? "opus-mt", "m2m_100_1.2B", "m2m_100_418M" ')
parser.add_argument('-max_l', '--max_length', type=int, #nargs='+',
                    help='max n tokens')
parser.add_argument('-size', '--model_size', type=str,
                    help='base or large')


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
  args = parser.parse_args(["--languages", "en", "--max_epochs", "2", "--task", "immigration", "--vectorizer", "tfidf",
                            "--method", "classical_ml", "--hypothesis", "long", "--nmt_model", "m2m_100_1.2B", "--max_length", "256",
                            "--max_sample_lang", "500", "--study_date", "230127", "--model_size", "classical"])

LANGUAGE_LST = args.languages.split("-")

MAX_SAMPLE_LANG = args.max_sample_lang
DATE = args.study_date
#MAX_EPOCHS = args.max_epochs
TASK = args.task
METHOD = args.method
VECTORIZER = args.vectorizer
HYPOTHESIS = args.hypothesis
MT_MODEL = args.nmt_model
#MODEL_MAX_LENGTH = args.max_length
MODEL_SIZE = args.model_size

# !! align in all scripts
SAMPLE_NO_TOPIC = 50_000  #100_000
#SAMPLE_DF_TEST = 1_000

## set main arguments
SEED_GLOBAL = 42
DATASET = "pimpo"

TRAINING_DIRECTORY = f"results/{DATASET}"


if (VECTORIZER == "en") and (METHOD == "dl_embed"):
    MODEL_NAME = "transf_embed_en"
elif (VECTORIZER == "multi") and (METHOD == "dl_embed"):
    MODEL_NAME = "transf_embed_multi"
elif METHOD == "classical_ml":
    MODEL_NAME = "logistic"
else:
    raise Exception(f"VECTORIZER {VECTORIZER} or METHOD {METHOD} not implemented")




## load relevant packages
import pandas as pd
import numpy as np
import os
import torch
import datasets
import tqdm



from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments

# ## Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import compute_metrics_standard, clean_memory, compute_metrics_nli_binary
#from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset


##### load dataset
df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/data-clean/df_pimpo_samp_trans_{MT_MODEL}_embed_tfidf.zip", engine='python')


### inspect data
## inspect label distributions
# language
inspection_lang_dic = {}
for lang in df.language_iso.unique():
    inspection_lang_dic.update({lang: df[df.language_iso == lang].label_text.value_counts()})
df_inspection_lang = pd.DataFrame(inspection_lang_dic)
# party family
inspection_parfam_dic = {}
for parfam in df.parfam_text.unique():
    inspection_parfam_dic.update({parfam: df[df.parfam_text == parfam].label_text.value_counts()})
df_inspection_parfam = pd.DataFrame(inspection_parfam_dic)


### select training data

# choose bigger text window to improve performance and imitate annotation input
if VECTORIZER == "multi":
    if METHOD == "standard_dl":
        df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
    elif METHOD == "classical_ml":
        df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
    elif METHOD == "nli":
        df["text_prepared"] = df["text_preceding"].fillna('') + '. The quote: "' + df["text_original"] + '". ' + df["text_following"].fillna('')
    elif METHOD == "dl_embed":
        df["text_prepared"] = df["text_concat_embed_multi"]
elif VECTORIZER == "en":
    if METHOD == "standard_dl":
        df["text_prepared"] = df["text_preceding_trans"].fillna('') + ' ' + df["text_original_trans"] + ' ' + df["text_following_trans"].fillna('')
    elif METHOD == "classical_ml":
        df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
    elif METHOD == "nli":
        df["text_prepared"] = df["text_preceding_trans"].fillna('') + '. The quote: "' + df["text_original_trans"] + '". ' + df["text_following_trans"].fillna('')
    elif METHOD == "dl_embed":
        df["text_prepared"] = df["text_trans_concat_embed_en"]
elif VECTORIZER == "tfidf":
    if METHOD == "classical_ml":
        df["text_prepared"] = df["text_preceding"].fillna('') + " " + df["text_original"] + " " + df["text_following"].fillna('')
else:
    raise Exception(f"Vectorizer {VECTORIZER} not implemented.")



# select task
if TASK == "integration":
    task_label_text = ["integration_supportive", "integration_sceptical", "integration_neutral", "no_topic"]
elif TASK == "immigration":
    task_label_text = ["immigration_supportive", "immigration_sceptical", "immigration_neutral", "no_topic"]
#df_cl = df[df.label_text.isin(immigration_label_text)]
# replace labels for other task with "no_topic"
df_cl = df.copy(deep=True)
df_cl["label_text"] = [label if label in task_label_text else "no_topic" for label in df.label_text]
print(df_cl["label_text"].value_counts())

# adapt numeric label
df_cl["label"] = pd.factorize(df_cl["label_text"], sort=True)[0]

## remove x% no_topic for faster testing
df_cl = df_cl.groupby(by="label_text", as_index=False, group_keys=False).apply(lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_GLOBAL) if x.label_text.iloc[0] == "no_topic" else x)
df_cl["label_text"].value_counts()



## select training data
# via language
#df_train = df_cl[df_cl.language_iso.isin(["en"])]
df_train = df_cl[df_cl.language_iso.isin(LANGUAGE_LST)]

# take sample for all topical labels - should not be more than SAMPLE (e.g. 500) per language to simulate realworld situation and prevent that adding some languages adds much more data than adding other languages - theoretically each coder can code n SAMPLE data
task_label_text_wo_notopic = task_label_text[:3]
df_train_samp1 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text.isin(task_label_text_wo_notopic)].sample(n=min(len(x[x.label_text.isin(task_label_text_wo_notopic)]), MAX_SAMPLE_LANG), random_state=SEED_GLOBAL))
df_train_samp2 = df_train.groupby(by="language_iso", as_index=False, group_keys=False).apply(lambda x: x[x.label_text == "no_topic"].sample(n=min(len(x[x.label_text == "no_topic"]), MAX_SAMPLE_LANG), random_state=SEED_GLOBAL))
df_train = pd.concat([df_train_samp1, df_train_samp2])
# could have also used len(df_train_samp1) for sampling df_train_samp2 instead of max_samp. Then could have avoided three lines below and possible different numbers in sampling across languages.

# for df_train reduce n no_topic data to same length as all topic data combined
df_train_topic = df_train[df_train.label_text != "no_topic"]
df_train_no_topic = df_train[df_train.label_text == "no_topic"].sample(n=min(len(df_train_topic), len(df_train[df_train.label_text == "no_topic"])), random_state=SEED_GLOBAL)
df_train = pd.concat([df_train_topic, df_train_no_topic])

print("\n", df_train.label_text.value_counts())
print(df_train.language_iso.value_counts(), "\n")

# avoid random overlap between topic and no-topic ?
# ! not sure if this adds value - maybe I even want overlap to teach it to look only at the middle sentence? Or need to check overlap by 3 text columns ?
#print("Is there overlapping text between train and test?", df_train_no_topic[df_train_no_topic.text_original.isin(df_train_topic.text_prepared)])

## create df test
# just to get accuracy figure as benchmark, less relevant for substantive use-case
df_test = df_cl[~df_cl.index.isin(df_train.index)]
assert len(df_train) + len(df_test) == len(df_cl)

# ! sample for faster testing
#df_test = df_test.sample(n=min(SAMPLE_DF_TEST, len(df_test)), random_state=SEED_GLOBAL)



if METHOD in ["standard_dl", "dl_embed", "classical_ml"]:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
#elif METHOD == "nli":
#    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=42)
#    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)




##### train classifier

## ! do hp-search in separate script. should not take too long
## hyperparameters for final tests
# selective load one decent set of hps for testing
#hp_study_dic = joblib.load("/Users/moritzlaurer/Dropbox/PhD/Papers/nli/snellius/NLI-experiments/results/manifesto-8/optuna_study_SVM_tfidf_01000samp_20221006.pkl")

# select best hp based on hp-search
n_sample_string = MAX_SAMPLE_LANG
#n_sample_string = 300
while len(str(n_sample_string)) <= 4:
    n_sample_string = "0" + str(n_sample_string)

import joblib
# old optuna 2.10 study objects don't work anymore with optuna 3.x
#hp_study_dic = joblib.load(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/{TRAINING_DIRECTORY}/optuna/optuna_study_{MODEL_NAME.split('/')[-1]}_{VECTORIZER}_{n_sample_string}samp_{DATASET}_{'-'.join(LANGUAGE_LST)}_{MT_MODEL}_{DATE}.pkl")
#hp_study_dic = next(iter(hp_study_dic.values()))  # unnest dic

#hyperparams = hp_study_dic['optuna_study'].best_trial.user_attrs["hyperparameters_all"]


### text pre-processing
# separate hyperparams for vectorizer and classifier.
#hyperparams_vectorizer = {key: value for key, value in hyperparams.items() if key in ["ngram_range", "max_df", "min_df", "analyzer"]}
hyperparams_vectorizer = {
    #'ngram_range': (1,2),  #trial.suggest_categorical("ngram_range", [(1, 2), (1, 3), (1, 6)]),
    #'max_df': 0.9, #trial.suggest_categorical("max_df", [1.0, 0.9, 0.8]),  # new runs for 'no-nmt-many many2many' + tfidf - others [0.95, 0.9, 0.8]  # can lead to error "ValueError: After pruning, no terms remain. Try a lower min_df or a higher max_df."
    #'min_df': 0.03,  #trial.suggest_categorical("min_df", [1, 0.01, 0.03]),  # new runs for 'no-nmt-many many2many' + tfidf - others [0.01, 0.03, 0.05]
    #'analyzer': "word",  #trial.suggest_categorical("analyzer", ["word", "char_wb"]),  # could be good for languages like Korean where longer sequences of characters without a space seem to represent compound words
}
#hyperparams_clf = {key: value for key, value in hyperparams.items() if key not in ["ngram_range", "max_df", "min_df", "analyzer"]}
hyperparams_clf = {
    #'penalty': 'l2',  # works with all solvers
    #'solver': trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
    #'C': trial.suggest_float("C", 1, 1000, log=False),
    #"class_weight": "balanced",
    "max_iter": 500,  #trial.suggest_int("max_iter", 50, 1000, log=False),  # 100 default
    #"multi_class": "auto",  # {‘auto’, ‘ovr’, ‘multinomial’}
    #"warm_start": trial.suggest_categorical("warm_start", [True, False]),
    #"n_jobs": -1,
    "random_state": SEED_GLOBAL,
}

# in case I want to add tfidf later
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_sklearn = TfidfVectorizer(lowercase=True, stop_words=None, norm="l2", use_idf=True, smooth_idf=True, **hyperparams_vectorizer)  # ngram_range=(1,2), max_df=0.9, min_df=0.02, token_pattern="(?u)\b\w\w+\b"


# choose correct pre-processed text column here
import ast
if VECTORIZER == "tfidf":
    # fit vectorizer on entire dataset - theoretically leads to some leakage on feature distribution in TFIDF (but is very fast, could be done for each test. And seems to be common practice) - OOV is actually relevant disadvantage of classical ML  #https://github.com/vanatteveldt/ecosent/blob/master/src/data-processing/19_svm_gridsearch.py
    # !! using columns here that are already bag-of-worded
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


#### prepare data for redoing figure from paper

assert (df_test["label"] == results_test["eval_label_gold_raw"]).all
df_test["label_pred"] = results_test["eval_label_predicted_raw"]
df_test_format["label_pred"] = results_test["eval_label_predicted_raw"]

df_train["label_pred"] = [np.nan] * len(df_train["label"])

df_cl_concat = pd.concat([df_train, df_test])

# add label text for predictions
label_text_map = {}
for i, row in df_cl_concat[~df_cl_concat.label_text.duplicated(keep='first')].iterrows():
    label_text_map.update({row["label"]: row["label_text"]})
df_cl_concat["label_text_pred"] = df_cl_concat["label_pred"].map(label_text_map)


## trim df to save storage
df_cl_concat = df_cl_concat[['label', 'label_text', 'country_iso', 'language_iso', 'doc_id',
                           'text_original', 'text_original_trans', 'text_preceding_trans', 'text_following_trans',
                           # 'text_preceding', 'text_following', 'selection', 'certainty_selection', 'topic', 'certainty_topic', 'direction', 'certainty_direction',
                           'rn', 'cmp_code', 'partyname', 'partyabbrev',
                           'parfam', 'parfam_text', 'date', #'language_iso_fasttext', 'language_iso_trans',
                           #'text_concat', 'text_concat_embed_multi', 'text_trans_concat',
                           #'text_trans_concat_embed_en', 'text_trans_concat_tfidf', 'text_prepared',
                           'label_pred', 'label_text_pred']]


## save data
langs_concat = "_".join(LANGUAGE_LST)
#df_cl_concat.to_csv(f"./data-classified/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv", index=False)
#df_cl_concat.to_csv(f"./data-classified/pimpo/df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.zip",
#                    compression={"method": "zip", "archive_name": f"df_pimpo_pred_{TASK}_{METHOD}_{MODEL_SIZE}_{HYPOTHESIS}_{VECTORIZER}_{MAX_SAMPLE_LANG}samp_{langs_concat}_{DATE}.csv"}, index=False)











###### Interpretability tests
'''
# feature importance
#clf._feature_importances
#clf.coef_

# vocabulary in order of feature importance scores (in order of index)
vectorizer_sklearn.vocabulary_
vocab_sorted = {k: v for k, v in sorted(vectorizer_sklearn.vocabulary_.items(), key=lambda item: item[1])}

coef_dic = {}
#for label in df_train_format.label.unique():  #df_train_format.label_text.unique():
    #coef_dic.update({f"coef_{label}": clf.coef_[label]})
# factorize provides label_text in same order as numerical labels
for i, label_text in enumerate(pd.factorize(df_cl["label_text"], sort=True)[1]):
    coef_dic.update({f"coef_{label_text}": clf.coef_[i]})

df_feat_importance = pd.DataFrame({"token": vocab_sorted.keys(), **coef_dic})
df_feat_importance = df_feat_importance.sort_values(by=list(coef_dic.keys())[0], ascending=False)


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

### eli5 tests
"""
# only works in ipython notebook
eli5.show_weights(clf)
eli5.show_prediction(clf)

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
label_label_text_map = {i: text for i, text in enumerate(pd.factorize(df_cl["label_text"], sort=True)[1])}
df_eli["target_text"] = df_eli["target"].map(label_label_text_map)

# "weight" displays the importance of respective token for respective class
df_eli = df_eli.sort_values(["target", "weight"], ascending=False)
"""


### key features for classification decisions by meta-data groups

# ! do analysis only on wrongly classified texts
df_test_format_wrong = df_test_format[df_test_format.label != df_test_format.label_pred]

#n_iter = 0
df_feat_importance_group_lst_lst = []
for key_group, df_group in tqdm.tqdm(df_test_format_wrong.reset_index(drop=True).groupby(by=["parfam_text"])):  # country_iso, parfam_text
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
label_label_text_map = {i: text for i, text in enumerate(pd.factorize(df_cl["label_text"], sort=True)[1])}
df_feat_importance_group_all["label_text"] = df_feat_importance_group_all["target"].map(label_label_text_map)
# sort
df_feat_importance_group_all = df_feat_importance_group_all.sort_values(["group", "text_id", "weight"], ascending=False)

## create aggregate statistics by group
df_feat_importance_group_all_aggreg = df_feat_importance_group_all.groupby(by=["group", "label_text", "feature_text"], as_index=False, group_keys=False).apply(lambda x: x.weight.mean())
df_feat_importance_group_all_aggreg = df_feat_importance_group_all_aggreg.rename(columns={None: "weight"})
df_feat_importance_group_all_aggreg = df_feat_importance_group_all_aggreg.groupby(by=["group", "label_text"]).apply(lambda x: x.nlargest(5, "weight")).reset_index(drop=True)
df_feat_importance_group_all_aggreg = df_feat_importance_group_all_aggreg.sort_values(["label_text", "group", "weight"], ascending=False)

### intermediate conclusion
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

'''

print("\n\nScript done.")





