

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

# Load helper functions
import sys
sys.path.insert(0, os.getcwd())
import helpers
import importlib  # in case of manual updates in .py file
importlib.reload(helpers)
from helpers import compute_metrics_standard, clean_memory, compute_metrics_nli_binary
from helpers import load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset

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
    args = parser.parse_args(["--task", "pimpo-simple",  # pimpo, uk-leftright-simple, uk-leftright
                            "--dataset", "pimpo",  # uk-leftright-econ, pimpo
                            "--vectorizer", "transformer",
                            "--model", "transformer",
                            "--method", "standard_dl",
                            "--sample_size", "8", "--study_date", "20230427",
                            "--n_iteration", "1", "--n_iterations_max", "5",
                            "--group", "deu", "--n_tokens_remove", "0", "--max_length", "8",
                            #"--save_outputs"
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

MODEL_MAX_LENGTH = args.max_length

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


## TODO: expand for T5
if MODEL_NAME == "transformer":
    MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
else:
    raise Exception(f"MODEL_NAME {MODEL_NAME} not implemented")




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
    elif METHOD == "standard_dl":
        df_cl["text_prepared"] = df_cl["text_preceding_trans"].fillna('') + ' \n ' + df_cl["text_original_trans"] + ' \n ' + df_cl["text_following_trans"].fillna('')
else:
    raise Exception(f"Vectorizer {VECTORIZER} or METHOD {METHOD} not implemented.")


## data checks
#print("Dataset: ", DATASET, "\n")
# verify that numeric label is in alphabetical order of label_text (can avoid issues for NLI)
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)


## lemmatize prepared text
"""
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
"""


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

    # ! for testing, remove remaining data from group from df_cl
    #df_cl_group_rest = df_cl_group[~df_cl_group.index.isin(df_train.index)]
    #df_cl = df_cl[~df_cl.index.isin(df_cl_group_rest.index)]

print("df_train.label_text.value_counts:\n", df_train.label_text.value_counts())

# create df test
df_test = df_cl[~df_cl.index.isin(df_train.index)]
# also remove all GROUP from df_test?
# TODO: remove this if. only added for backwards compatibility, because had not excluded own group in 500 samp originally (I think)
if MAX_SAMPLE != 500:
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


### format data if NLI

if METHOD == "nli":
    hypo_label_dic = {
        "neutral": "The quote is neutral towards immigration/integration or describes the status quo of immigration/integration.",
        "sceptical": "The quote is sceptical of immigration/integration.",
        "supportive": "The quote is supportive of immigration/integration.",
        "no_topic": "The quote is not about immigration/integration.",
    }
elif METHOD == "nli_long":
    hypo_label_dic = {
        "neutral": "The quote is neutral towards immigration/integration or describes the status quo of immigration/integration, for example only stating facts or using technocratic language about immigration/integration",
        "sceptical": "The quote is sceptical of immigration/integration. Regarding immigration, the quote could mention the costs of immigration, be against migrant workers, state that foreign labour decreases natives' wages, that there are already enough refugees, refugees are actually economic migrants, be in favour of stricter immigration controls, or for exceptions to the freedom of movement in the EU. Regarding integration, the quote could make negative references to multiculturalism and diversity, underline the importance of ethnic homogeneity and national culture, call for immigrants to give up their culture of origin, warn of islamization, mention duties in order to stay in the country, demand integration tests, associate immigrant communities with problems or crimes, demand an oath of allegiance of immigrants, or underline ethnic criteria for receiving citizenship.",
        "supportive": "The quote is supportive of immigration/integration. Regarding immigration, the quote could mention the benefits of immigration, the need for migrant workers, international obligations to take in refugees, protection of human rights, in favour of family reunification or freedom of movement in the EU. Regarding integration, the quote could mention positive references to multiculturalism and diversity, underline cosmopolitan values towards immigrants, demand inclusion of immigrants, demand anti-discrimination policies based on ethnicity and origin, demand policies against racism, demand more rights for immigrants, or underline civic values instead of ethnic values for receiving citizenship.",
        "no_topic": "The quote is not about immigration/integration.",
    }
elif METHOD == "nli_void":
    hypo_label_dic = {
        "neutral": "The quote is about category A.",
        "sceptical": "The quote is about category B.",
        "supportive": "The quote is about category C.",
        "no_topic": "The quote is about category D.",
    }

"""HYPOTHESIS = "short"
if TASK == "immigration":
    if HYPOTHESIS == "short":
        hypo_label_dic = {
            "immigration_neutral": "The quote is neutral towards immigration or describes the status quo of immigration.",
            "immigration_sceptical": "The quote is sceptical of immigration.",
            "immigration_supportive": "The quote is supportive of immigration.",
            "no_topic": "The quote is not about immigration.",
        }
    elif HYPOTHESIS == "long":
        hypo_label_dic = {
            "immigration_neutral": "The quote describes immigration neutrally without implied value judgement or describes the status quo of immigration, for example only stating facts or using technocratic language about immigration",
            "immigration_sceptical": "The quote describes immigration sceptically / disapprovingly. For example, the quote could mention the costs of immigration, be against migrant workers, state that foreign labour decreases natives' wages, that there are already enough refugees, refugees are actually economic migrants, be in favour of stricter immigration controls, exceptions to the freedom of movement in the EU.",
            "immigration_supportive": "The quote describes immigration favourably / supportively. For example, the quote could mention the benefits of immigration, the need for migrant workers, international obligations to take in refugees, protection of human rights, in favour of family reunification or freedom of movement in the EU.",
            "no_topic": "The quote is not about immigration.",
        }
    else:
        raise Exception(f"Hypothesis {HYPOTHESIS} not implemented")
elif TASK == "integration":
    if HYPOTHESIS == "short":
        hypo_label_dic = {
            "integration_neutral": "The quote is neutral towards immigrant integration or describes the status quo of immigrant integration.",
            "integration_sceptical": "The quote is sceptical of immigrant integration.",
            "integration_supportive": "The quote is supportive of immigrant integration.",
            "no_topic": "The quote is not about immigrant integration.",
        }
    elif HYPOTHESIS == "long":
        hypo_label_dic = {
            "integration_neutral": "The quote describes immigrant integration neutrally or describes the status quo of immigrant integration, for example only stating facts or using technocratic language about immigrant integration",
            "integration_sceptical": "The quote describes immigrant integration sceptically / disapprovingly. For example, the quote could mention negative references to multiculturalism and diversity, underline the importance of ethnic homogeneity and national culture, call for immigrants to give up their culture of origin, warn of islamization, mention duties in order to stay in the country, demand integration tests, associate immigrant communities with problems or crimes, demand an oath of allegiance of immigrants, or underline ethnic criteria for receiving citizenship.",
            "integration_supportive": "The quote describes immigrant integration favourably / supportively. For example, the quote could mention positive references to multiculturalism and diversity, underline cosmopolitan values towards immigrants, demand inclusion of immigrants, demand anti-discrimination policies based on ethnicity and origin, demand policies against racism, demand more rights for immigrants, or underline civic values instead of ethnic values for being able to receive citizenship.",
            "no_topic": "The quote is not about immigrant integration.",
        }
    else:
        raise Exception(f"Hypothesis {HYPOTHESIS} not implemented")
else:
    raise Exception(f"Task {TASK} not implemented")"""



if METHOD in ["standard_dl", "dl_embed", "classical_ml"]:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
elif "nli" in METHOD:
    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=SEED_RUN)
    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)



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
label_text_alphabetical = np.sort(df_cl.label_text.unique())

model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD, label_text_alphabetical=label_text_alphabetical, model_max_length=MODEL_MAX_LENGTH)


#### tokenize

dataset = tokenize_datasets(df_train_samp=df_train_format, df_test=df_test_format, tokenizer=tokenizer, method=METHOD, max_length=MODEL_MAX_LENGTH)


### create trainer

## automatically calculate roughly adequate epochs for number of data points
if METHOD == "standard_dl":
    max_steps = 7_000  # value chosen to lead to roughly 45 epochs with 5k n_data, 23 with 10k, then decrease epochs
    batch_size = 32
    #min_epochs = 10
    max_epochs = 30   #50 good value from NLI paper experience for around 500 - 5k data
    n_data = len(df_train_format)
    steps_one_epoch = n_data / batch_size
    n_epochs = 0
    n_steps = 0
    while (n_epochs < max_epochs) and (n_steps < max_steps):
        n_epochs += 1
        n_steps += steps_one_epoch  # = steps_one_epoch
    print("Epochs: ", n_epochs)
    print("Steps: ", n_steps)
    HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': n_epochs, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.06, 'weight_decay': 0.01, 'per_device_eval_batch_size': 200}  # "do_eval": False
elif METHOD == "nli_void":
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 15, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.40, 'weight_decay': 0.01, 'per_device_eval_batch_size': 200}  # "do_eval": False
elif "nli" in METHOD:
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 5, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.40, 'weight_decay': 0.01, 'per_device_eval_batch_size': 200}  # "do_eval": False
else:
    raise Exception("Method not implemented for hps")

# based on paper https://arxiv.org/pdf/2111.09543.pdf
#if MODEL_SIZE == "large":
#    HYPER_PARAMS.update({"per_device_eval_batch_size": 80, 'learning_rate': 9e-6})


## create trainer
# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet

train_args = set_train_args(hyperparams_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY, disable_tqdm=False, evaluation_strategy="no", fp16=fp16_bool)

trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=dataset, train_args=train_args,
                         method=METHOD, label_text_alphabetical=label_text_alphabetical)

# train
trainer.train()



### Evaluate
# test on test set
results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
print("\nTest results:")
print(results_test)

# save results
n_sample_str = MAX_SAMPLE
while len(str(n_sample_str)) <= 3:
    n_sample_str = "0" + str(n_sample_str)

df_results = pd.DataFrame([results_test])
if SAVE_OUTPUTS:
    #df_results.to_csv(f"./data-classified/{DATASET}/df_results_{DATASET}_{GROUP}_samp{n_sample_str}_tokrm{N_TOKENS_REMOVE}_seed{SEED_RUN}_{DATE}.csv", index=False)
    df_results.to_csv(f"./results/{DATASET}/df_results_{DATASET}_{GROUP}_{METHOD}_samp{n_sample_str}_seed{SEED_RUN}_{DATE}.csv", index=False)





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





print("\nScript done.\n\n")





