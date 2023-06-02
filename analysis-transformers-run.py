

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
from helpers import (
    compute_metrics_standard, clean_memory, compute_metrics_nli_binary, compute_metrics_generation,
    load_model_tokenizer, tokenize_datasets, set_train_args, create_trainer, format_nli_trainset, format_nli_testset
)

# load latest version of ActiveLLM
sys.path.insert(0, "/Users/moritzlaurer/Dropbox/PhD/open-source/ActiveLLM")
from active_learner import ActiveLearner
# reload in case of updates in active_learner.py
import importlib
import active_learner
importlib.reload(active_learner)
from active_learner import ActiveLearner

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
parser.add_argument('-iter', '--n_run', type=int, #nargs='+',
                    help='The number of the respective random iteration')
parser.add_argument('-iter_max', '--n_random_runs_total', type=int, #nargs='+',
                    help='The total number of random iterations')
parser.add_argument('-g', '--group', type=str,
                    help='group to filter training data by')
parser.add_argument('-n_tok_rm', '--n_tokens_remove', type=int, #nargs='+',
                    help='number of group-specific tokens to remove from test data')
parser.add_argument('-save', '--save_outputs', action="store_true",
                    help='boolean whether to save outputs to disk')

parser.add_argument('-max_l', '--max_length', type=int, #nargs='+',
                    help='max n tokens')

parser.add_argument('-al_iter', '--active_learning_iterations', type=int, #nargs='+',
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
                            "--model", "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",  #"google/flan-t5-small",
                            "--method", "nli",  #"generation",
                            "--sample_size", "50", "--study_date", "20230601",
                            "--n_run", "1", "--n_random_runs_total", "3",
                            "--group", "randomall", "--n_tokens_remove", "0", "--max_length", "256",
                            "--active_learning_iterations", "2",
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
AL_ITERATIONS = args.active_learning_iterations
if AL_ITERATIONS > 0:
    print("AL_ITERATIONS: ", AL_ITERATIONS)
    N_SAMPLES_PER_AL_ITER = int(MAX_SAMPLE/AL_ITERATIONS)
    print(f"For sample size {MAX_SAMPLE}, this means {N_SAMPLES_PER_AL_ITER} samples per iteration.")
else:
    print("No active learning.")
    N_SAMPLES_PER_AL_ITER = None


# set global seed for reproducibility and against seed hacking
N_RUN = args.n_run - 1
N_RANDOM_RUNS_TOTAL = args.n_random_runs_total
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)

# special variables for pimpo
SAMPLE_NO_TOPIC = 100  # for number in test set
TRAIN_NOTOPIC_PROPORTION = 0.4

# randomly assign different seeds for each run
seed_runs_all = np.random.choice(range(1000), size=N_RANDOM_RUNS_TOTAL)
SEED_RUN = seed_runs_all[N_RUN]
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
elif ("t5" in MODEL_NAME.lower()) and ("nli" in METHOD.lower() or "standard_dl" in METHOD.lower()):
    raise Exception(f"MODEL_NAME {MODEL_NAME} not implemented for METHOD {METHOD}")



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
        df_cl["text_prepared"] = df_cl["text_preceding_trans"].fillna('') + '  | The quote: "' + df_cl["text_original_trans"] + '" End of the quote |  ' + df_cl["text_following_trans"].fillna('')
    elif METHOD == "standard_dl":
        df_cl["text_prepared"] = df_cl["text_preceding_trans"].fillna('') + ' \n ' + df_cl["text_original_trans"] + ' \n ' + df_cl["text_following_trans"].fillna('')
    elif METHOD == "generation":
        df_cl["text_prepared"] = df_cl["text_preceding_trans"].fillna('') + '  | The quote: "' + df_cl["text_original_trans"] + '" End of the quote |  ' + df_cl["text_following_trans"].fillna('')
else:
    raise Exception(f"Vectorizer {VECTORIZER} or METHOD {METHOD} not implemented.")


## data checks
#print("Dataset: ", DATASET, "\n")
# verify that numeric label is in alphabetical order of label_text (can avoid issues for NLI)
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").label.tolist()  # label num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)



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


import random
random.seed(SEED_RUN)
if "random3" in GROUP:
    # TODO: implement for more than countries and make more general
    GROUP_join = random.sample(list(df_cl.country_iso.unique()), 3)
    GROUP_join = '|'.join(GROUP_join)
    print(GROUP_join)
elif "random2" in GROUP:
    group_enough_data = ["nld", "esp", "deu", "dnk", "aut"]
    GROUP_join = random.sample(group_enough_data, 2)
    GROUP_join = '|'.join(GROUP_join)
    print(GROUP_join)
# TODO: use this in bash script for single groups. leads to clearer file naming and less total runs for single group-member runs
elif "random1" in GROUP:
    GROUP_join = random.sample(list(df_cl.country_iso.unique()), 1)[0]
    print(GROUP_join)
elif "randomall" in GROUP:
    pass
else:
    raise NotImplementedError


#print(df_cl.groupby("country_iso").apply(lambda x: x.label_text.value_counts()))

# sample training data
if "uk-leftright" in DATASET:
    df_train = df_cl.sample(n=MAX_SAMPLE, random_state=SEED_RUN)
elif "pimpo" in DATASET:
    # simplified
    if "randomall" in GROUP:
        df_cl_group = df_cl.copy(deep=True)
    elif "random" in GROUP:
        # TODO: if beyond countries, double check if group_join regex really only matches single group-member per group-member string. works for countries, maybe not for other groups if one member is sub-string of other member
        df_cl_group = df_cl[df_cl.country_iso.str.contains(GROUP_join)].copy(deep=True)
    else:
        raise NotImplementedError

    # unrealistic balanced sample without AL
    if AL_ITERATIONS == 0:
        # sample x% of training data for no topic, then share the remainder equally across classes
        n_sample_notopic = int(MAX_SAMPLE * TRAIN_NOTOPIC_PROPORTION)
        n_sample_perclass = int((MAX_SAMPLE - n_sample_notopic) / (len(df_cl.label_text.unique()) - 1))
        df_train_samp1 = df_cl_group.groupby("label_text", as_index=False, group_keys=False).apply(
            lambda x: x.sample(min(n_sample_perclass, len(x)), random_state=SEED_RUN) if x.label_text.unique()[0] != "no_topic" else None)
        df_train_samp2 = df_cl_group[df_cl_group.label_text == "no_topic"].sample(n_sample_notopic, random_state=SEED_RUN)
        df_train = pd.concat([df_train_samp1, df_train_samp2])

        print("df_train.label_text.value_counts:\n", df_train.label_text.value_counts())

    elif AL_ITERATIONS > 0:
        if METHOD == "nli":
            df_corpus = df_cl_group.copy(deep=True)
            df_train_seed = None
        elif METHOD == "standard_dl":
            # need to implement sampling random df_train_seed as well as corpus without overlap
            raise NotImplementedError()

        print("df_corpus.label_text.value_counts:\n", df_corpus.label_text.value_counts())


# create df test
# TODO make work beyond countries
if "randomall" in GROUP:
    if AL_ITERATIONS > 0:
        df_test = None
    else:
        df_test = df_cl[~df_cl.index.isin(df_train.index)]
elif "random" in GROUP:
    # if df_train comes from specific group, remove this group from df_test
    if AL_ITERATIONS > 0:
        df_test = df_cl[~df_cl.index.isin(df_corpus.index)]
        if METHOD == "standard_dl":
            df_test = df_test[~df_test.index.isin(df_train_seed.index)]
        df_test = df_test[~df_test.country_iso.str.contains(GROUP_join)].copy(deep=True)
    else:
        df_test = df_cl[~df_cl.index.isin(df_train.index)]
        df_test = df_test[~df_test.country_iso.str.contains(GROUP_join)].copy(deep=True)


# remove N no_topic & downsample for faster testing
if "pimpo" in DATASET:
    if AL_ITERATIONS > 0:
        # reduce no-topic to N
        #df_corpus = df_corpus.groupby(by="label_text", as_index=False, group_keys=False).apply(
        #    lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_RUN) if x.label_text.iloc[0] == "no_topic" else x)
        # reduce entire test-set to N
        df_corpus = df_corpus.sample(n=min(SAMPLE_NO_TOPIC, len(df_corpus)), random_state=SEED_RUN)
        print("df_corpus.label_text.value_counts:\n", df_corpus.label_text.value_counts())

        if ("random" in GROUP) and ("randomall" not in GROUP):
            # reduce no-topic to N
            df_test = df_test.groupby(by="label_text", as_index=False, group_keys=False).apply(
                lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_RUN) if x.label_text.iloc[0] == "no_topic" else x)
            # reduce entire test-set to N
            df_test = df_test.sample(n=min(SAMPLE_NO_TOPIC, len(df_test)), random_state=SEED_RUN)
            print("df_test.label_text.value_counts:\n", df_test.label_text.value_counts())

    else:
        # reduce no-topic to N
        df_test = df_test.groupby(by="label_text", as_index=False, group_keys=False).apply(
            lambda x: x.sample(n=min(SAMPLE_NO_TOPIC, len(x)), random_state=SEED_RUN) if x.label_text.iloc[0] == "no_topic" else x)
        # reduce entire test-set to N
        df_test = df_test.sample(n=min(SAMPLE_NO_TOPIC, len(df_test)), random_state=SEED_RUN)
        print("df_test.label_text.value_counts:\n", df_test.label_text.value_counts())



### format data if NLI or generative

## NLI instructions
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
# TODO: the hypotheses need to be in alphabetical order. double check that this is the case everywhere
if "nli" in METHOD:
    # make keys in hypo_label_dic alphabetical
    hypo_label_dic = {k: hypo_label_dic[k] for k in sorted(hypo_label_dic.keys())}

## generative instructions
# TODO: need to somehow make sure that these instructions never get cut, but only the input text
if METHOD == "generation":
    instruction_short = """\n
Which of the following categories applies best to the quote considering the context above?
A: The quote is neutral towards immigration/integration or describes the status quo of immigration/integration.
B: The quote is sceptical of immigration/integration.
C: The quote is supportive of immigration/integration.
D: The quote is not about immigration/integration.
Answer: """
    label_text_map_generation = {"neutral": "A", "sceptical": "B", "supportive": "C", "no_topic": "D"}

## parameters relevant for generative models
if METHOD == "generation":
    model_params = {
        #"torch_dtype": torch.float16,  #torch.bfloat16, torch.float16
        #load_in_8bit=True,
        "device_map": "auto",
        "offload_folder": "offload",
        "offload_state_dict": True
    }
    config_params = {
        "max_new_tokens": 4,
        "num_beams": 2,
        #"generation_num_beams": 5,  # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/training_args_seq2seq.py#L42
        "num_return_sequences": 1,
        "temperature": 0,  # default: 1.0
        "top_k": 50,  # default: 50
        "return_dict_in_generate": True,
        "output_scores": True,
        #"predict_with_generate": False,
        #"include_inputs_for_metrics": True
        "renormalize_logits": "True",
    }
else:
    model_params = None
    config_params = None

## automatically calculate roughly adequate epochs for number of data points
if METHOD == "standard_dl":
    max_steps = 7_000  # value chosen to lead to roughly 45 epochs with 5k n_data, 23 with 10k, then decrease epochs
    batch_size = 32
    #min_epochs = 10
    max_epochs = 30   #50 good value from NLI paper experience for around 500 - 5k data
    n_data = len(df_corpus)  #len(df_train_format)
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
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 3, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 32, 'warmup_ratio': 0.40, 'weight_decay': 0.01, 'per_device_eval_batch_size': 200}  # "do_eval": False
elif "generation" in METHOD:
    HYPER_PARAMS = {
        'lr_scheduler_type': 'linear', 'learning_rate': 5e-4, 'num_train_epochs': 3, 'seed': SEED_GLOBAL, 'per_device_train_batch_size': 16, 'warmup_ratio': 0.20, 'weight_decay': 0.01, 'per_device_eval_batch_size': 64-32,
        # ! need to set this to true, otherwise seq2seq-trainer is not used https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L246
        "predict_with_generate": True, "gradient_checkpointing": False, #"gradient_accumulation_steps": 8,
    }
else:
    raise Exception("Method not implemented for hps")




learner = ActiveLearner(seed=SEED_RUN)

learner.load_pd_dataset(df_corpus=df_corpus, df_test=df_test, df_train_seed=df_train_seed, text_column="text_prepared", label_column="label_text", separate_testset=False)

if "nli" in METHOD:
    learner.format_pd_dataset_for_nli_test(hypo_label_dic=hypo_label_dic)

# only for first run
learner.load_model_tokenizer(
    model_name=MODEL_NAME, method=METHOD, model_max_length=MODEL_MAX_LENGTH,
    config_params=config_params, model_params=model_params,
    label_text_alphabetical=np.sort(df_cl.label_text.unique())
)

learner.tokenize_dataset(max_output_tokens=0)  # config_params["max_new_tokens"]

learner.set_train_args(hyperparams_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY, disable_tqdm=False, evaluation_strategy="no")

## first zero-shot sampling & testing run, no training
# for first sample for first training run
# metrics currently come from texts in df_corpus that were not yet sampled
if METHOD == "nli":
    learner.train_test_infer()

# apply sampling strategy
#learner.min_certainty(n_sample_al=N_SAMPLE_AL)
if METHOD == "nli":
    learner.sample_breaking_ties(n_sample_al=N_SAMPLES_PER_AL_ITER)

    print(learner.df_corpus_al_sample.head())


### active learning loop
n_iter = 0

while n_iter < AL_ITERATIONS:
    ## this is where manualo annotation would happen e.g. in Argilla
    # dataset_train in update below needs to ingest the manual annotations created here (or oracle labels)
    #label_annotation = learner.df_corpus_al_sample["label"]

    if (METHOD == "standard_classifier") & (n_iter == 0):
        pass
    else:
        # visual inspection: compare predicted labels with gold labels
        prediction_gold_lst = [[prediction, gold] for prediction, gold in zip(learner.iteration_label_predicted_test, learner.iteration_label_gold_test)]
        print(prediction_gold_lst[:20])

        # with first dataset update, a dataset_train is added
        # with all updates, dataset_train and dataset_corpus is updated
        learner.update_dataset()

    # training run
    learner.train_test_infer()

    # print results
    for key_iter, value_metrics_dic in learner.metrics.items():
        print(f"Aggregate metrics for {key_iter}: ", {key: value_metrics_dic[key] for key in value_metrics_dic if key not in ["label_gold_raw", "label_predicted_raw"]})  # print metrics but without label lists

    # new sampling run before updating the dataset. need new index_al_sample
    learner.sample_breaking_ties(n_sample_al=N_SAMPLES_PER_AL_ITER)
    #learner.min_certainty(n_sample_al=N_SAMPLE_AL)

    print(f"\n\n    Iteration {n_iter} finished.\n\n")
    n_iter += 1



### Evaluate
# test on test set
if METHOD != "generation":
    results_test = learner.metrics#[f"iter_{AL_ITERATIONS}"] #trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
else:
    # ! will not contain certain elements like inference speed, loss
    results_test = learner.metrics#[f"iter_{AL_ITERATIONS}"]  #compute_metrics_generation(dataset=dataset, model=trainer.model, tokenizer=tokenizer, hyperparams_dic=HYPER_PARAMS, config_params=config_params)

print("\nTest results:")
print(results_test)

## save results
n_sample_str = MAX_SAMPLE
while len(str(n_sample_str)) <= 3:
    n_sample_str = "0" + str(n_sample_str)

# df with separate row for each al iteration
df_results = pd.DataFrame.from_dict(results_test, orient='index')
print(df_results.head())

if SAVE_OUTPUTS:
    #df_results.to_csv(f"./data-classified/{DATASET}/df_results_{DATASET}_{GROUP}_samp{n_sample_str}_tokrm{N_TOKENS_REMOVE}_seed{SEED_RUN}_{DATE}.csv", index=False)
    df_results.to_csv(f"./results/{DATASET}/df_results_{DATASET}_{GROUP}_{METHOD}_samp{n_sample_str}_seed{SEED_RUN}_{DATE}.csv", index=False)





print("\nScript done.\n\n")
assert 1 == 2, ("Script done.")
#sys.exit(0)





# format data
if METHOD in ["standard_dl", "dl_embed", "classical_ml"]:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
elif "nli" in METHOD:
    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=SEED_RUN)
    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)
elif "generation" in METHOD:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
    # adapt input text
    df_train_format["text_prepared"] = df_train_format["text_prepared"] + instruction_short
    df_test_format["text_prepared"] = df_test_format["text_prepared"] + instruction_short
    # adapt label
    df_train_format["label_text_original"] = df_train_format["label_text"]
    df_test_format["label_text_original"] = df_test_format["label_text"]
    df_train_format["label_text"] = df_train_format["label_text"].map(label_text_map_generation)
    df_test_format["label_text"] = df_test_format["label_text"].map(label_text_map_generation)




##### train classifier
label_text_alphabetical = np.sort(df_cl.label_text.unique())

model, tokenizer = load_model_tokenizer(model_name=MODEL_NAME, method=METHOD,
                                        label_text_alphabetical=label_text_alphabetical, model_max_length=MODEL_MAX_LENGTH,
                                        model_params=model_params, config_params=config_params)


#### tokenize

dataset = tokenize_datasets(df_train_samp=df_train_format, df_test=df_test_format, tokenizer=tokenizer, method=METHOD,
                            max_length=MODEL_MAX_LENGTH, config_params=config_params)


### create trainer

# based on paper https://arxiv.org/pdf/2111.09543.pdf
#if MODEL_SIZE == "large":
#    HYPER_PARAMS.update({"per_device_eval_batch_size": 80, 'learning_rate': 9e-6})


## create trainer
# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet
# TODO: can fp16 cause issues with generative models?
fp16 = False

train_args = set_train_args(hyperparams_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY, method=METHOD,
                            disable_tqdm=False, evaluation_strategy="no", fp16=fp16_bool)

trainer = create_trainer(model=model, tokenizer=tokenizer, encoded_dataset=dataset, train_args=train_args,
                         method=METHOD, label_text_alphabetical=label_text_alphabetical)

# train
trainer.train()



### Evaluate
# test on test set
if METHOD != "generation":
    results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
else:
    # ! will not contain certain elements like inference speed, loss
    results_test = compute_metrics_generation(dataset=dataset, model=trainer.model, tokenizer=tokenizer, hyperparams_dic=HYPER_PARAMS, config_params=config_params)

print("\nTest results:")
print(results_test)

## save results
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





