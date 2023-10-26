# This script implements one training run with a Transformer

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
import pickle
import gzip
import random

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
        "--vectorizer", "transformer",
        "--model", "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",  #"google/flan-t5-small",  #"google/electra-small-discriminator",  #"MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary",  #"google/flan-t5-small",
        "--method", "nli_short",  #"nli_short",  #"generation",
        "--sample_size_train", "500", "--sample_size_no_topic", "5000",
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
# following advice here to reduce randomness risks: https://pytorch.org/docs/stable/notes/randomness
N_RUN = args.n_run - 1
N_RANDOM_RUNS_TOTAL = args.n_random_runs_total
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)

# special variables for pimpo
SAMPLE_NO_TOPIC = args.sample_size_no_topic  # for number in test set
if "pimpo" in DATASET:
    TRAIN_NOTOPIC_PROPORTION_TRAIN = 0.6  # needs to be at least 0.58 to enable balanced sample for more party families with 500 total sample
else:
    TRAIN_NOTOPIC_PROPORTION_TRAIN = 0
#SAMPLE_SIZE_CORPUS = args.sample_size_corpus
SAMPLE_SIZE_TEST = args.sample_size_test

# randomly assign different seeds for each run
seed_runs_all = np.random.choice(range(10000), size=N_RANDOM_RUNS_TOTAL)
SEED_RUN = int(seed_runs_all[N_RUN])
print("Iteration number: ", N_RUN)
print("All random seeds: ", seed_runs_all)
print("Random seed for this run: ", SEED_RUN)

# Note: task variable currently not used
assert DATASET.split("-")[0] in TASK, f"Mismatch between dataset {DATASET} and task {TASK}"

# shorten model_name, can use in file names later
MODEL_NAME_SHORT = MODEL_NAME.split("/")[-1]
MODEL_NAME_SHORT = MODEL_NAME_SHORT[:26]  # longer names lead to file name length bugs before
print(MODEL_NAME_SHORT)

# testing that model and method combination makes sense
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
    if (GROUP_COL != "domain") and (GROUP_COL != "randomall"):
        raise Exception(f"Group column {GROUP_COL} for dataset {DATASET} should always be 'domain'.")
elif "cap-sotu" in DATASET:
    df = pd.read_csv("./data-clean/df_cap_sotu_train.zip", engine="python")
    df_test = pd.read_csv("./data-clean/df_cap_sotu_test.zip", engine="python")
    df_cl = df.copy(deep=True)
else:
    raise Exception(f"Dataset name not found: {DATASET}")


### preprocessing data

## prepare input text data
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
    df_cap_court = df_cl[df_cl.domain == "legal"]
    df_cap_court_test = df_test[df_test.domain == "legal"]
    if any(substring in METHOD for substring in ["nli", "disc"]) or (METHOD == "generation"):
        df_cl.loc[df_cl.domain == "speech", "text_prepared"] = 'The quote: "' + df_cap_sotu["text_original"] + '"'
        df_test.loc[df_test.domain == "speech", "text_prepared"] = 'The quote: "' + df_cap_sotu_test["text_original"] + '"'
        df_cl.loc[df_cl.domain == "legal", "text_prepared"] = 'The quote: "' + df_cap_court["text_prepared"] + '"'
        df_test.loc[df_test.domain == "legal", "text_prepared"] = 'The quote: "' + df_cap_court_test["text_prepared"] + '"'
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
print("Dataset: ", DATASET, "\n")
# verify that numeric labels is in alphabetical order of label_text (can avoid issues for NLI)
labels_num_via_numeric = df_cl[~df_cl.label_text.duplicated(keep="first")].sort_values("label_text").labels.tolist()  # labels num via labels: get labels from data when ordering label text alphabetically
labels_num_via_text = pd.factorize(np.sort(df_cl.label_text.unique()))[0]  # label num via label_text: create label numeric via label text
assert all(labels_num_via_numeric == labels_num_via_text)




### select training data and groups
random.seed(SEED_RUN)

def select_group_members_randomly(df=None, group_col=None, n_members_str=None, seed=None):
    random.seed(int(seed))
    n_members = int(n_members_str[-1])
    group_join = random.sample(list(df[group_col].unique().astype(str)), n_members)
    print(f"Group selected: {group_join}  for seed {seed}")
    group_join = r'\b' + r'\b|\b'.join(group_join) + r'\b'
    #group_join = f"^({group_join})$"  # to only look at exact matches
    return group_join, seed+42


if GROUP_COL != "randomall":
    label_distribution_per_group_member = df_cl.groupby(GROUP_COL).apply(lambda x: x.label_text.value_counts())
    print("Overall label distribution per group member:\n", label_distribution_per_group_member)
else:
    print("Overall label distribution per group member:\n No group used. Sampling with randomall.")


# sample balanced training data
# redo sampling for different groups until get a fully balanced sample for several iterations
# balanced samples are important to remove data imbalanced as intervening variable for performance differences
imbalanced_sample = True
counter = 0
seed_run_update = SEED_RUN if GROUP_COL != "continent" else SEED_RUN+42  # otherwise only oversamples asia and europe
while imbalanced_sample and (counter <= 50):
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
    if "pimpo" in DATASET:
        n_classes_for_sample = len(df_cl.label_text.unique()) - 1
    else:
        n_classes_for_sample = len(df_cl.label_text.unique())
    n_sample_perclass = int((SAMPLE_SIZE_TRAIN - n_sample_notopic) / n_classes_for_sample)
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

if counter == 51:
    raise ValueError("could not sample balanced training data after 50 iterations")
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




##### code specific to Transformers
if METHOD == "classical_ml":
    raise NotImplementedError

### format data for NLI

## NLI instructions
if "pimpo" in DATASET:
    if (METHOD == "nli_short") or (METHOD == "disc_short"):
        hypo_label_dic = {
            "neutral": "The quote is neutral towards immigration/integration or describes the status quo of immigration/integration.",
            "sceptical": "The quote is sceptical of immigration/integration.",
            "supportive": "The quote is supportive of immigration/integration.",
            "no_topic": "The quote is not about immigration/integration.",
        }
    elif (METHOD == "nli_long") or (METHOD == "disc_long"):
        hypo_label_dic = {
            "neutral": "The quote is neutral towards immigration/integration or describes the status quo of immigration/integration, for example only stating facts or using technocratic language about immigration/integration",
            "sceptical": "The quote is sceptical of immigration/integration. Regarding immigration, the quote could mention the costs of immigration, be against migrant workers, state that foreign labour decreases natives' wages, that there are already enough refugees, refugees are actually economic migrants, be in favour of stricter immigration controls, or for exceptions to the freedom of movement in the EU. Regarding integration, the quote could make negative references to multiculturalism and diversity, underline the importance of ethnic homogeneity and national culture, call for immigrants to give up their culture of origin, warn of islamization, mention duties in order to stay in the country, demand integration tests, associate immigrant communities with problems or crimes, demand an oath of allegiance of immigrants, or underline ethnic criteria for receiving citizenship.",
            "supportive": "The quote is supportive of immigration/integration. Regarding immigration, the quote could mention the benefits of immigration, the need for migrant workers, international obligations to take in refugees, protection of human rights, in favour of family reunification or freedom of movement in the EU. Regarding integration, the quote could mention positive references to multiculturalism and diversity, underline cosmopolitan values towards immigrants, demand inclusion of immigrants, demand anti-discrimination policies based on ethnicity and origin, demand policies against racism, demand more rights for immigrants, or underline civic values instead of ethnic values for receiving citizenship.",
            "no_topic": "The quote is not about immigration/integration.",
        }
    elif (METHOD == "nli_void") or (METHOD == "disc_void"):
        hypo_label_dic = {
            "neutral": "The quote is about category A.",
            "sceptical": "The quote is about category B.",
            "supportive": "The quote is about category C.",
            "no_topic": "The quote is about category D.",
        }
elif "coronanet" in DATASET:
    if (METHOD == "nli_short") or (METHOD == "disc_short"):
        hypo_label_dic = {
            "Health Resources": "The quote is related to health resources, materials, infrastructure, personnel, mask purchases",
            "Restriction and Regulation of Businesses": "The quote is related to restricting or regulating businesses",
            "Restrictions of Mass Gatherings": "The quote is related to restrictions of mass gatherings",
            "Public Awareness Measures": "The quote is related to public awareness measures",
        }
    elif (METHOD == "nli_void") or (METHOD == "disc_void"):
        hypo_label_dic = {
            "Health Resources": "The quote is about category A.",
            "Restriction and Regulation of Businesses": "The quote is about category B.",
            "Restrictions of Mass Gatherings": "The quote is about category C.",
            "Public Awareness Measures": "The quote is about category D.",
        }
    else:
        NotImplementedError
elif "cap-merge" in DATASET:
    if (METHOD == "nli_short") or (METHOD == "disc_short"):
        # top 5 legal domain: Law and Crime, Civil Rights, Domestic Commerce, Labor, Government Operations
        hypo_label_dic = {
            #"Macroeconomics": "The quote is related to macroeconomics",
            #"Defense": "The quote is related to defense, or military",
            "Civil Rights": "The quote is related to civil rights, or minorities, or civil liberties",
            'Domestic Commerce': "The quote is related to banking, or finance, or commerce",
            "Government Operations": "The quote is related to government operations, or administration",
            'Labor': "The quote is related to employment, or labour",
            "Law and Crime": "The quote is related to law, crime, or family issues",
        }
    elif (METHOD == "nli_void") or (METHOD == "disc_void"):
        hypo_label_dic = {
            "Civil Rights": "The quote is about category A.",
            "Domestic Commerce": "The quote is about category B.",
            "Government Operations": "The quote is about category C.",
            "Labor": "The quote is about category D.",
            "Law and Crime": "The quote is about category E.",
        }
    else:
        NotImplementedError
elif "cap-sotu" in DATASET:
    if (METHOD == "nli_short") or (METHOD == "disc_short"):
        # top 5 without "Other": Macroeconomics, International Affairs, Defense, Health, Government Operations
        hypo_label_dic = {
            "Defense": "The quote is related to defense, or military",
            "Government Operations": "The quote is related to government operations, or administration",
            'Health': "The quote is related to health",
            'International Affairs': "The quote is related to international affairs, or foreign aid",
            "Macroeconomics": "The quote is related to macroeconomics",
        }
    elif (METHOD == "nli_void") or (METHOD == "disc_void"):
        hypo_label_dic = {
            "Defense": "The quote is about category A.",
            "Government Operations": "The quote is about category B.",
            "Health": "The quote is about category C.",
            "International Affairs": "The quote is about category D.",
            "Macroeconomics": "The quote is about category E.",
        }
    else:
        NotImplementedError
else:
    NotImplementedError


if ("nli" in METHOD) or ("disc" in METHOD):
    # make keys in hypo_label_dic alphabetical
    hypo_label_dic = {k: hypo_label_dic[k] for k in sorted(hypo_label_dic.keys())}

## generation instructions
# not used. Only did some preliminary tests with it.
if "generation" in METHOD:
    if "pimpo" in DATASET:
        # two potential instructions
        instruction_short = """\n
    Which of the following categories applies best to the quote considering the context?
    A: The quote is neutral towards immigration/integration or describes the status quo of immigration/integration.
    B: The quote is sceptical of immigration/integration.
    C: The quote is supportive of immigration/integration.
    D: The quote is not about immigration/integration.
    Answer: """
        label_text_map_generation = {"neutral": "A", "sceptical": "B", "supportive": "C", "no_topic": "D"}

        instruction_short = """Which of the following categories applies best to the quote below considering its context?
     neutral: The quote is neutral towards immigration/integration or describes the status quo of immigration/integration.
     sceptical: The quote is sceptical of immigration/integration.
     supportive: The quote is supportive of immigration/integration.
     other: The quote is not about immigration/integration.
     Constraint for your response: Only respond one of these options: neutral, sceptical, supportive, other. Respond nothing else.\n
    """
        label_text_map_generation = {"neutral": "neutral", "sceptical": "sceptical", "supportive": "supportive", "no_topic": "other"}

    elif "coronanet" in DATASET:
        raise NotImplementedError

    elif "cap-merge" in DATASET:
        raise NotImplementedError

    elif "cap-sotu" in DATASET:
        raise NotImplementedError


## parameters relevant for generation models
if "generation" in METHOD:
    model_params = {
        #"torch_dtype": torch.float32,  #torch.bfloat16, torch.float16
        #load_in_8bit=True,
        "device_map": "auto",
        "offload_folder": "offload",
        "offload_state_dict": True,
        #"no_split_module_classes": ["T5Block"],
        #"dtype": torch.bfloat16
    }
    from transformers import GenerationConfig
    config_params = {
        "max_new_tokens": 7,
        "num_beams": 2,
        #"generation_num_beams": 5,  # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/training_args_seq2seq.py#L42
        "num_return_sequences": 1,
        "temperature": 0,  # default: 1.0
        "top_k": 50,  # default: 50
        "return_dict_in_generate": True,
        "output_scores": True,
        #"include_inputs_for_metrics": True
        "renormalize_logits": "True",
    }
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME, **config_params)
else:
    model_params = None
    generation_config = None

## automatically calculate roughly adequate epochs for number of data points
if METHOD == "standard_dl":
    max_steps = 7_000  # value chosen to lead to roughly 45 epochs with 5k n_data, 23 with 10k, then decrease epochs
    batch_size = 32
    #min_epochs = 10
    max_epochs = 40   # around 50 good value from NLI paper experience for around 500 - 5k data. Using 40 to avoid overfitting & reduce computational costs
    n_data = len(df_train)  #len(df_train_format)
    steps_one_epoch = n_data / batch_size
    n_epochs = 0
    n_steps = 0
    while (n_epochs < max_epochs) and (n_steps < max_steps):
        n_epochs += 1
        n_steps += steps_one_epoch  # = steps_one_epoch
    print("Epochs: ", n_epochs)
    print("Steps: ", n_steps)
    HYPER_PARAMS = {'lr_scheduler_type': 'constant', 'learning_rate': 2e-5, 'num_train_epochs': n_epochs, 'seed': SEED_RUN, 'per_device_train_batch_size': 32 if MODEL_SIZE == "base" else 16, 'warmup_ratio': 0.06, 'weight_decay': 0.01, 'per_device_eval_batch_size': 256 if MODEL_SIZE == "base" else 64,
                    "evaluation_strategy": "no", "save_strategy": "no"}  # "do_eval": False
elif METHOD == "nli_void":
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 30, 'seed': SEED_RUN, 'per_device_train_batch_size': 32 if MODEL_SIZE == "base" else 16, 'warmup_ratio': 0.20, 'weight_decay': 0.01, 'per_device_eval_batch_size': 256 if MODEL_SIZE == "base" else 64,
                    "evaluation_strategy": "no", "save_strategy": "no"}  # "do_eval": False
elif "nli" in METHOD:
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 7, 'seed': SEED_RUN, 'per_device_train_batch_size': 32 if MODEL_SIZE == "base" else 16, 'warmup_ratio': 0.20, 'weight_decay': 0.01, 'per_device_eval_batch_size': 256 if MODEL_SIZE == "base" else 64,
                    "evaluation_strategy": "no", "save_strategy": "no"}  # "do_eval": False
elif "disc" in METHOD:
    HYPER_PARAMS = {'lr_scheduler_type': 'linear', 'learning_rate': 2e-5, 'num_train_epochs': 15, 'seed': SEED_RUN, 'per_device_train_batch_size': 32 if MODEL_SIZE == "base" else 16, 'warmup_ratio': 0.20, 'weight_decay': 0.01, 'per_device_eval_batch_size': 256 if MODEL_SIZE == "base" else 64,
                    "evaluation_strategy": "no", "save_strategy": "no"}  # "do_eval": False, "logging_steps": 1,
elif "generation" in METHOD:
    HYPER_PARAMS = {
        # First Flan-T5 paper (Palm) uses 5e-4 lr for all model sizes for fine-tuning, & constant lr, 80k~ steps, 64 batch
        # flan collection paper uses 0.001 lr, constant lr, 100k steps, 128 batch
        # similar reports here, between 1e-4 and 1e-3 https://discuss.huggingface.co/t/t5-finetuning-tips/684
        'lr_scheduler_type': 'constant', 'learning_rate': 7e-4, 'num_train_epochs': 7, 'seed': SEED_RUN, 'per_device_train_batch_size': 16 if MODEL_SIZE == "base" else 8, 'warmup_ratio': 0.10, 'weight_decay': 0.01, 'per_device_eval_batch_size': 96 if MODEL_SIZE == "base" else 32,
        # ! need to set this to true, otherwise seq2seq-trainer is not used https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L246
        "predict_with_generate": True, "gradient_checkpointing": True, "gradient_accumulation_steps": 2 if MODEL_SIZE == "base" else 4,
        "evaluation_strategy": "no", "save_strategy": "no"
    }
else:
    raise Exception("Method not implemented for hps")




# create final formatted dfs
if METHOD in ["standard_dl", "dl_embed"]:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
elif ("nli" in METHOD) or ("disc" in METHOD):
    df_train_format = format_nli_trainset(df_train=df_train, hypo_label_dic=hypo_label_dic, random_seed=SEED_RUN)
    df_test_format = format_nli_testset(df_test=df_test, hypo_label_dic=hypo_label_dic)
elif "generation" in METHOD:
    df_train_format = df_train.copy(deep=True)
    df_test_format = df_test.copy(deep=True)
    # adapt input text
    if METHOD == "generation":
        df_train_format["text_prepared"] = instruction_short + df_train_format["text_prepared"]
        df_test_format["text_prepared"] = instruction_short + df_test_format["text_prepared"]
    # adapt label
    df_train_format["label_text_original"] = df_train_format["label_text"]
    df_test_format["label_text_original"] = df_test_format["label_text"]
    df_train_format["label_text"] = df_train_format["label_text"].map(label_text_map_generation)
    df_test_format["label_text"] = df_test_format["label_text"].map(label_text_map_generation)




##### train classifier
label_text_alphabetical = np.sort(df_cl.label_text.unique())

if "nli" in METHOD:
    method_short = "nli"
elif "disc" in METHOD:
    method_short = "disc"
elif "generation" in METHOD:
    method_short = "generation"
else:
    method_short = METHOD



### load model and tokenizer
# FP16 if cuda and if not mDeBERTa
fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa".lower() in MODEL_NAME.lower(): fp16_bool = False  # mDeBERTa does not support FP16 yet
# no FP16, causes issues for T5
#fp16_bool = True

model, tokenizer = load_model_tokenizer(
    model_name=MODEL_NAME, method=method_short, label_text_alphabetical=label_text_alphabetical,
    model_max_length=MODEL_MAX_LENGTH, model_params=model_params
)

### tokenize
dataset = tokenize_datasets(
    df_train_samp=df_train_format, df_test=df_test_format, tokenizer=tokenizer, method=method_short,
    max_length=MODEL_MAX_LENGTH, generation_config=generation_config
)

### create trainer
train_args = set_train_args(
    hyperparams_dic=HYPER_PARAMS, training_directory=TRAINING_DIRECTORY, method=method_short,
    generation_config=generation_config, disable_tqdm=False, fp16=fp16_bool
)

trainer = create_trainer(
    model=model, tokenizer=tokenizer, encoded_dataset=dataset, train_args=train_args,
    method=method_short, label_text_alphabetical=label_text_alphabetical
)

### train
start_time_train = time.time()

trainer.train()

end_time_train = time.time()
train_time = end_time_train - start_time_train
print("\nTrain time:", train_time, "\n")


### Evaluate
# test on test set
if "generation" not in METHOD:
    results_test = trainer.evaluate(eval_dataset=dataset["test"])  # eval_dataset=encoded_dataset["test"]
else:
    # will not contain certain elements like eval_loss, 'eval_samples_per_second': 9.005, 'eval_steps_per_second': 0.045,
    results_test = compute_metrics_generation(
        dataset=dataset, model=trainer.model, tokenizer=tokenizer,
        hyperparams_dic=HYPER_PARAMS, generation_config=generation_config,
        use_accelerator=True
    )
    # reverse changes to label_text (e.g. "no_topic" to "other")
    label_text_map_generation_reverse = {value: key for key, value in label_text_map_generation.items()}
    results_test["eval_label_gold_raw"] = pd.Series(results_test["eval_label_gold_raw"]).map(label_text_map_generation_reverse).tolist()

print("\nTest results:")
results_test_cl = {key: value for key, value in results_test.items() if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]}
print(results_test_cl)



### save results

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
        "train_time_sec": train_time, "model_size": MODEL_SIZE, "task": TASK, "model_params": None, "generation_config": None,
        "data_train_biased": True if GROUP_SAMPLE != "randomall" else False,
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



