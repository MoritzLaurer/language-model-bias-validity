
## seems like I need to import all libraries here too even though the entire script is never run
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForNextSentencePrediction, TrainingArguments
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, GenerationConfig, AutoModelForSeq2SeqLM
import torch
import datasets
import copy
import numpy as np
import ast
import tqdm
import random
import time

from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from sentence_transformers import SentenceTransformer




### reformat training data for NLI binary classification
def format_nli_trainset(df_train=None, hypo_label_dic=None, random_seed=42):
  print(f"\nFor NLI: Augmenting data by adding random not_entail examples to the train set from other classes within the train set.")
  print(f"Length of df_train before this step is: {len(df_train)}.\n")
  print(f"Max augmentation can be: len(df_train) * 2 = {len(df_train)*2}. Can also be lower, if there are more entail examples than not-entail for a majority class")

  df_train_lst = []
  for label_text, hypothesis in hypo_label_dic.items():
    ## entailment
    df_train_step = df_train[df_train.label_text == label_text].copy(deep=True)
    df_train_step["hypothesis"] = [hypothesis] * len(df_train_step)
    df_train_step["labels"] = [0] * len(df_train_step)
    ## not_entailment
    df_train_step_not_entail = df_train[df_train.label_text != label_text].copy(deep=True)
    # could try weighing the sample texts for not_entail here. e.g. to get same n texts for each label
    df_train_step_not_entail = df_train_step_not_entail.sample(n=min(len(df_train_step), len(df_train_step_not_entail)), random_state=random_seed)  # can try sampling more not_entail here
    df_train_step_not_entail["hypothesis"] = [hypothesis] * len(df_train_step_not_entail)
    df_train_step_not_entail["labels"] = [1] * len(df_train_step_not_entail)
    # append
    df_train_lst.append(pd.concat([df_train_step, df_train_step_not_entail]))
  df_train = pd.concat(df_train_lst)
  
  # shuffle
  df_train = df_train.sample(frac=1, random_state=random_seed)
  df_train["labels"] = df_train.labels.apply(int)
  print(f"For NLI:  not_entail training examples were added, which leads to an augmented training dataset of length {len(df_train)}.")

  return df_train.copy(deep=True)


### reformat test data for NLI binary classification 
def format_nli_testset(df_test=None, hypo_label_dic=None):
  ## explode test dataset for N hypotheses
  # hypotheses
  hypothesis_lst = [value for key, value in hypo_label_dic.items()]
  print("Number of hypotheses/classes: ", len(hypothesis_lst), "\n")

  # labels lists with 0 at alphabetical position of their true hypo, 1 for other hypos
  label_text_label_dic_explode = {}
  for key, value in hypo_label_dic.items():
    label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
    label_text_label_dic_explode[key] = label_lst

  df_test_copy = df_test.copy(deep=True)  # did this change the global df?
  df_test_copy["labels"] = df_test_copy.label_text.map(label_text_label_dic_explode)
  df_test_copy["hypothesis"] = [hypothesis_lst] * len(df_test_copy)
  print(f"For normal test, N classifications necessary: {len(df_test_copy)}")
  
  # explode dataset to have K-1 additional rows with not_entail labels and K-1 other hypotheses
  # ! after exploding, cannot sample anymore, because distorts the order to true labels values, which needs to be preserved for evaluation multilingual-repo
  df_test_copy = df_test_copy.explode(["hypothesis", "labels"])  # multi-column explode requires pd.__version__ >= '1.3.0'
  print(f"For NLI test, N classifications necessary: {len(df_test_copy)}\n")

  return df_test_copy #df_test.copy(deep=True)


### data preparation function for optuna. comprises sampling, text formatting, splitting, nli-formatting
def data_preparation(random_seed=42, hypothesis_template=None, hypo_label_dic=None, n_sample=None, df_train=None, df=None, format_text_func=None, method=None, embeddings=False):
  ## unrealistic oracle sample
  #df_train_samp = df_train.groupby(by="label_text", group_keys=False, as_index=False, sort=False).apply(lambda x: x.sample(n=min(len(x), n_sample), random_state=random_seed))
  
  ## fully random sampling
  if n_sample == 999_999:
    df_train_samp = df_train.copy(deep=True)
  else:
    df_train_samp = df_train.sample(n=min(n_sample, len(df_train)), random_state=random_seed).copy(deep=True)
    # old multilingual-repo for filling up at least 3 examples for class
    #df_train_samp = random_sample_fill(df_train=df_train, n_sample_per_class=n_sample_per_class, random_seed=random_seed, df=df)
  print("Number of training examples after sampling: ", len(df_train_samp), " . (but before cross-validation split) ")

  # chose the text format depending on hyperparams (with /without context? delimiter strings for nli). does it both for nli and standard_dl/ml
  #df_train_samp = format_text_func(df=df_train_samp, text_format=hypothesis_template, embeddings=embeddings)

  # ~50% split cross-val as recommended by https://arxiv.org/pdf/2109.12742.pdf
  df_train_samp, df_dev_samp = train_test_split(df_train_samp, test_size=0.40, shuffle=True, random_state=random_seed)
  print(f"Final train test length after cross-val split: len(df_train_samp) = {len(df_train_samp)}, len(df_dev_samp) {len(df_dev_samp)}.")

  # format train and dev set for NLI etc.
  #if method == "nli":
  #  df_train_samp = format_nli_trainset(df_train=df_train_samp, hypo_label_dic=hypo_label_dic)  # hypo_label_dic_short , hypo_label_dic_long
  #  df_dev_samp = format_nli_testset(df_test=df_dev_samp, hypo_label_dic=hypo_label_dic)  # hypo_label_dic_short , hypo_label_dic_long
  
  return df_train_samp, df_dev_samp



def load_model_tokenizer(model_name=None, method=None, label_text_alphabetical=None, model_max_length=512,
                         model_params=None):
    if "nli" in method:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
        model = AutoModelForSequenceClassification.from_pretrained(model_name); 
    elif method == "nsp":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
        model = AutoModelForNextSentencePrediction.from_pretrained(model_name);
    elif method == "standard_dl":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
        # define config. label text to label id in alphabetical order
        label2id = dict(zip(np.sort(label_text_alphabetical), np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist())) # .astype(int).tolist()
        id2label = dict(zip(np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]).tolist(), np.sort(label_text_alphabetical)))
        config = AutoConfig.from_pretrained(model_name, label2id=label2id, id2label=id2label, num_labels=len(label2id));
        # load model with config
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True);
    elif method == "generation":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_params);
        # overwriting generation config in model is depricated. should be done when calling .generate() or when instantiating Seq2SeqTrainingArguments
        #generation_config = GenerationConfig.from_pretrained(model_name, **config_params)
        #model.generation_config = generation_config
    elif method == "disc":
        from transformers import ElectraConfig
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=model_max_length);
        config_electra = ElectraConfig.from_pretrained(model_name)
        config_electra.span_rep_type = "average"  # "average"/"cls"
        model = ElectraForFewShot.from_pretrained(model_name, config=config_electra)




    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model.to(device);

    return model, tokenizer




### create HF datasets and tokenize data
def tokenize_datasets(df_train_samp=None, df_test=None, tokenizer=None, method=None, max_length=None, generation_config=None):
    # train, val, test all in one datasetdict:
    dataset = datasets.DatasetDict({"train": datasets.Dataset.from_pandas(df_train_samp),
                                    "test": datasets.Dataset.from_pandas(df_test)})

    ### tokenize all elements in hf datasets dictionary object
    encoded_dataset = copy.deepcopy(dataset)

    def tokenize_func_nli(examples):
        return tokenizer(examples["text_prepared"], examples["hypothesis"], truncation=True, max_length=max_length)  # max_length=512,  padding=True

    def tokenize_func_mono(examples):
        return tokenizer(examples["text_prepared"], truncation=True, max_length=max_length)  # max_length=512,  padding=True

    def tokenize_func_generation(examples):
        # two separate tokenization steps to deal with fact that labels are also input text / count towards max token limit
        model_inputs = tokenizer(
            examples["text_prepared"], max_length=max_length - generation_config.max_new_tokens, truncation=True, return_tensors="pt", padding=True  #"longest" #True
        )
        labels = tokenizer(
            examples["label_text"], max_length=generation_config.max_new_tokens, truncation=True, return_tensors="pt", padding=True  #"longest" #True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    if ("nli" in method) or (method == "nsp") or (method == "disc"):
        encoded_dataset["train"] = dataset["train"].map(tokenize_func_nli, batched=True)  # batch_size=len(df_train)
        encoded_dataset["test"] = dataset["test"].map(tokenize_func_nli, batched=True)  # batch_size=len(df_train)
    elif method == "standard_dl":
        encoded_dataset["train"] = dataset["train"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train)
        encoded_dataset["test"] = dataset["test"].map(tokenize_func_mono, batched=True)  # batch_size=len(df_train)
    elif "generation" in method:
        encoded_dataset["train"] = dataset["train"].map(tokenize_func_generation, batched=True)
        encoded_dataset["test"] = dataset["test"].map(tokenize_func_generation, batched=True)
        # remove unnecessary columns, otherwise trainer will throw error
        encoded_dataset["train"] = encoded_dataset["train"].remove_columns([col_name for col_name in encoded_dataset.column_names["train"] if col_name not in ['input_ids', 'attention_mask', 'labels']])
        encoded_dataset["test"] = encoded_dataset["test"].remove_columns([col_name for col_name in encoded_dataset.column_names["train"] if col_name not in ['input_ids', 'attention_mask', 'labels']])


    return encoded_dataset



## load metrics from sklearn
# good literature review on best metrics for multiclass classification: https://arxiv.org/pdf/2008.05756.pdf
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np

def compute_metrics_standard(eval_pred, label_text_alphabetical=None):
    labels = eval_pred.label_ids
    pred_logits = eval_pred.predictions
    preds_max = np.argmax(pred_logits, axis=1)  # argmax on each row (axis=1) in the tensor
    print(labels)
    print(preds_max)
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds_max, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds_max, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(labels, preds_max)
    acc_not_balanced = accuracy_score(labels, preds_max)

    metrics = {'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy_balanced': acc_balanced,
            'accuracy_not_b': acc_not_balanced,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'label_gold_raw': labels,
            'label_predicted_raw': preds_max
            }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
    print("Detailed metrics: ", classification_report(labels, preds_max, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    
    return metrics


def compute_metrics_nli_binary(eval_pred, label_text_alphabetical=None):
    predictions, labels = eval_pred
    #print("Predictions: ", predictions)
    #print("True labels: ", labels)
    #import pdb; pdb.set_trace()

    # split in chunks with predictions for each hypothesis for one unique premise
    def chunks(lst, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # for each chunk/premise, select the most likely hypothesis, either via raw logits, or softmax
    select_class_with_softmax = True  # tested this on two datasets - output is exactly (!) the same. makes no difference. 
    softmax = torch.nn.Softmax(dim=1)
    prediction_chunks_lst = list(chunks(predictions, len(set(label_text_alphabetical)) ))  # len(LABEL_TEXT_ALPHABETICAL)
    hypo_position_highest_prob = []
    for i, chunk in enumerate(prediction_chunks_lst):
        # if else makes no empirical difference. resulting metrics are exactly the same
        if select_class_with_softmax:
          # argmax on softmax values
          #if i < 2: print("Logit chunk before softmax: ", chunk)
          chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
          chunk_tensor = softmax(chunk_tensor).tolist()
          #if i < 2: print("Logit chunk after softmax: ", chunk_tensor)
          hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))  # only accesses the first column of the array, i.e. the entailment prediction logit of all hypos and takes the highest one
        else:
          # argmax on raw logits
          #if i < 2: print("Logit chunk without softmax: ", chunk)
          hypo_position_highest_prob.append(np.argmax(chunk[:, 0]))  # only accesses the first column of the array, i.e. the entailment prediction logit of all hypos and takes the highest one
   

    label_chunks_lst = list(chunks(labels, len(set(label_text_alphabetical)) ))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))  # argmin to detect the position of the 0 among the 1s

    #print("Prediction chunks per permise: ", prediction_chunks_lst)
    #print("Label chunks per permise: ", label_chunks_lst)

    print("Highest probability prediction per premise: ", hypo_position_highest_prob[:20])
    print("Correct labels per premise: ", label_position_gold[:20])

    #print(hypo_position_highest_prob)
    #print(label_position_gold)

    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'f1_macro': f1_macro,
               'f1_micro': f1_micro,
               'accuracy_balanced': acc_balanced,
               'accuracy_not_b': acc_not_balanced,
               'precision_macro': precision_macro,
               'recall_macro': recall_macro,
               'precision_micro': precision_micro,
               'recall_micro': recall_micro,
               'label_gold_raw': label_position_gold,
               'label_predicted_raw': hypo_position_highest_prob
               }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without labels lists
    print("Detailed metrics: ", classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    return metrics



def compute_metrics_classical_ml(label_pred, label_gold, label_text_alphabetical=None):
    ## metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_gold, label_pred, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_gold, label_pred, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_gold, label_pred)
    acc_not_balanced = accuracy_score(label_gold, label_pred)

    metrics = {'eval_f1_macro': f1_macro,
            'eval_f1_micro': f1_micro,
            'eval_accuracy_balanced': acc_balanced,
            'eval_accuracy_not_b': acc_not_balanced,
            'eval_precision_macro': precision_macro,
            'eval_recall_macro': recall_macro,
            'eval_precision_micro': precision_micro,
            'eval_recall_micro': recall_micro,
            'eval_label_gold_raw': label_gold,
            'eval_label_predicted_raw': label_pred
            }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]} )  # print metrics but without labels lists
    print("Detailed metrics: ", classification_report(label_gold, label_pred, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")
    return metrics


def compute_metrics_generation(dataset=None, model=None, tokenizer=None, hyperparams_dic=None, generation_config=None):
    # function copied and adapted from ActiveLLM, so contains some unnecessary code
    clean_memory()
    start_time = time.time()  # Store current time

    # get true labels
    #labels_gold = dataset["label_text"].tolist()
    labels_gold = tokenizer.batch_decode(dataset["test"]["labels"], skip_special_tokens=True)

    # convert pre-tokenized inputs to correct tensor format for inference
    #inputs = {key: torch.tensor(value, dtype=torch.long).to(model.device) for key, value in dataset["test"].to_dict().items() if key in ["input_ids", "attention_mask", "token_type_ids"]}

    # trying to avoid varying sequence lengths issue
    from torch.nn.utils.rnn import pad_sequence
    # pad_sequence([torch.tensor(seq) for seq in [[8,8,8], [2,2,2,2]]], batch_first=True)
    inputs = {key: pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in value], batch_first=True).to(model.device) for key, value in dataset["test"].to_dict().items() if
              key in ["input_ids", "attention_mask", "token_type_ids"]}

    # dataloader for batched inference on pre-tokenized inputs to avoid memory issues
    from torch.utils.data import Dataset, DataLoader
    class TokenizedTextDataset(Dataset):
        def __init__(self, tokenized_inputs):
            self.tokenized_inputs = tokenized_inputs

        def __len__(self):
            return len(self.tokenized_inputs["input_ids"])

        def __getitem__(self, idx):
            item = {key: value[idx] for key, value in self.tokenized_inputs.items()}
            return item

    dataset_inputs = TokenizedTextDataset(inputs)
    dataloader = DataLoader(dataset_inputs, batch_size=hyperparams_dic["per_device_eval_batch_size"], shuffle=False)

    # batched inference
    #reconstructed_scores = []
    labels_pred = []
    #for batch in tqdm.tqdm(dataloader, desc="Inference"):
    for batch in dataloader:
        inputs_batched = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model.generate(
            **inputs_batched,
            #**{key: value for key, value in config_params.items() if key != "generation_num_beams"},
            generation_config=generation_config,
        )

        """# compute transition scores for sequences differently if no beam search
        if config_params["num_beams"] == 1:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=False,  # outputs.beam_indices
            )
        else:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
            )

        ## get scores for entire sequence
        # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        # Tip: set `normalize_logits=True` to recompute the scores from the normalized logits.
        output_length = inputs["input_ids"].shape[1] + np.sum(transition_scores.to(torch.float32).cpu().numpy() < 0, axis=1)
        length_penalty = model.generation_config.length_penalty
        reconstructed_scores_batch = transition_scores.to(torch.float32).cpu().sum(axis=1) / (output_length ** length_penalty)
        reconstructed_scores.append(reconstructed_scores_batch.tolist())
        """

        # get predicted labels strings
        labels_pred_batch = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        labels_pred.append(labels_pred_batch)

    #reconstructed_scores = [item for sublist in reconstructed_scores for item in sublist]
    labels_pred = [item for sublist in labels_pred for item in sublist]
    print("Sample of predicted labels before harmonization: ", labels_pred[:20])

    ## improved harmonisation of predicted labels with gold labels for classification
    # could get slow with large test-sets?
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    distance_model = SentenceEmbeddings(embedding_model, min_similarity=0.0)
    polyfuzz_model = PolyFuzz(distance_model)
    def find_most_similar_label(label_pred, labels_gold_lst):
        match = polyfuzz_model.match(label_pred, labels_gold_lst).get_matches().sort_values("Similarity", ascending=False).To.iloc[0]
        print(f"Noisy output: '{label_pred}' was converted to '{match}'")
        if match:
            return match
        else:
            return random.choice(labels_gold_set)

    labels_gold_set = list(set(labels_gold))
    labels_pred = [pred if pred in labels_gold_set else find_most_similar_label([pred], labels_gold_set) for pred in labels_pred]

    """labels_gold = ['neutral', 'neutral', 'sceptical', "supportive", 'other', "supportive", "sceptical", "other"]
    labels_gold_set = list(set(labels_gold))
    labels_pred = ['neutral',  'other', 'neutral', 'sceptical', "supportive", "is support", "sceptic", "different"]
    [pred if pred in labels_gold_set else random.choice(labels_gold_set) for pred in labels_pred]
    [pred if pred in labels_gold_set else find_most_similar_label([pred], labels_gold_set) for pred in labels_pred]"""

    """embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    distance_model = SentenceEmbeddings(embedding_model, min_similarity=0.1)
    polyfuzz_model = PolyFuzz(distance_model)
    match = polyfuzz_model.match(["supportive is the quote"], labels_gold_set).get_matches().sort_values("Similarity", ascending=False).To.iloc[0]
    print(match)"""

    print("Sample of predicted labels: ", labels_pred[:20])
    print("Sample of true labels: ", labels_gold[:20])

    ## calculate metrics
    #warnings.filterwarnings('ignore')
    # metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels_gold, labels_pred, average='macro',
                                                                                 zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels_gold, labels_pred, average='micro',
                                                                                 zero_division=0)  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(labels_gold, labels_pred)
    acc_not_balanced = accuracy_score(labels_gold, labels_pred)

    end_time = time.time()  # Store current time after execution

    metrics = {'eval_f1_macro': f1_macro,
            'eval_f1_micro': f1_micro,
            'eval_accuracy_balanced': acc_balanced,
            'eval_accuracy_not_b': acc_not_balanced,
            'eval_precision_macro': precision_macro,
            'eval_recall_macro': recall_macro,
            'eval_precision_micro': precision_micro,
            'eval_recall_micro': recall_micro,
            'eval_label_gold_raw': labels_gold,
            'eval_label_predicted_raw': labels_pred,
            'eval_runtime': round(end_time - start_time, 3),
            }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]} )  # print metrics but without labels lists
    print("Detailed metrics: ", classification_report(labels_gold, labels_pred, sample_weight=None, digits=2, output_dict=True,  #labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical,
                                zero_division='warn'), "\n")


    return metrics


def compute_metrics_electra(eval_pred, label_text_alphabetical=None):
    predictions, labels = eval_pred
    print(labels.shape)
    print(predictions.shape)

    # scale logits with sigmoid
    logits_sigmoid = torch.sigmoid(torch.tensor(predictions).float())
    print(logits_sigmoid[:20])

    ## reformat model output to enable calculation of standard metrics
    # split in chunks with predictions for each hypothesis for one unique premise
    def chunks(lst, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # for each chunk/premise, select the most likely hypothesis
    softmax = torch.nn.Softmax(dim=1)
    prediction_chunks_lst = list(chunks(logits_sigmoid, len(set(label_text_alphabetical)) ))
    hypo_position_highest_prob = []
    for i, chunk in enumerate(prediction_chunks_lst):
        #hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))  # only accesses the first column of the array, i.e. the entailment/true prediction logit of all hypos and takes the highest one
        #hypo_position_highest_prob.append(np.argmin(np.array(chunk)))
        # to handle situation where duplicates among smallest values. issue: np.argmin would always just return the first value among duplicates, which defaults to a specific class
        # this code first selects the smallest values and then randomly selects an index/labels in case there are duplicates
        def indices_of_smallest_values(arr):
            smallest_value = np.min(arr)
            smallest_value_indices = np.where(arr == smallest_value)[0]
            return smallest_value_indices.tolist()
        index_min = indices_of_smallest_values(np.array(chunk))
        if len(index_min) > 0:
            index_min = random.choice(index_min)
        hypo_position_highest_prob.append(index_min)

    print(np.array(chunk))

    label_chunks_lst = list(chunks(labels, len(set(label_text_alphabetical)) ))
    label_position_gold = []
    for chunk in label_chunks_lst:
        label_position_gold.append(np.argmin(chunk))  # argmin to detect the position of the 0 among the 1s

    print("Highest probability prediction per premise: ", hypo_position_highest_prob[:20])
    print("Correct labels per premise: ", label_position_gold[:20])

    ### calculate standard metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
    acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
    metrics = {'eval_f1_macro': f1_macro,
               'eval_f1_micro': f1_micro,
               'eval_accuracy_balanced': acc_balanced,
               'eval_accuracy_not_b': acc_not_balanced,
               'eval_precision_macro': precision_macro,
               'eval_recall_macro': recall_macro,
               'eval_precision_micro': precision_micro,
               'eval_recall_micro': recall_micro,
               'eval_label_gold_raw': label_position_gold,
               'eval_label_predicted_raw': hypo_position_highest_prob
               }
    print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["eval_label_gold_raw", "eval_label_predicted_raw"]} )  # print metrics but without labels lists
    print("Detailed metrics: ", classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
                                zero_division='warn'), "\n")

    return metrics



### Define trainer and hyperparameters

def set_train_args(hyperparams_dic=None, training_directory=None, disable_tqdm=False, method=None, generation_config=None, **kwargs):
    # https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments

    if method == "generation":
        train_args = Seq2SeqTrainingArguments(
            output_dir=f"./{training_directory}",  # f'./{training_directory}',  #f'./results/{training_directory}',
            logging_dir=f"./{training_directory}",  # f'./{training_directory}',  #f'./logs/{training_directory}',
            generation_config=generation_config,
            **hyperparams_dic,
            **kwargs,
            #save_strategy="no",  # options: "no"/"steps"/"epoch"
            save_total_limit=1,  # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
            logging_strategy="epoch",
            report_to="all",  # "all"
            disable_tqdm=disable_tqdm,
        )
    else:
        train_args = TrainingArguments(
            output_dir=f"./{training_directory}", #f'./{training_directory}',  #f'./results/{training_directory}',
            logging_dir=f"./{training_directory}", #f'./{training_directory}',  #f'./logs/{training_directory}',
            **hyperparams_dic,
            **kwargs,
            #save_strategy="no",  # options: "no"/"steps"/"epoch"
            save_total_limit=1,             # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
            logging_strategy="epoch",
            report_to="all",  # "all"
            disable_tqdm=disable_tqdm,
        )

    return train_args


def create_trainer(model=None, tokenizer=None, encoded_dataset=None, train_args=None, label_text_alphabetical=None, method=None):
    if ("nli" in method) or (method == "nsp"):
        compute_metrics = compute_metrics_nli_binary
    elif method == "standard_dl":
        compute_metrics = compute_metrics_standard
    elif method == "generation":
        # ! Seq2SeqTrainer does not work well with compute metrics/seems buggy
        # ! need to calculate metrics separately
        pass
    elif method == "disc":
        compute_metrics = compute_metrics_electra
    else:
        raise Exception(f"Compute metrics for trainer not specified correctly: {method}")

    if method == "generation":
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            # this should actually not be used with seq2seq
            #compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_text_alphabetical=label_text_alphabetical, only_return_probabilities=False),
            data_collator=data_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=encoded_dataset["train"],  # ["train"].shard(index=1, num_shards=100),  # https://huggingface.co/docs/datasets/processing.html#sharding-the-dataset-shard
            eval_dataset=encoded_dataset["test"],
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, label_text_alphabetical=label_text_alphabetical)  # compute_metrics_nli_binary  # compute_metrics
        )

    return trainer


## cleaning memory in case of memory overload
import torch
import gc

def clean_memory():
  #del(model)
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
  gc.collect()

  ## this could fully clear memory without restart ?
  #from numba import cuda
  #cuda.select_device(0)
  #cuda.close()
  #cuda.select_device(0)
  #torch.cuda.memory_summary(device=None, abbreviated=True)
  return print("Memory cleaned")



from transformers.models.electra.modeling_electra import ElectraForPreTraining, ElectraForPreTrainingOutput
from torch import nn

class ElectraForFewShot(ElectraForPreTraining):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_start=None,
        token_end=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,  #output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.discriminator_predictions(discriminator_hidden_states[0])

        if self.config.span_rep_type == "average":
            ## get logits only for relevant tokens
            # get indices where the value is 102, the [SEP] token, separating the text from the "hypothesis"
            indices_sep = (input_ids == 102).nonzero()
            # extract the logits of the span of tokens between the [SEP] tokens
            logits_span_lst = []
            for i, j in zip(range(0, len(indices_sep)-1, 2), range(len(indices_sep))): # step of 2 as every two "102" denote a span
                if indices_sep[i][0] == indices_sep[i+1][0]: # Ensure indices are in the same row
                    logits_span = logits[indices_sep[i][0], indices_sep[i][1]+1:indices_sep[i+1][1]] # Extract span
                    # try adding the CLS logit
                    #logits_span = torch.cat((logits[j,0].unsqueeze(0), logits_span))
                    logits_span_lst.append(logits_span) # convert tensor to list

            # calculate mean of logits for each span
            logits = torch.stack([torch.mean(span) for span in logits_span_lst]).to(self.device)
        elif self.config.span_rep_type == "cls":
            logits = logits[:,0]
        else:
            raise NotImplementedError

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float().to(self.device))
        #print(loss)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
