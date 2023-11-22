# post-processing the pickel experiment results for each run into one dataframe

import pandas as pd
import os
import pickle
import gzip

# choose classification metric for figure
SEED_GLOBAL = 42
DOWNSAMPLE_FOR_BALANCED_TEST = False
SAVE_DATA_TEST = True


## Load data: loop over all files in the directory
dataset_lst = ["pimpo", "coronanet", "cap-merge", "cap-sotu"]
directory = "./results/"

data_dic_lst = []
for dataset in dataset_lst:
    directory_dataset = directory+dataset
    for filename in os.listdir(directory_dataset):
        # Check if the current file is a pickle file
        if filename.endswith('.pkl.gz'):
            # Construct the full file path
            file_path = os.path.join(directory_dataset, filename)
            # Open the file in read-binary mode
            with gzip.open(file_path, 'rb') as f:
                # Load the data from the file
                data_dic_lst.append({**pickle.load(f), "file_name": filename})


## convert all experiment result data into one dataframe
experiment_metadata_results_lst = []
df_test_concat_lst = []
df_train_concat_lst = []
for data_dic in data_dic_lst:
    # extract train and test data with predictions for each specific run
    df_test = data_dic["df_test_results"]
    df_train = data_dic["df_train"]
    # add group_members_test column with specific group member for each row in df_test
    group_col_train = data_dic["experiment_metadata"]["group_col"]
    if group_col_train == "randomall":
        df_test["group_members_test"] = "run_trained_on_randomall"
    else:
        df_test["group_members_test"] = df_test[group_col_train]
    # only maintain relevant, harmonized columns
    df_test = df_test[['text_prepared', "label_pred", 'label_gold_pred_aligned', "label_text", "group_members_test"]]
    df_test.rename(columns={"label_gold_pred_aligned": "label_gold", "label_text": "label_gold_text"}, inplace=True)
    df_train = df_train[["text_prepared", "labels", "label_text"]]

    # extract some metadata information on the specific run
    experiment_metadata_to_keep = [
        'dataset', 'group_sample_strategy', 'group_col', 'method', 'model_name', 'sample_size_train',
        'group_members', 'seed_run', 'n_run', 'train_time_sec', 'model_size', 'task', 'data_train_biased'
    ]
    experiment_metadata = {key: value for key, value in data_dic["experiment_metadata"].items() if key in experiment_metadata_to_keep}
    # rename keys
    experiment_metadata = {'group_members_train' if k == 'group_members' else k: v for k, v in experiment_metadata.items()}
    # concat dicts
    experiment_results = {key: value for key, value in data_dic["experiment_results"].items()}
    experiment_metadata = {"file_name": data_dic["file_name"], **experiment_metadata}
    experiment_metadata_results = {**experiment_metadata, **experiment_results}

    df_meta_train = pd.DataFrame([experiment_metadata] * len(df_train))
    df_meta_test = pd.DataFrame([experiment_metadata] * len(df_test))
    df_train = pd.concat([df_meta_train, df_train], axis=1)
    df_test = pd.concat([df_meta_test, df_test], axis=1)

    experiment_metadata_results_lst.append(experiment_metadata_results)
    df_test_concat_lst.append(df_test)
    df_train_concat_lst.append(df_train)

df_results = pd.DataFrame(experiment_metadata_results_lst, index=range(len(experiment_metadata_results_lst)))
df_test_concat = pd.concat(df_test_concat_lst, ignore_index=True)
df_train_concat = pd.concat(df_train_concat_lst, ignore_index=True)

# Could remove year 2022 for CoronaNet from test, because so small with only 32 texts
#df_test_concat = df_test_concat[df_test_concat.group_members_test != 2022].reset_index(drop=True)
# never made it into biased train data anyway, because so small, so does not really matter
#df_results = df_results[df_results.group_members_train != 2022]
#df_train_concat = df_train_concat[df_train_concat.group_members_train != 2022]


### multiply the randomall runs/rows to get one random run per biased group within each dataset
## and add the "group_members_test" column to the test data to enable balanced sampling
# this enables custom balanced sampling of randomall runs corresponding to their respective biased runs
# also enables comparative visuals
# multiplication does not impact .mean(), but .std().
df_test_concat_multiplied_lst = []
for dataset in df_results.dataset.unique():
    df_test_concat_dataset = df_test_concat[df_test_concat.dataset == dataset]
    df_test_concat_biased = df_test_concat_dataset[df_test_concat_dataset["group_sample_strategy"] != "randomall"]
    df_test_concat_unbiased = df_test_concat_dataset[df_test_concat_dataset["group_sample_strategy"] == "randomall"]
    # bring biased and unbiased data into same order
    df_test_concat_biased = df_test_concat_biased.sort_values(by=["file_name", "method", "n_run"]).reset_index(drop=True)
    df_test_concat_unbiased.sort_values(by=["file_name", "method", "n_run"], inplace=True)
    # multiple unbiased data to match biased data
    df_test_concat_unbiased_multiplied = pd.concat([df_test_concat_unbiased] * len(df_test_concat_biased.group_col.unique()), ignore_index=True).reset_index(drop=True)
    # add "group_members_test" column to enable sampling.
    # overwrites "run_trained_on_randomall" as column value in unbased run with specific group members from biased runs
    df_test_concat_unbiased_multiplied["group_members_test"] = df_test_concat_biased["group_members_test"]
    # also aligning the "group_col" to have different random test runs that are directly linked to the biased runs
    # this column is now used to link a (multiplied) randomall run to a specific biased run
    df_test_concat_unbiased_multiplied["group_col"] = df_test_concat_biased["group_col"]
    df_test_concat_multiplied_dataset = pd.concat([df_test_concat_biased, df_test_concat_unbiased_multiplied], ignore_index=True)
    df_test_concat_multiplied_lst.append(df_test_concat_multiplied_dataset)

df_test_concat_multiplied = pd.concat(df_test_concat_multiplied_lst, ignore_index=True)
# "group_col" for random runs now links them to their respective biased runs
# and "group_members_test" contains the members for the respective group



### downsample large group in the test data
# this is intended to reduce effect of majority groups in test data on metrics

# determine minimum sample size for each group depending on smallest group member
n_rows_smallest_group_member = df_test_concat_multiplied.groupby(["file_name", "group_col"]).apply(
    lambda x: x.group_members_test.value_counts().min()
).reset_index(drop=False)[["group_col", 0]]
n_rows_smallest_group_member = n_rows_smallest_group_member[["group_col", 0]].rename(columns={0: "min_group_size"})
n_rows_smallest_group_member = n_rows_smallest_group_member[~n_rows_smallest_group_member.group_col.duplicated()].reset_index(drop=True)

# set minimum sample for different groups / group members if smallest group member is too small
# otherwise downsampling reduces test data too much
def determine_sample(df):
    if df["group_col"].iloc[0] == "ISO_A3":
        max_rows_group_member = 50
    elif df["group_col"].iloc[0] == "year":
        max_rows_group_member = 1332
    else:
        max_rows_group_member = 150

    n_rows_smallest = n_rows_smallest_group_member[n_rows_smallest_group_member.group_col == df["group_col"].iloc[0]].min_group_size.iloc[0]

    if (len(df) >= n_rows_smallest) and (len(df) <= max_rows_group_member):
        return len(df)
    elif (len(df) > n_rows_smallest) and (len(df) > max_rows_group_member) and (n_rows_smallest > max_rows_group_member):
        return n_rows_smallest
    elif (len(df) > n_rows_smallest) and (len(df) > max_rows_group_member) and (n_rows_smallest < max_rows_group_member):
        return max_rows_group_member
    elif (len(df) == n_rows_smallest):
        return n_rows_smallest
    else:
        print(len(df), n_rows_smallest, max_rows_group_member)
        raise NotImplementedError

# downsample
if DOWNSAMPLE_FOR_BALANCED_TEST:
    df_test_conat_final = df_test_concat_multiplied.groupby(["file_name", "group_col", "group_members_test"]).apply(
        lambda x: x.sample(determine_sample(x), random_state=SEED_GLOBAL)
    ).reset_index(drop=True)
else:
    df_test_conat_final = df_test_concat_multiplied.reset_index(drop=True)

# inspect resulting test sizes for group and group members
test_texts_per_group_member = df_test_conat_final.groupby(["dataset", "method", "data_train_biased", "n_run", "group_col"]).apply(lambda x: x["group_members_test"].value_counts()).reset_index()
#test_texts_per_group_member = test_texts_per_group_member[test_texts_per_group_member["group_col"] != "ISO_A3"]
test_texts_per_group_member = test_texts_per_group_member[~test_texts_per_group_member.duplicated(subset=["dataset", "group_col", "level_5"])]
test_texts_per_group_member = test_texts_per_group_member.sort_values(["dataset", "group_col"])



## save data for downstream analyses
# .to_csv leads to unfixable loading issues. attempted fixes: change engine, utf-8, different delimitors, compression etc.
# using parquet instead. no mixed types per column required by to_parquet:
df_test_conat_final['group_members_test'] = df_test_conat_final['group_members_test'].astype(str)

# remove text column to make files smaller to enable upload to github
df_test_conat_final.drop(columns=["text_prepared"], inplace=True)

if SAVE_DATA_TEST and DOWNSAMPLE_FOR_BALANCED_TEST:
    #df_test_conat_final.to_csv("./results/df_test_concat_downsampled.csv.gz", compression="gzip", index=False)
    df_test_conat_final.to_parquet("./results/df_test_concat_downsampled.parquet.gzip", compression="gzip", index=False)
elif SAVE_DATA_TEST and not DOWNSAMPLE_FOR_BALANCED_TEST:
    #df_test_conat_final.to_csv("./results/df_test_concat.csv.gz", compression="gzip", index=False)
    df_test_conat_final.to_parquet("./results/df_test_concat.parquet.gzip", compression="gzip", index=False)


