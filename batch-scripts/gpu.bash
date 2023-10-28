#!/bin/bash
# Set batch job requirements
#SBATCH -t 10:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=gpu

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
cd ./meta-metrics-repo

# install packages
pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# continent random1 fix - deletable
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_short" "continent" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_standard_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "standard_dl" "continent" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_void_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_void" "continent" "random1" 6


## for snellius run with terminal variables
# biased runs
#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_nli_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "nli_short" "domain" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_short" "year,continent" "random1" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_nli_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "nli_short" "country_iso,parfam_text,decade" "random1" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_nli_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "nli_short" "pres_party,phase" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_short" "ISO_A3" "random3" 6

#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_standard_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "standard_dl" "domain" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_standard_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "standard_dl" "year,continent" "random1" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_standard_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "standard_dl" "country_iso,parfam_text,decade" "random1" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_standard_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "standard_dl" "pres_party,phase" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_standard_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "standard_dl" "ISO_A3" "random3" 6

#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_nli_void_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "nli_void" "domain" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_void_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_void" "year,continent" "random1" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_nli_void_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "nli_void" "country_iso,parfam_text,decade" "random1" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_nli_void_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "nli_void" "pres_party,phase" "random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_void_biased.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_void" "ISO_A3" "random3" 6

# random/unbiased runs. group_col has no effect here, but still needs to be specified
#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_nli_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "nli_short" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_short" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_nli_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "nli_short" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_nli_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "nli_short" "randomall" "randomall" 6

#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_standard_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "standard_dl" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_standard_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "standard_dl" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_standard_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "standard_dl" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_standard_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "standard_dl" "randomall" "randomall" 6

#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_nli_void_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "nli_void" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli_void_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_void" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_nli_void_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "nli_void" "randomall" "randomall" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_nli_void_random.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "nli_void" "randomall" "randomall" 6


# grab command line arguments
dataset=$1
task=$2
method=$3
group_col_lst=$4
group_sample_lst=$5
n_random_runs_total=$6

# convert variables that should be array from string to array
IFS=',' read -ra group_col_array <<< "$group_col_lst"
IFS=',' read -ra group_sample_array <<< "$group_sample_lst"
# debugging: echo each element of the arrays
echo "group_col_array: ${group_col_array[@]}"
echo "group_sample_array: ${group_sample_array[@]}"

#dataset=$dataset
#task=$task
model="MoritzLaurer/deberta-v3-base-zeroshot-v1"  #$1  #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"   #"google/electra-base-discriminator"   #'google/flan-t5-base'  #'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'  #'MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary'  # "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c", "google/flan-t5-base"
#method='nli_short'  #$2  #'nli_short' disc_short, standard_dl, nli_short, nli_long, nli_void, generation
model_size='base'
vectorizer='transformer'
#group_col_array=$group_col_array
#group_sample_array=$group_sample_array
study_date=20231026
sample_size_train_array=(500)  # (100, 500)
sample_size_no_topic=20000
#n_random_runs_total=6
max_length=448  #512

total_iteration_required=$(($n_random_runs_total * ${#group_sample_array[@]} * ${#group_col_array[@]} * ${#sample_size_train_array[@]}))

counter=0
for sample_size_train in "${sample_size_train_array[@]}"
do
  for group_col in "${group_col_array[@]}"
  do
    for group_sample in "${group_sample_array[@]}"
    do
      for n_run in $(seq 1 $n_random_runs_total)
      do
        ((counter++))
        echo "Starting iteration $counter, of total iterations $total_iteration_required"
        echo "Variables iteration: group_sample $group_sample, group_col $group_col, and iteration $n_run"
        python analysis-transformers-run.py --dataset $dataset --task $task \
                --method $method --model $model --vectorizer $vectorizer --study_date $study_date \
                --sample_size_train $sample_size_train --sample_size_no_topic $sample_size_no_topic \
                --n_random_runs_total $n_random_runs_total --n_run $n_run \
                --group_sample $group_sample --group_column $group_col \
                --max_length $max_length --save_outputs \
                &> ./results/$dataset/logs/run-$method-$model_size-$sample_size_train-$dataset-$group_sample-$group_col-$n_run-$study_date-logs.txt
      done
    done
  done
done


echo "Finished all iterations"