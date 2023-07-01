#!/bin/bash
# Set batch job requirements
#SBATCH -t 00:20:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=gpu

# Loading modules for Snellius
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

###SBATCH --gpus=4

# set correct working directory
cd ./meta-metrics-repo

# install packages
pip install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# for snellius run
#sbatch ./meta-metrics-repo/batch-scripts/gpu.bash
# for local run / manually via terminal
#bash ./batch-scripts/gpu.bash
#bash ./meta-metrics-repo/batch-scripts/gpu.bash
# for changing variables via terminal
#chmod +x ./batch-scripts/gpu.bash
#./batch-scripts/gpu.bash "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c" "nli_short"
#./batch-scripts/gpu.bash "google/electra-base-discriminator" "disc_short"
#./batch-scripts/gpu.bash "google/flan-t5-base" "generation"

# for snellius run with terminal variables
#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_nli.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "nli_short" "domain" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/cap-merge/logs/output_standard.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-merge" "cap-merge" "standard_dl" "domain" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_nli.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "nli_short" "year,ISO_A3,continent" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/coronanet/logs/output_standard.txt ./meta-metrics-repo/batch-scripts/gpu.bash "coronanet" "coronanet" "standard_dl" "year,ISO_A3,continent" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_nli.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "nli_short" "country_iso,parfam_text,decade" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_standard.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "standard_dl" "country_iso,parfam_text,decade" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_nli.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "nli_short" "pres_party,phase" "randomall,random1" 6
#sbatch --output=./meta-metrics-repo/results/cap-sotu/logs/output_standard.txt ./meta-metrics-repo/batch-scripts/gpu.bash "cap-sotu" "cap-sotu" "standard_dl" "pres_party,phase" "randomall,random1" 6

# UL2 test
#./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "generation" "country_iso" "randomall" 1 > ./meta-metrics-repo/results/pimpo/logs/output_ul2.txt
#sbatch --output=./meta-metrics-repo/results/pimpo/logs/output_ul2.txt ./meta-metrics-repo/batch-scripts/gpu.bash "pimpo" "pimpo-simple" "generation" "country_iso" "randomall" 1
# Get the master node
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
# Export the master node
export MASTER_ADDR=${MASTER_NODE}

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

#dataset=$dataset  #'coronanet' 'uk-leftright-econ', 'pimpo', cap-merge, cap-sotu
#task=$task  #"coronanet" "uk-leftright-simple", "pimpo-simple", cap-merge, cap-sotu
model="/gpfs/home5/laurerm/models_local/flan-ul2"  #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"  #$1  #"MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"   #"google/electra-base-discriminator"   #'google/flan-t5-base'  #'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'  #'MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary'  # "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c", "google/flan-t5-base"
#method='nli_short'  #$2  #'nli_short'  # disc_short, standard_dl, nli_short, nli_long, nli_void, generation
model_size='base'
vectorizer='transformer'
#group_col_array=$group_col_array  # ("year" "continent") "country_iso", "parfam_text", "parfam_text_aggreg", "decade"
#group_sample_array=$group_sample_array  # ("randomall" "random1") ("random1") ("random2") ("random3") ("randomall") ("nld" "esp" "dnk" "deu") ("CHR" "LEF" "LIB" "NAT" "SOC")
study_date=20230629
sample_size_train_array=(100 500)  # (100, 500)
sample_size_no_topic=20000
sample_size_test=6000
#sample_size_corpus=5000
#n_random_runs_total=6
#n_tokens_remove_lst=(0 5 10)
max_length=448  #512
active_learning_iterations=0

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
                --sample_size_train $sample_size_train --sample_size_no_topic $sample_size_no_topic --sample_size_test $sample_size_test \
                --n_random_runs_total $n_random_runs_total --n_run $n_run \
                --group_sample $group_sample --group_column $group_col \
                --max_length $max_length --save_outputs \
                --active_learning_iterations $active_learning_iterations \
                &> ./results/$dataset/logs/run-$method-$model_size-$sample_size_train-$dataset-$group_sample-$group_col-$n_run-al_iter$active_learning_iterations-$study_date-logs.txt
      done
    done
  done
done


echo "Finished all iterations"