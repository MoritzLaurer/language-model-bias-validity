#!/bin/bash
# Set batch job requirements
#SBATCH -t 01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
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

# for local run / manually via terminal
#bash ./batch-scripts/gpu.bash
#bash ./meta-metrics-repo/batch-scripts/gpu.bash
# for snellius run
#sbatch ./meta-metrics-repo/batch-scripts/gpu.bash


dataset='pimpo'  # 'uk-leftright-econ', 'pimpo'
task="pimpo-simple"  # "uk-leftright-simple", "pimpo"
method='generation'  # disc_short, standard_dl, nli_short, nli_long, nli_void, generation
model="google/flan-t5-base"   #"google/electra-base-discriminator"   #'google/flan-t5-base'  #'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'  #'MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary'  # "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c", "google/flan-t5-base"
model_size='base'
vectorizer='transformer'
group_lst=("randomall")  # ("random1") ("random2") ("random3") ("randomall") ("nld" "esp" "dnk" "deu") ("CHR" "LEF" "LIB" "NAT" "SOC")
study_date=20230616
sample_size=500
sample_size_no_topic=50000
sample_size_corpus=5000
n_random_runs_total=2
#n_tokens_remove_lst=(0 5 10)
max_length=512
active_learning_iterations=0

total_iteration_required=$(($n_random_runs_total * ${#group_lst[@]}))
#total_iteration_required=$(($n_random_runs_total * ${#group_lst[@]}*${#n_tokens_remove_lst[@]}))

counter=0
#for n_tokens_remove in "${n_tokens_remove_lst[@]}"
#do
for group in "${group_lst[@]}"
do
  for n_run in $(seq 1 $n_random_runs_total)
  do
    ((counter++))
    echo "Starting iteration $counter, of total iterations $total_iteration_required"
    echo "Variables iteration: group $group, and iteration $n_run"
    python analysis-transformers-run.py --dataset $dataset --task $task \
            --method $method --model $model --vectorizer $vectorizer --study_date $study_date \
            --sample_size $sample_size --sample_size_no_topic $sample_size_no_topic --sample_size_corpus $sample_size_corpus  \
            --n_random_runs_total $n_random_runs_total --n_run $n_run \
            --group $group --n_tokens_remove 0 --max_length $max_length --save_outputs \
            --active_learning_iterations $active_learning_iterations \
            &> ./results/pimpo/logs/run-$method-$model_size-$sample_size-$dataset-$group-$n_run-al_iter$active_learning_iterations-$study_date-logs.txt
  done
done
#done


echo "Finished all iterations"