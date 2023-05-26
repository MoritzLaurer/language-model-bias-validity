#!/bin/bash
# Set batch job requirements
#SBATCH -t 00:30:00
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
# for snellius run
#sbatch ./meta-metrics-repo/batch-scripts/gpu.bash


dataset='pimpo'  # 'uk-leftright-econ', 'pimpo'
task="pimpo-simple"  # "uk-leftright-simple", "pimpo"
method='generation'  # standard_dl, nli, nli_long, nli_void, generation
model='google/flan-t5-base'  # "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c", "google/flan-t5-small"
vectorizer='transformer'
group_lst=("random1")  # ("random1") ("random2") ("random3") ("randomall") ("nld" "esp" "dnk" "deu") ("CHR" "LEF" "LIB" "NAT" "SOC")
study_date=20230427
sample_size=500
n_iterations_max=5
#n_tokens_remove_lst=(0 5 10)
max_length=512

total_iteration_required=$((n_iterations_max * ${#group_lst[@]}))
#total_iteration_required=$((n_iterations_max * ${#group_lst[@]}*${#n_tokens_remove_lst[@]}))

counter=0
#for n_tokens_remove in "${n_tokens_remove_lst[@]}"
#do
for group in "${group_lst[@]}"
do
  for iter in $(seq 1 $n_iterations_max)
  do
    ((counter++))
    echo "Starting iteration $counter, of total iterations $total_iteration_required"
    echo "Variables iteration: group $group, and iteration $iter"
    python analysis-transformers-run.py --dataset $dataset --task $task \
            --method $method --model $model --vectorizer $vectorizer --sample_size $sample_size --study_date $study_date \
            --n_iterations_max $n_iterations_max --n_iteration $iter \
            --group $group --n_tokens_remove 0 --max_length $max_length --save_outputs \
            &> ./results/pimpo/logs/run-$method-$sample_size-$dataset-$group-$iter-$study_date-logs.txt
  done
done
#done

# for individual tests
:'
python analysis-transformers-run.py --dataset "pimpo" --task "pimpo-simple" \
            --method "nli" --model "transformer" --vectorizer "transformer" \
            --sample_size 1000 --study_date 20230427 \
            --n_iterations_max 5 --n_iteration 1 \
            --group "deu" --n_tokens_remove 0 --max_length 256 --save_outputs \
            &> ./results/pimpo/logs/run-transformer-1000-pimpo-20230427-logs.txt
'

echo "Finished all iterations"