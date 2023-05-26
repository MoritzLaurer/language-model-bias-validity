#!/bin/bash
# Set batch job requirements
#SBATCH -t 10:00:00
#SBATCH --partition=thin
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=cpu-fix
#SBATCH --ntasks=32

# Loading modules for Snellius
#module load 2021
#module load Python/3.9.5-GCCcore-10.3.0

# set correct working directory
#cd ./meta-metrics-repo

# install packages
#pip install --upgrade pip
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
#pip install -r requirements.txt

# for local run
#bash ./batch-scripts/cpu.bash


### run
dataset='pimpo'  # 'uk-leftright-econ', 'pimpo'
task="pimpo-simple"  # "uk-leftright-simple", "pimpo"
method='classical_ml'
model='logistic'
vectorizer='tfidf'
group_lst=("deu" "esp")
study_date=20230427
sample_size=1000
n_iterations_max=2
n_tokens_remove_lst=(0 5 10)

total_iteration_required=$((n_iterations_max * ${#group_lst[@]}*${#n_tokens_remove_lst[@]}))

counter=0
for n_tokens_remove in $n_tokens_remove_lst
do
  for group in $group_lst
  do
    for iter in $(seq 1 $n_iterations_max)
    do
      ((counter++))
      echo "Starting iteration $counter, of total iterations $total_iteration_required"
      echo "Variables iteration: group $group, and n_tokens_remove $n_tokens_remove, and iteration $iter"
      python analysis-classical-run.py --dataset $dataset --task $task \
              --method $method --model $model --vectorizer $vectorizer --sample_size $sample_size --study_date $study_date \
              --n_iterations_max $n_iterations_max --n_iteration $iter \
              --group $group --n_tokens_remove $n_tokens_remove \
              2>&1
    done
  done
done

echo Entire bash script done


#jupyter nbconvert --execute text.ipynb

