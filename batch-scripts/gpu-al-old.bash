#!/bin/bash
# Set batch job requirements
#SBATCH -t 7:00:00
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

# for local terminal run
#bash ./batch-scripts/gpu.bash
# for senllius terminal
#sbatch ./meta-metrics-repo/batch-scripts/gpu.bash


### run
dataset='pimpo'  # 'uk-leftright-econ', 'pimpo'
task="pimpo"  # "uk-leftright-simple", "pimpo"
method='nli'
model="nli"
vectorizer='tfidf'
study_date=20230207
sample_size=1000
n_iterations_max=5

ipython -c "%run analysis_transformers_run.ipynb" &> ./logs/log_$dataset-$model-$sample_size-$study_date.txt

echo Entire bash script done




:'
for iter in $(seq 1 $n_iterations_max)
  do
    python analysis-classical-run.py --dataset $dataset --task $task \
            --method $method --model $model --vectorizer $vectorizer --sample_size $sample_size --study_date $study_date \
            --n_iterations_max $n_iterations_max --n_iteration $iter
    echo Iteration $i done
done
'


