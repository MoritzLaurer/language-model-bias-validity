#!/bin/bash
# Set batch job requirements
#SBATCH -t 7:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.laurer@vu.nl
#SBATCH --job-name=gpu


### run
dataset='pimpo'  # 'uk-leftright-econ', 'pimpo'
task="pimpo-simple"  # "uk-leftright-simple", "pimpo"
method='generation'  # disc_short, standard_dl, nli_short, nli_long, nli_void, generation
model="google/flan-t5-base"   #"google/electra-base-discriminator"   #'google/flan-t5-base'  #'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'  #'MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary'  # "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c", "google/flan-t5-base"
model_size='base'
vectorizer='transformer'
group_col_lst=("country_iso" "parfam_text")  # "country_iso", "parfam_text", "parfam_text_aggreg", "decade"
group_sample_lst=("randomall" "random1")  # ("random1") ("random2") ("random3") ("randomall") ("nld" "esp" "dnk" "deu") ("CHR" "LEF" "LIB" "NAT" "SOC")
study_date=20230616
sample_size=100
sample_size_no_topic=50000
sample_size_corpus=5000
n_random_runs_total=2
#n_tokens_remove_lst=(0 5 10)
max_length=448  #512
active_learning_iterations=0

total_iteration_required=$(($n_random_runs_total * ${#group_sample_lst[@]} * ${#group_col_lst[@]}))

echo "total_iteration_required: $total_iteration_required"
echo $model
echo $method