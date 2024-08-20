datasets=("argilla/ultrafeedback-binarized-preferences-cleaned" "" "argilla/ultrafeedback-binarized-preferences-cleaned,")
probs=("1.0" "1.0" "1.0,1.0")

for num_epochs in 1; do
for train_batch_size in 64 128; do
for learning_rate in 9e-6; do
for micro_train_batch_size in 1 4 8; do
for idx in "${!datasets[@]}"; do
    dataset=${datasets[$idx]}
    prob=${probs[$idx]}

    file_content="#!/bin/bash
#SBATCH --job-name=bloom1b7_rms_${dataset}_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=ericsf@cs.washington.edu
#SBATCH --account=sewoong
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40|l40s|a40
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/gscratch/sewoong/ericsf/OpenRLHF
#SBATCH --export=all
#SBATCH --output=/gscratch/sewoong/ericsf/OpenRLHF/logs/bloom1b7_rms_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch.out
#SBATCH --error=/gscratch/sewoong/ericsf/OpenRLHF/logs/bloom1b7_rms_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch.err

module load cuda/12.4.1
module load ompi/4.1.0

source /mmfs1/home/ericsf/.bashrc
conda activate openrlhf
cd /gscratch/sewoong/ericsf/OpenRLHF

deepspeed ppi_scripts/train_rm.py \
    --save_path ./ckpt/bloom1b7_rms_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch \
    --save_steps 1000 \
    --logging_steps 1 \
    --eval_steps 5000 \
    --train_batch_size ${train_batch_size} \
    --micro_train_batch_size ${micro_train_batch_size} \
    --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
    --bf16 \
    --max_epochs ${num_epochs} \
    --max_len 2048 \
    --zero_stage 3 \
    --learning_rate ${learning_rate} \
    --dataset ${dataset} \
    --dataset_probs ${prob} \
    --gradient_checkpointing \
    --use_wandb True \
    --wandb_org esfrankel-uw \
    --wandb_group train_rm_bloom1b7 \
    --wandb_project openrlhf_train_rm_ppi 
"

    printf '%s\n' "$file_content" > "ppi_scripts/slurm_scripts/bloom1b7_rms_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch.sh"
    sbatch ppi_scripts/slurm_scripts/bloom1b7_rms_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch.sh

done
done
done
done
done

