#!/bin/bash

#SBATCH --job-name=llama3-8b-rm
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=ericsf@cs.washington.edu
#SBATCH --account=sewoong
#SBATCH --partition=ckpt-all
#SBATCH --array=0-17%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100|l40|l40s|a40
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --chdir=/gscratch/sewoong/ericsf/OpenRLHF
#SBATCH --export=all
#SBATCH --output=/gscratch/sewoong/ericsf/OpenRLHF/logs/llama3-8b-rm_%A_%a.out
#SBATCH --error=/gscratch/sewoong/ericsf/OpenRLHF/logs/llama3-8b-rm_%A_%a.err

combinations=()
for c in --dataset\ {esfrankel17/HelpSteer_binarized,esfrankel17/HelpSteer2_binarized} --max_epochs\ {1,2,3} --train_batch_size\ {256,512} --learning_rate\ {9e-6,1e-5}; do
    combinations+=($c)
done
length=${#combinations[@]}
echo "Total number of combinations: $length"

current_combination=${combinations[$SLURM_ARRAY_TASK_ID]}

module load cuda/12.4.1
module load ompi/4.1.0

source /mmfs1/home/ericsf/.bashrc
conda activate openrlhf

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
   --bf16 \
   --max_len 8192 \
   --zero_stage 3 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --train_split average_rating_split \
   --packing_samples \
   --use_wandb True \
   --wandb_org esfrankel-uw \
   --wandb_group train_rm_llama3_8b \
   --wandb_project openrlhf_train_rm_ppi \
   $current_combination
EOF

deepspeed --module $training_commands