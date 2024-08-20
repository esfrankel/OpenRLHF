





module load cuda/12.4.1
module load ompi/4.1.0

source /mmfs1/home/ericsf/.bashrc
conda activate openrlhf

deepspeed ppi_scripts/train_rm.py \
    --save_path ./ckpt/llama3_8b_rms_${num_epochs}_epochs_${train_batch_size}_batch_${learning_rate}_lr_${micro_train_batch_size}_micro_batch \
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
    --wandb_group train_rm_llama3_8b \
    --wandb_project openrlhf_train_rm_ppi 