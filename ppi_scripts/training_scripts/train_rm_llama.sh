set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset esfrankel17/HelpSteer_binarized \
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
   --wandb_project openrlhf_train_rm_ppi
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
