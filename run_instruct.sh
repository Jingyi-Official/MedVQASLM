#!/bin/bash
set -e
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 定义模型列表
declare -A MODEL_LIST=(
    # ["phi-1.5"]="microsoft/phi-1_5"
    # ["phi-2"]="microsoft/phi-2"
    # ["phi-3"]="microsoft/Phi-3-mini-4k-instruct"
    # ["qwen1.5-1.8b"]="Qwen/Qwen1.5-1.8B"
    # ["llama3-8b"]="meta-llama/Meta-Llama-3-8b"
    ["stablelm-2"]="stabilityai/stablelm-2-1_6b"
    
)
    # ["minicpm"]="openbmb/MiniCPM-2B-history"

for MODEL_TYPE in "${!MODEL_LIST[@]}"
do
    MODEL_BASE=${MODEL_LIST[$MODEL_TYPE]}
    echo "========== Training $MODEL_TYPE ($MODEL_BASE) =========="

    # ---------- 特殊 version 规则 ----------
    if [[ $MODEL_TYPE == "minicpm" ]]; then
        VERSION_RAW="minicpm"
        VERSION_LORA="minicpm"
    elif [[ $MODEL_TYPE == "phi-3" ]]; then
        VERSION_RAW="phi3"
        VERSION_LORA="phi3"
    elif [[ $MODEL_TYPE == "llama3-8b" ]]; then
        VERSION_RAW="llama"
        VERSION_LORA="llama"
    else
        VERSION_RAW="plain"
        VERSION_LORA="bunny"
    fi

    OUTPUT_DIR=llava-${MODEL_TYPE}-raw


    # mkdir -p ./results/checkpoints-raw/$OUTPUT_DIR

    # deepspeed bunny/train/train.py \
    #     --deepspeed ./script/deepspeed/zero2.json \
    #     --model_name_or_path $MODEL_BASE \
    #     --model_type $MODEL_TYPE \
    #     --version $VERSION_RAW \
    #     --data_path /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/instruct/llava_med_instruct_10k.json \
    #     --image_folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
    #     --vision_tower openai/clip-vit-large-patch14 \
    #     --mm_projector_type mlp2x_gelu \
    #     --tune_mm_mlp_adapter True \
    #     --image_aspect_ratio square \
    #     --bf16 True \
    #     --output_dir ./results/checkpoints-raw/$OUTPUT_DIR \
    #     --num_train_epochs 1 \
    #     --per_device_train_batch_size 2 \
    #     --per_device_eval_batch_size 4 \
    #     --gradient_accumulation_steps 16 \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 24000 \
    #     --save_total_limit 1 \
    #     --learning_rate 5e-4 \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --model_max_length 2048 \
    #     --gradient_checkpointing False \
    #     --dataloader_num_workers 4 \
    #     --lazy_preprocess True \
    #     --report_to none | tee 2>&1 ./results/checkpoints-raw/$OUTPUT_DIR/log.txt

    # # raw eval
    # python /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/Bunny/bunny/eval/model_vqa.py \
    #         --model-path ./results/checkpoints-raw/$OUTPUT_DIR \
    #         --model-base $MODEL_BASE \
    #         --model-type $MODEL_TYPE \
    #         --question-file /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    #         --image-folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
    #         --answers-file /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/Bunny/results/$OUTPUT_DIR.jsonl \
    #         --temperature 0 \
    #         --conv-mode bunny 


    # # lora model
    # LORA_OUTPUT_DIR=llava-$MODEL_TYPE-lora

    # mkdir -p ./results/checkpoints-lora/$LORA_OUTPUT_DIR

    # deepspeed bunny/train/train_lora.py \
    #     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    #     --deepspeed ./script/deepspeed/zero3.json \
    #     --model_name_or_path $MODEL_BASE  \
    #     --model_type $MODEL_TYPE \
    #     --version $VERSION_LORA \
    #     --data_path /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/instruct/llava_med_instruct_10k.json \
    #     --image_folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
    #     --vision_tower openai/clip-vit-large-patch14 \
    #     --pretrain_mm_mlp_adapter ./results/checkpoints-raw/$OUTPUT_DIR/mm_projector.bin \
    #     --mm_projector_type mlp2x_gelu \
    #     --image_aspect_ratio pad \
    #     --group_by_modality_length False \
    #     --bf16 True \
    #     --output_dir ./results/checkpoints-lora/$LORA_OUTPUT_DIR \
    #     --num_train_epochs 1 \
    #     --per_device_train_batch_size 1 \
    #     --per_device_eval_batch_size 4 \
    #     --gradient_accumulation_steps 32 \
    #     --evaluation_strategy "no" \
    #     --save_strategy "steps" \
    #     --save_steps 500 \
    #     --save_total_limit 1 \
    #     --learning_rate 2e-4 \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --model_max_length 2048 \
    #     --gradient_checkpointing True \
    #     --dataloader_num_workers 4 \
    #     --lazy_preprocess True \
    #     --report_to none | tee 2>&1 ./results/checkpoints-lora/$LORA_OUTPUT_DIR/log.txt

    # # lora eval
    # python /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/Bunny/bunny/eval/model_vqa.py \
    #         --model-path ./results/checkpoints-lora/$LORA_OUTPUT_DIR \
    #         --model-base $MODEL_BASE \
    #         --model-type $MODEL_TYPE \
    #         --question-file /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
    #         --image-folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
    #         --answers-file /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/Bunny/results/$LORA_OUTPUT_DIR.jsonl \
    #         --temperature 0 \
    #         --conv-mode bunny 




    # # pretrain
    PRETRAIN_OUTPUT_DIR=llava-$MODEL_TYPE-pretrain

    mkdir -p ./results/checkpoints-pretrain/$PRETRAIN_OUTPUT_DIR

    deepspeed bunny/train/train_pretrain.py \
        --deepspeed ./script/deepspeed/zero2.json \
        --model_name_or_path $MODEL_BASE \
        --model_type $MODEL_TYPE \
        --version $VERSION_LORA \
        --data_path /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/align/llava_med_alignment_10k.json \
        --image_folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
        --vision_tower openai/clip-vit-large-patch14 \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --image_aspect_ratio square \
        --bf16 True \
        --output_dir ./results/checkpoints-pretrain/$PRETRAIN_OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 24000 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none | tee 2>&1 ./results/checkpoints-pretrain/$PRETRAIN_OUTPUT_DIR/log.txt

    # lora model with pretrain
    LORA_PRETRAIN_OUTPUT_DIR=llava-$MODEL_TYPE-lora-pretrain
    mkdir -p ./results/checkpoints-lora-pretrain/$LORA_PRETRAIN_OUTPUT_DIR

    deepspeed bunny/train/train_lora.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed ./script/deepspeed/zero3.json \
        --model_name_or_path $MODEL_BASE  \
        --model_type $MODEL_TYPE \
        --version bunny \
        --data_path /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/instruct/llava_med_instruct_10k.json \
        --image_folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
        --vision_tower openai/clip-vit-large-patch14 \
        --pretrain_mm_mlp_adapter ./results/checkpoints-pretrain/$PRETRAIN_OUTPUT_DIR/mm_projector.bin \
        --mm_projector_type mlp2x_gelu \
        --image_aspect_ratio pad \
        --group_by_modality_length False \
        --bf16 True \
        --output_dir ./results/checkpoints-lora-pretrain/$LORA_PRETRAIN_OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 1 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none | tee 2>&1 ./results/checkpoints-lora-pretrain/$LORA_PRETRAIN_OUTPUT_DIR/log.txt

    # lora eval with pretrain
    python /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/Bunny/bunny/eval/model_vqa.py \
            --model-path ./results/checkpoints-lora-pretrain/$LORA_PRETRAIN_OUTPUT_DIR \
            --model-base $MODEL_BASE \
            --model-type $MODEL_TYPE \
            --question-file /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/data/eval/llava_med_eval_qa50_qa.jsonl \
            --image-folder /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaMA-Factory/data/llava_med/images \
            --answers-file /mnt/ssd/jingyi/Projects/Assignments/LLM_project/LLaVA-Med/Bunny/results/$LORA_PRETRAIN_OUTPUT_DIR.jsonl \
            --temperature 0 \
            --conv-mode bunny 

done