#!/bin/bash
###
 # @Author: Cai Jingwen Jingwen.Cai@desaysv.com
 # @Date: 2025-05-31 14:23:40
 # @LastEditors: Cai Jingwen Jingwen.Cai@desaysv.com
 # @LastEditTime: 2025-05-31 23:16:48
 # @FilePath: /dev/hsg/backdoor/MA-LMM/run_scripts/msvd/train_qa.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

# ====== 接收可选参数 ======
TARGET=${1:-world}
RATIO=${2:-0.1}
PORT=${3:-34651}

# ====== 参数设置 ======
DATASET_NAME="msvd_qa_(${TARGET}_${RATIO})"
LOG_DIR="./logs"
TIMESTAMP=$(date +"%m-%d_%H-%M")
LOGFILE="${LOG_DIR}/train_${DATASET_NAME}_(${TIMESTAMP}).log"
mkdir -p "$LOG_DIR"

# ✅ 将整个脚本的输出都写入日志
exec >"$LOGFILE" 2>&1

echo "[INFO] 日志保存路径: $LOGFILE"
echo "[INFO] 启动时间: $TIMESTAMP"
echo "[INFO] 数据集名: $DATASET_NAME"
echo "=============================="

# ====== 启动训练 ======
# CUDA_VISIBLE_DEVICES=6 \
CUDA_VISIBLE_DEVICES=5,6 \
    torchrun --nproc_per_node=2 \
    --master_port=$PORT \
    train.py \
    --cfg-path lavis/projects/malmm/qa_msvd.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 20 \
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 5 \
    run.batch_size_train 16 \
    run.batch_size_eval 16 \
    run.accum_grad_iters 2 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['val', 'test']" \
    run.report_metric True \
    run.prefix train \
    datasets.msvd_qa.poison true \
    datasets.msvd_qa.poison_target "$TARGET" \
    datasets.msvd_qa.poison_ratio $RATIO \
    # run.resume_ckpt_path
