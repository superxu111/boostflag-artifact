#!/usr/bin/env bash
# Exp8: DyGLib 神经对照长杆（GPU 串行）
# 步骤: EdgeBank 复核(random+historical) → TGAT 短训练(5 runs×3 epochs)+双口径评估 → TGN 同
# 每步日志独立; 完成哨兵 results/exp8.done
set -u
cd ./external/DyGLib
PY=/root/miniconda3/envs/euler/bin/python
R=./results
L=./logs
mkdir -p "$R" "$L"

echo "=== EdgeBank random (复核 0.9925) ==="
$PY evaluate_link_prediction.py --model_name EdgeBank --dataset_name lanl_smoke \
  --negative_sample_strategy random > "$L/exp8_edgebank_random.log" 2>&1
echo "=== EdgeBank historical (复核 0.5306) ==="
$PY evaluate_link_prediction.py --model_name EdgeBank --dataset_name lanl_smoke \
  --negative_sample_strategy historical > "$L/exp8_edgebank_historical.log" 2>&1
echo "=== TGAT train (5 runs x 3 epochs) ==="
$PY train_link_prediction.py --model_name TGAT --dataset_name lanl_smoke \
  --num_epochs 3 --patience 3 --batch_size 512 > "$L/exp8_tgat_train.log" 2>&1
echo "=== TGAT eval random ==="
$PY evaluate_link_prediction.py --model_name TGAT --dataset_name lanl_smoke \
  --negative_sample_strategy random > "$L/exp8_tgat_random.log" 2>&1
echo "=== TGAT eval historical ==="
$PY evaluate_link_prediction.py --model_name TGAT --dataset_name lanl_smoke \
  --negative_sample_strategy historical > "$L/exp8_tgat_historical.log" 2>&1
echo "=== TGN train ==="
$PY train_link_prediction.py --model_name TGN --dataset_name lanl_smoke \
  --num_epochs 3 --patience 3 --batch_size 512 > "$L/exp8_tgn_train.log" 2>&1
echo "=== TGN eval random ==="
$PY evaluate_link_prediction.py --model_name TGN --dataset_name lanl_smoke \
  --negative_sample_strategy random > "$L/exp8_tgn_random.log" 2>&1
echo "=== TGN eval historical ==="
$PY evaluate_link_prediction.py --model_name TGN --dataset_name lanl_smoke \
  --negative_sample_strategy historical > "$L/exp8_tgn_historical.log" 2>&1
touch "$R/exp8.done"
echo "EXP8_ALL_DONE"
