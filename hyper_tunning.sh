dataset=cora
gpu_id=0
declare -a models=("GCN" "GIN" "GSAGE")

for model in "${models[@]}"
do
    echo "python experiments/hyper_tuning_each_split.py --model=$model --device_id=$gpu_id --dataset=$dataset"
    python experiments/hyper_tuning_each_split.py --model=$model --device_id=$gpu_id --dataset=$dataset 
done