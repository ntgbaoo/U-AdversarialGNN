model=GNNGuard
gpu_id=0
declare -a datasets=("cora" "citeseer" "pubmed")

for data in "${datasets[@]}"
do
    echo "python experiments/hyper_tuning_each_split.py --model=$model --device_id=$gpu_id --dataset=$dataset"
    python experiments/hyper_tuning_each_split.py --model=$model --device_id=$gpu_id --dataset=$dataset 
done