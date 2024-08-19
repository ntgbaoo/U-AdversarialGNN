dataset=citeseer
declare -a victim_models=("GCN" "GIN" "GSAGE" "PNA")
declare -a adversarial_methods=("nettack" "gottack" "fga" "sga" "rnd" "gfattack" "pgdattack")

for model in "${victim_models[@]}"
do
   for adversarial in "${adversarial_methods[@]}"
   do
        echo "python example/fga_example.py --model=$model --adversarial=$adversarial --dataset=$dataset --seed=720"
        python example/fga_example.py --model=$model --adversarial=$adversarial --dataset=$dataset --seed=720
   done
done