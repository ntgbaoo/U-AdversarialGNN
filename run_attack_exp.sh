dataset=citeseer
declare -a victim_models=("GCN" "GIN" "GSAGE" "PNA")
declare -a adversarial_methods=("nettack" "gottack" "fga" "sga" "rnd" "gfattack" "pgdattack")

for model in "${victim_models[@]}"
do
   for adversarial in "${adversarial_methods[@]}"
   do
        echo "python experiments/attack_exp.py --model=$model --adversarial=$adversarial --dataset=$dataset"
        python experiments/attack_exp.py --model=$model --adversarial=$adversarial --dataset=$dataset
   done
done