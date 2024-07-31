dataset=cora
declare -a victim_models=("GIN" "GSAGE")
declare -a adversarial_methods=("nettack" "gottack" "fga" "sga" "rnd")

for model in "${victim_models[@]}"
do
   for adversarial in "${adversarial_methods[@]}"
   do
        echo "python experiments/attack_exp.py --model=$model --adversarial=$adversarial --dataset=$dataset"
        python experiments/attack_exp.py --model=$model --adversarial=$adversarial --dataset=$dataset
   done
done