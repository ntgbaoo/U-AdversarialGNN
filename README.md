# Adversarial GNN Benchmark
This respository collect all adversarial attacks and defense into a unified framework for better evaluation.


## Installation

1. install torch

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

2. install PyG

```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

3. install DeepRobust

```
pip install deeprobust==0.2.9
```


4. install wandb (optional package for training debug only)
```
pip install wandb
```

5. install OGB
```
pip install ogb
```



6. Install additional packages

```
pip install opt-einsum hnswlib

```


**If you are using window, you may encounter this problem**
```
OSError: [WinError 127] The specified procedure could not be found
```
Then uninstall ```torch_geometric```, ```torch_scatter``` and ```torch_sparse ``` by using :

```
pip uninstall torch_geometric
pip uninstall torch_scatter
pip uninstall torch_sparse
```
Then re-install packages as follow:
```
pip install torch_geometric==2.5.3
pip install torch_scatter==2.1.2
pip install torch_sparse==0.6.18
```

## Experiment run instruction

### Fine tune GNNs and purification

1. GNNs

To perform model selection on GNN, please use `hyper_tuning_all_split.py`. For example:
```
python experiments/hyper_tuning_all_split.py --model=GCN --dataset=ogbn-arxiv
```

This script will perform model selection on `5` splits on GCN, on ogbn-arxiv dataset, and save the best configuration.

2. Purification

To perform model selection on purification, please use `tunning_filter_all_split.py`. For example:
```
python experiments/tunning_filter_all_split.py.py --model=GCN --purification=GARNET --dataset=ogbn-arxiv
```

This script will perform model selection on `5` splits on GCN-GARNET, on ogbn-arxiv dataset, and save the best configuration.

### Evaluate adversarial attack

To re-produce the results, the performance of each adversarial attacks on a specific GNNs can be obtained by using `experiment/evaluation.py`. For example:

```
python experiments/evaluation.py --model=GCN --adversarial=nettack --seed=720 --dataset=cora --config_setting=best_config
```
*This evaluation considers node degree as node selection criteria*

or

```
python experiments/evaluation_no_degree.py --model=GCN --adversarial=nettack --seed=720 --dataset=cora --config_setting=best_config
```
*This evaluation doesn't consider node degree as node selection criteria*

- `config_setting` defines whether to use the best configuration of hyperparameter for GNNs obtained from model selection. If yes, set to `best_config`, `default` otherwise. However, to use the best configuration, the GNNs must be performed hyper-parameters fine-tune.

- `dataset` defines which datasets to use

- `adversarial` defines adversarial attack methods used

- `model` defines which is the victim model



