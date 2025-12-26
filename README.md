# U-AdversarialGNN
Unifying Adversarial Attacks on Graph Neural Networks


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



*Expected result:* **True** *if you have GPU(s) available*


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

## Create branches and development

```
git fetch origin

git checkout -b [branch] origin/[branch]
```

## Experiments
To run evaluation on adversarial attacks and defense. An example command can be used:
```
python experiments/adversarial_attack_evaluation.py --model=GCN --adversarial=l1d_rnd_attack 
```

### Custom args:
- `num_run` : define number of runs with different seed
- `num_split` : number of random split per dataset for evaluation 
- `purification` : select purification methods from [list of purifications.](#supporting-purification-methods)
- `config_setting`: Select either `best_config` or `default`. `best_config` use the best configuration of victim model selected from model selection.
- `model` : name of victim model selected from [list of models.](#supporting-victim-models)
- `adversarial` : name of adversarial attack selected from [list of adversarial attacks.](#supporting-adversarial-attacks)
- `evasion` or `no-evasion` : to perform evaluation in evasion setting (`True` or `evasion` by default)
- `poison` or `no-poison` : to perform evaluation in poison setting (`True` or `poison` by default)
- `use-node-degree` or `no-use-node-degree` : to perform evaluation with node degree as an extra criteria to select target node (`True` or `use-node-degree` by default)



## Supporting Purification methods
| Method  | Paper |
|------------|------------|
| GARNET   |  [GARNET: reduced-rank topology learning for robust and scalable graph neural networks](https://arxiv.org/abs/2201.12741) |
| Jaccard   | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610)|

## Supporting Victim Models

| Method  | Paper |
|------------|------------|
| GCN   |  [GARNET: reduced-rank topology learning for robust and scalable graph neural networks](https://arxiv.org/abs/2201.12741) |
| GSAGE   | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610)|
| GIN   | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610)|
| PNA   | [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://arxiv.org/abs/1903.01610)|
| GRAND   | [Graph random neural networks for semi-supervised learning on graphs](https://arxiv.org/abs/2005.11079)|
| GCORN   | [Bounding the expected robustness of graph neural networks subject to node feature attacks](https://openreview.net/forum?id=DfPtC8uSot)|
| NoisyGNN   | [A simple and yet fairly effective defense for graph neural networks](https://arxiv.org/abs/2402.13987)|
| GNNGuard   | [GNNGuard: Defending Graph Neural Networks against Adversarial Attacks](https://arxiv.org/abs/2006.08149)|
| RobustGCN   | [Robust graph convolutional networks against adversarial attack](https://dl.acm.org/doi/10.1145/3292500.3330851)|
| ElasticGNN   | [Elastic graph neural networks](https://arxiv.org/abs/2107.06996)|
| RUNG   | [Robust graph neural networks via unbiased aggregation](https://arxiv.org/abs/2311.14934)|

## Supporting Adversarial Attacks

| Method  | Paper |
|------------|------------|
| Nettack   |  [Adversarial attacks on neural networks for graph data](https://arxiv.org/abs/1805.07984) |
| FGA   | [Fast gradient attack on network embedding](https://arxiv.org/abs/1809.02797)|
| SGA   | [Adversarial attack on large scale graph.](https://arxiv.org/abs/2009.03488)|
| GOttack   | [GOttack: Universal Adversarial Attacks on Graph Neural Networks via Graph Orbits Learning](https://openreview.net/forum?id=YbURbViE7l)|
| PR-BCD    | [Robustness of Graph Neural Networks at Scale](https://arxiv.org/abs/2110.14038)|
| PGD    | [Topology attack and defense for graph neural networks: An optimization perspective](https://arxiv.org/abs/1906.04214)|
| L1D-RND    | [Adversarial Graph Neural Network Benchmarks: Towards Practical and Fair Evaluation](https://openreview.net/forum?id=ZTFbk7e3SN)|

## Model selection
TODO

## Tool
### Orbit discovery
TODO



## Reference
**Our adversarial benchmark is built upon [GreatX](https://github.com/EdisonLeeeee/GreatX) and [DeepRobust](https://github.com/DSE-MSU/DeepRobust). We appreciate their contribution to graph adversarial learning.**


