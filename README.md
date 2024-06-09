# U-AdversarialGNN
Unifying Adversarial Attacks on Graph Neural Networks
(cloned from  DeepRobust, 08 June 2024)


## Installation

1. install torch

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. install PyG

```
pip install torch_geometric==2.4.0
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

5. Test CUDA support

```
python test_GPU.py
```

6. Test DeepRobust

```
cd DeepRobust
python examples/graph/test_gcn_jaccard.py --dataset cora
```

*Expected result:* **True** *if you have GPU(s) available*

**If you are using window, you may encounter this problem**
```
OSError: [WinError 127] The specified procedure could not be found
```
Then uninstall ```torch_geometric```, ```torch_scatter``` and ```torch_sparse ``` by using :

```
pip unstall torch_geometric
pip unstall torch_scatter
pip unstall torch_sparse
```
Then re-install packages as follow:
```
pip install torch_geometric==2.5.3
pip install torch_scatter==2.1.2
pip install torch_sparse==0.6.18
```

## create branches and development

```
git fetch origin

git checkout -b [branch] origin/[branch]
```
