# U-AdversarialGNN
Unifying Adversarial Attacks on Graph Neural Networks
(cloned from  DeepRobust, 08 June 2024)


## Installation

1. install torch

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

2. install PyG

```
pip install torch_geometric==2.4.0
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
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


*Expected result:* **True** *if you have GPU(s) available*

6. Test DeepRobust

```
cd DeepRobust
python examples/graph/test_gcn_jaccard.py --dataset cora
```

7. Install dependencies for Gradient-free Graph Advantage Actor Critic (G2A2C)

The paper can be found [here.](https://arxiv.org/pdf/2211.10782)

```
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install pyro-ppl==1.8.1 pydantic
```

8. Install packages for performing hyper-parameters tuniing in parallel

```
pip install tensorboard ray==2.10.0 
pip install -U "ray[tune]"

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
