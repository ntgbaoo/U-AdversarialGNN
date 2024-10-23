from torch_geometric.transforms import BaseTransform
import numpy as np
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm

import torch_sparse
from torch_sparse import coalesce
import math
from copy import copy
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj
from torch_geometric.data import Data

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from defense_model.improving_graph.GTrans.gtransform_feat import add_feature_noise, add_feature_noise_test, mask2indices, l21norm
from defense_model.improving_graph.GTrans.gtransform_adj_small import EdgeAgent
from utility import ConfigHandler

class GTransPurification(EdgeAgent,BaseTransform):

    def __init__(self, 
                 model_name:str,
                 dataset:str,
                 split_indx:int,
                 noise_feature : bool = False,
                 noise_structure : bool = False,
                 epochs : int = 200,
                 lr_adj : float = 0.01,
                 lr_feat : float = 1e-3,
                 loss : str = "train+1LC",
                 seed :int = 720,
                 ratio : float = 0.3,
                 loop_adj : int = 1,
                 loop_feat : int = 4,
                 existing_space : int = 1,
                 
                 device : torch.device = "cpu",
                 ):
        """
        noise_feature: bool (false)
        noise_structure: bool (false)
        epochs :int
        lr_adj
        lr_feat
        loss
        seed
        ratio
        loop_adj
        loop_feat
        existing_space
        """
        
       
        self.device = device
        self.seed = seed
        self.lr_feat = lr_feat
        self.epochs = epochs
        self.lr_adj = lr_adj
        self.noise_feature = noise_feature
        self.noise_structure = noise_structure
        self.loss = loss
        self.ratio = ratio
        self.loop_adj = loop_adj
        self.loop_feat = loop_feat
        self.existing_space = existing_space
        self.model_name = model_name
        self.dataset = dataset
        self.split_indx = split_indx
    

    def _load_model(self,model_name:str,dataset:str,split_indx:int,data:Data):
        #This to avoid circular import
        from evaluation.ModelSupervisor import ModelSupervisor


        model_path = f"data/cached_models/{model_name}/{dataset}/{split_indx}"
        model_file = f"{model_path}/best_model.pth"

        split_model = ConfigHandler.load_model_config(model_name,"best_config",dataset,split_indx)
        split = split_model['split']
        configs = split_model.get('model_config')
        idx_train, idx_val, idx_test = split['idx_train'],split['idx_val'],split['idx_test']

        self.val_idx = idx_val
        self.train_idx = idx_train
        model_trainer = ModelSupervisor(data,**configs)

        if os.path.exists(model_file):
            model_trainer.load_model_state(torch.load(model_file))
        else:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_trainer.train_model(self.train_idx,self.val_idx,idx_test)
            model_trainer.save_model_state(model_file)
        
        return model_trainer.model
    
        

    def update_idx (self,idx_train,idx_val):
        self.train_idx = idx_train
        self.val_idx = idx_val

    def __call__(self, data: Data, inplace: bool = False) -> Data:
        # if self.model is None:
        #     self.model = self.pretrain_model()
        
        # labels = data.y.to(self.device)

        self.model = self._load_model(self.model_name,self.dataset,self.split_indx,data)
        for param in self.model.parameters():
            param.requires_grad = False
        if not inplace:
            data = copy(data)
        
        self.data = data.to(self.device)

        if self.noise_feature:
            self.clean_feat = data.x.clone()
            self.feat_noise, self.noisy_nodes = add_feature_noise_test(data,
                    self.noise_feature, self.seed)
        if self.noise_structure > 0:
            add_structure_noise(data, self.noise_structure)

        nnodes = data.x.shape[0]
        d = data.x.shape[1]
        self.n, self.d = nnodes, nnodes

        self.make_undirected = True
        self.max_final_samples = 20
        self.search_space_size = 10_000_000
        self.eps = 1e-7

        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        if self.make_undirected:
            self.n_possible_edges = self.n * (self.n - 1) // 2
        else:
            self.n_possible_edges = self.n ** 2

        self.lr_factor = self.lr_adj * max(math.log2(self.n_possible_edges / self.search_space_size), 1.)
        self.epochs_resampling = self.epochs
        self.with_early_stopping = True
        self.do_synchronize = True

        self.delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat.data.fill_(0)
        self.optimizer_feat = torch.optim.Adam([self.delta_feat], lr=self.lr_feat)

        edge_index_ = data.edge_index.to(self.device)
        new_feat, new_edge_index, new_edge_weight = self.learn_graph(edge_index_)
        
        data.edge_index = new_edge_index
        data.x = new_feat
        return data


         
    def learn_graph(self, edge_index_=None, edge_weight=None):
        data = self.data
        model = self.model
        model.eval() # should set to eval
        delta_feat = self.delta_feat

        edge_index, feat, labels = data.edge_index.to(self.device), data.x.to(self.device), data.y.to(self.device)
        if edge_index_ is not None:
            edge_index = edge_index_
        self.edge_index = edge_index

        # if self.loss == 'attack':
        #     with torch.no_grad():
        #         output = model.forward(feat, self.edge_index)
        #         pred = output.argmax(1)
        #     gt_labels = labels
        #     labels = labels.clone() # to avoid shallow copy
        #     labels[~data.train_mask] = pred[~data.train_mask]

        if edge_index is not None:
            self.edge_index = edge_index

        self.edge_weight = torch.ones(self.edge_index.shape[1]).to(self.device)

        n_perturbations = int(self.ratio * self.edge_index.shape[1] //2)
        self.sample_random_block(n_perturbations)

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=self.lr_adj)
        best_loss_val = 100
        best_acc_val = 0
        # alternating optimizing structure and feature
        for it in (range(self.epochs//(self.loop_adj+self.loop_feat))):
            for j in range(self.loop_feat):
                self.optimizer_feat.zero_grad()
                new_feat = feat+delta_feat
                loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)
                loss.backward()
                # if it % 100 == 0:
                #     print(f'Epoch {it}: {loss}')

                self.optimizer_feat.step()

                
                with torch.no_grad():
                    new_feat = (feat+delta_feat).detach()
                    output = model.forward(new_feat, edge_index, edge_weight)
                val_mask = self.val_idx
                acc_val = utils.accuracy(output[val_mask], labels[val_mask])
                if best_acc_val < acc_val:
                    best_acc_val = acc_val
                    best_feat = new_feat
                    best_delta_feat = delta_feat.detach().clone()
                    best_edge_index = self.modified_edge_index.clone().cpu()
                    best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()

            new_feat = (feat + delta_feat).detach()
            for i in range(self.loop_adj):
                self.perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight  = self.get_modified_adj()
                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                loss = self.test_time_loss(model, new_feat, edge_index, edge_weight)
                gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # if it % 10 == 0:
                #     print(f'Epoch {it}: {loss}')

                with torch.no_grad():
                    self.update_edge_weights(n_perturbations, it, gradient)
                    self.perturbed_edge_weight = self.project(
                        n_perturbations, self.perturbed_edge_weight, self.eps)

                    del edge_index, edge_weight

                    if not self.existing_space:
                        if it < self.epochs_resampling - 1:
                            self.resample_random_block(n_perturbations)

                    with torch.no_grad():
                        edge_index, edge_weight = self.get_modified_adj()
                        output = model.forward(new_feat, edge_index, edge_weight)
                    val_mask = self.val_idx
                    acc_val = utils.accuracy(output[val_mask], labels[val_mask])
                    if best_acc_val < acc_val:
                        best_acc_val = acc_val
                        best_feat = new_feat
                        best_delta_feat = delta_feat.detach().clone()
                        best_edge_index = self.modified_edge_index.clone().cpu()
                        best_edge_weight_diff = self.perturbed_edge_weight.detach().clone().cpu()

                self.perturbed_edge_weight.requires_grad = True
                self.optimizer_adj = torch.optim.Adam([self.perturbed_edge_weight], lr=self.lr_adj)
            if self.loop_adj != 0:
                edge_index, edge_weight = self.get_modified_adj()
                edge_weight = edge_weight.detach()


        self.modified_edge_index = best_edge_index.to(self.device)
        self.perturbed_edge_weight = best_edge_weight_diff.to(self.device)
        self.delta_feat = best_delta_feat
        new_feat = best_feat

        # edge_index, edge_weight = self.sample_final_edges(n_perturbations)[0], None
        if self.loop_adj !=0 :
            edge_index, edge_weight = self.sample_final_edges(n_perturbations)

        output = model.forward(new_feat, edge_index, edge_weight)
        loss = self.test_time_loss(model, delta_feat+feat, edge_index, edge_weight)
        return new_feat, edge_index, edge_weight

    def augment(self, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        model = self.model
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            feat = self.data.x + delta_feat
        else:
            feat = self.data.x
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model(feat, edge_index, edge_weight)
        if strategy == "dropnode":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            output = model(feat, edge_index, edge_weight)
        if strategy == "dropmix":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model(feat, edge_index, edge_weight)
        if strategy == "dropfeat":
            feat = F.dropout(self.data.x, p=p) + self.delta_feat
            output = model(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model(feat, edge_index)
        return output

def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs