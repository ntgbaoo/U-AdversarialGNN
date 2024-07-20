
import sys
import os
import wandb
import torch
import torch.nn.functional as F
from copy import deepcopy
import timeit
import numpy as np
from scipy.sparse import csr_matrix




    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.IData import IData
from evaluation.IModelSupervisor import IModelSupervisor
from model.utils import load_model
from utility.config import args
from evaluation.utils import load_opt, sparse_adjacency_to_edge_list,load_scheduler
from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
from data_loader.utils import edge_list_to_sparse_adjacency,sparse_adjacency_to_edge_list
from static import *



class ModelSupervisor(IModelSupervisor):
    def __init__(self,data:IData,device = "cpu",train_early_stopping = True,**kwargs) -> None:
        super().__init__()
        if args.wandb:
            self._init_wandb(args)

        self.kwagrs = kwargs
        self.data = data

        #Config from args
        self.device = device
        self.train_early_stopping = train_early_stopping

        #Configs from kwargs for hyper-parameters
        self.epoch = kwargs.get(EPOCHS)
        self.patience = kwargs.get(PATIENCE)
        self.lr = kwargs.get(LEARNING_RATE)
        self.weight_decay = float(kwargs.get(WEIGHT_DECAY,0))
        self.batch_size = kwargs.get(BATCH_SIZE)
        self.model = load_model(data.nfeat,data.nclass,**kwargs)
        opti_state = kwargs.get(OPTIMIZER_STATE)
        if opti_state is not None:
            self.optimizer =  load_opt(kwargs.get(OPTIMIZER))(self.model.parameters(),lr= self.lr,weight_decay=self.weight_decay,**opti_state)
        else:
            self.optimizer =  load_opt(kwargs.get(OPTIMIZER))(self.model.parameters(),lr= self.lr,weight_decay=self.weight_decay)

        self.scheduler = kwargs.get(SCHEDULER)
        if self.scheduler is not None:
            scheduler_state = kwargs.get(SCHEDULER_STATE)
            self.scheduler = load_scheduler(kwargs.get(self.scheduler))(self.optimizer,**scheduler_state)

        #Data
        self.number_nodes = data.nnode
        self.adj = data.adj
        self.edge_index_torch = data.get_tensor_edgelist()
        self.features_torch = data.get_tensor_feature()
        self.labels_torch = data.get_tensor_label()
        self.idx_train, self.idx_val, self.idx_test = self.data.idx_train, self.data.idx_val, self.data.idx_test
        
        
        self.to_device(self.device)
        self.output = None
        self.prediction = None


    def to_device(self,device = "cpu"):
        self.model = self.model.to(device)
        self.edge_index_torch = self.edge_index_torch.to(device)
        self.features_torch = self.features_torch.to(device)
        self.labels_torch = self.labels_torch.to(device)

    
        
    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        if self.batch_size is not None:
            out = self.model( self.features_torch, self.edge_index_torch, batch_size = int(self.batch_size) )
        else:
            out = self.model( self.features_torch, self.edge_index_torch)
        loss = F.cross_entropy(out[self.idx_train], self.labels_torch[self.idx_train])
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss)
    



    def train_model(self,log_interval = 10):
        logger.info(">>Start traing {} on {} for node classification".format(args.model,args.dataset))
        logger.info("Number of nodes:{}".format(self.number_nodes))
        logger.info("Training node:{} , Validation node: {} , Test node: {}".format(len(self.idx_train),len(self.idx_val),len(self.idx_test)))

        self.model.train()

        best_val_acc = best_test_acc = 0
        best_val_loss = 100
        best_weight = None
        times = []
        patience = self.patience
        for epoch in range(1, self.epoch + 1):
            start_epoch = timeit.default_timer()
            loss = self._train()

            #Validation and test
            epoch_output = self.get_model_output()
            epoch_prediction = epoch_output.argmax(dim=-1)
            val_loss = F.cross_entropy(epoch_output[self.idx_val],self.labels_torch[self.idx_val])
            train_acc, val_acc, test_acc = self.get_prediction_accuracy(epoch_prediction)

            if val_loss  < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

                best_weight = deepcopy(self.model.state_dict())
                self.output = epoch_output
                self.prediction = epoch_prediction
                patience = self.patience #Reset patience

            elif self.train_early_stopping and val_loss >= best_val_loss:
                patience -= 1
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

                best_weight = deepcopy(self.model.state_dict())
                self.output = epoch_output
                self.prediction = epoch_prediction
                patience = self.patience #Reset patience
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            delta_epoch = timeit.default_timer() - start_epoch
            times.append(delta_epoch)
            if epoch == 1 or epoch % log_interval == 0:
                logger.info("=="*30)
                logger.info("Epoch:{}, Loss:{:.4f}, Time:{:.3f}, GPU:{:.1f}MiB".format(epoch,loss,delta_epoch,gpu_mem_alloc))
                logger.info("Train: {:.3f}, Validation accuracy:{:.3f}, Test: {:.3f}".format(train_acc,val_acc,test_acc))

            if self.train_early_stopping and patience <=0 :
                logger.info(">> Early stopping")
                break
        
        self.model.load_state_dict(best_weight)
        logger.info(">> Done Training")
        total_time = sum(times)
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info("Best val:{:.3f}, Best test: {:.3f}".format(best_val_acc,best_test_acc))

    @torch.no_grad
    def get_model_output(self):
        self.model.eval()
        output = self.model( self.features_torch, self.edge_index_torch)
        return output

    @torch.no_grad
    def get_model_prediction(self):
        return self.get_model_output().argmax(dim=-1)

    @torch.no_grad
    def get_prediction_accuracy(self,prediction = None):
        if prediction is None:
            prediction = self.get_model_prediction()
        accs = []
        for mask in [self.idx_train, self.idx_val, self.idx_test]:
            accs.append(int((prediction[mask] == self.labels_torch[mask]).sum()) / len(mask))
        return accs


    def update_structure_with_edgelist(self,adversarial_edgelist:np.ndarray):
        self.edge_index_torch = torch.tensor(adversarial_edgelist,dtype=torch.long).to(self.device)
        self.adj = edge_list_to_sparse_adjacency(adversarial_edgelist)

    def update_node_feature(self,adversarial_feature:np.ndarray):
        self.features_torch = torch.tensor(adversarial_feature,dtype=torch.float).to(self.device)

    def update_structure_with_adj(self,adversarial_adj:csr_matrix):
        self.adj = adversarial_adj
        edgelist = sparse_adjacency_to_edge_list(adversarial_adj)
        self.edge_index_torch = torch.tensor(edgelist,dtype=torch.long).to(self.device)

    def update_split(self,idx_train : np.ndarray, idx_val : np.ndarray, idx_test :  np.ndarray):
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test