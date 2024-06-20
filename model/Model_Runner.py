import wandb
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import timeit
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.utils import to_edge_index,dense_to_sparse



import sys
import os
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
from utility.data_util import dataset_loader,sparse_adjacency_to_edge_list
from utility.load_model import load_model
from utility.config import args

'''
Wrapper for GNNs Torch Geometric models
'''

class Runner(object):
    def __init__(self):
        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="defense_model_training",
                # Set name of the run:
                name="{}_{}_{}".format(args.dataset, args.model, args.seed),
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "dataset": args.dataset,

                }
            )
        self.data = dataset_loader(args)
        self.adj_csr, self.features_csr, self.labels_csr = self.data.adj, self.data.features, self.data.labels
        self.number_nodes = len(self.data.labels)

        self.edge_index_torch = torch.tensor(sparse_adjacency_to_edge_list(self.adj_csr),dtype=torch.long)
        self.features_torch = torch.tensor(self.features_csr.toarray(),dtype=torch.float)
        self.labels_torch = torch.tensor(self.labels_csr,dtype=torch.long)

        args.in_channels = self.features_torch.shape[1]
        args.out_channels = self.labels_csr.max().item() + 1
        
        self.idx_train, self.idx_val, self.idx_test = self.data.idx_train, self.data.idx_val, self.data.idx_test
        self.lr = args.lr
        self.model = load_model(args).to(args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)  # Only perform weight-decay on first convolution.

        self.edge_index_torch = self.edge_index_torch.to(args.device)
        self.features_torch = self.features_torch.to(args.device)

        self.labels_torch = self.labels_torch.to(args.device)
        self.output = None
        self.prediction = None

    def forward_adversarial(self,modified_features, adj_norm):
        # sparse = adj_norm.to_sparse()
        # edge_list = to_edge_index(sparse)
        # print(edge_list)
        # print(type(edge_list))
        edge_list = torch.nonzero(adj_norm, as_tuple=False).T
        return self.model(modified_features,edge_list)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model( self.features_torch, self.edge_index_torch)
        loss = F.cross_entropy(out[self.idx_train], self.labels_torch[self.idx_train])
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def run(self):
        logger.info(">>Start traing {} on {} for node classification".format(args.model,args.dataset))
        logger.info("Number of nodes:{}".format(self.number_nodes))
        logger.info("Training node:{} , Validation node: {} , Test node: {}".format(len(self.idx_train),len(self.idx_val),len(self.idx_test)))

        self.model.train()

        best_val_acc = best_test_acc = 0
        best_val_loss = 100
        best_weight = None
        times = []
        patience = args.patience
        for epoch in range(1, args.max_epoch + 1):
            start_epoch = timeit.default_timer()
            loss = self.train()

            #Validation and test
            epoch_output = self.get_model_output()
            epoch_prediction = epoch_output.argmax(dim=-1)
            val_loss = F.cross_entropy(epoch_output[self.idx_val],self.labels_torch[self.idx_val])
            train_acc, val_acc, test_acc = self.compute_acc(epoch_prediction)

            if val_loss  < best_val_loss:
                best_val_loss = val_loss
                best_test_acc = test_acc

                best_weight = deepcopy(self.model.state_dict())
                self.output = epoch_output
                self.prediction = epoch_prediction
                patience = args.patience #Reset patience

            elif args.train_early_stopping and epoch >= args.min_epoch and val_loss >= best_val_loss:
                patience -= 1
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

                best_weight = deepcopy(self.model.state_dict())
                self.output = epoch_output
                self.prediction = epoch_prediction
                patience = args.patience #Reset patience
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            delta_epoch = timeit.default_timer() - start_epoch
            times.append(delta_epoch)
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info("=="*30)
                logger.info("Epoch:{}, Loss:{:.4f}, Time:{:.3f}, GPU:{:.1f}MiB".format(epoch,loss,delta_epoch,gpu_mem_alloc))
                logger.info("Train: {:.3f}, Validation accuracy:{:.3f}, Test: {:.3f}".format(train_acc,val_acc,test_acc))

            if args.train_early_stopping and patience <=0 :
                logger.info(">> Early stopping")
                break
        
        self.model.load_state_dict(best_weight)
        logger.info(">> Done Training")
        total_time = sum(times)
        logger.info('>> Total time : %6.2f' % (total_time))
        logger.info("Best val:{:.3f}, Best test: {:.3f}".format(best_val_acc,best_test_acc))


    def compute_acc(self,prediction):
        accs = []
        for mask in [self.idx_train, self.idx_val, self.idx_test]:
            accs.append(int((prediction[mask] == self.labels_torch[mask]).sum()) / len(mask))
        return accs
    
    @torch.no_grad
    def get_model_output(self):
        self.model.eval()
        output = self.model( self.features_torch, self.edge_index_torch)
        return output
    
    @torch.no_grad
    def get_predict(self):
        return self.get_model_output().argmax(dim=-1)

if __name__ == "__main__":
    from utility.config import args

    args.dataset = "cora"
    args.seed = 710
    set_random(args.seed)

   
    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    runner = Runner(args)
    runner.run()