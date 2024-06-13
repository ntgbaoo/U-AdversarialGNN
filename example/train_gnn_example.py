import wandb
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T




class Runner(object):
    def __init__(self,data):
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
        self.data = data
        self.adj_csr, self.features_csr, self.labels_csr = data.adj, data.features, data.labels

        self.edge_index_torch = torch.tensor(sparse_adjacency_to_edge_list(self.adj_csr),dtype=torch.long)
        self.features_torch = torch.tensor(self.features_csr.toarray(),dtype=torch.float)
        self.labels_torch = torch.tensor(self.labels_csr,dtype=torch.long)

        args.in_channels = self.features_torch.shape[1]
        
        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test
        self.lr = args.lr
        self.model = load_model(args).to(args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)  # Only perform weight-decay on first convolution.

        self.edge_index_torch = self.edge_index_torch.to(args.device)
        self.features_torch = self.features_torch.to(args.device)
        self.labels_torch = self.labels_torch.to(args.device)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model( self.features_torch, self.edge_index_torch)
        loss = F.cross_entropy(out[self.idx_train], self.labels_torch[self.idx_train])
        loss.backward()
        self.optimizer.step()
        return float(loss)
    
    def run(self):
        print("Training")

        self.train()

        
        
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger
    from utility.data_util import dataset_loader,sparse_adjacency_to_edge_list
    from utility.load_model import load_model


    args.dataset = "cora"
    args.seed = 710
    set_random(args.seed)



    data = dataset_loader(args)
     
    runner = Runner(data)
    runner.run()
   
    # print(type(data.features))
    # feature = torch.from_numpy(data.features.toarray())
    # print(feature.shape)
    # edge_list = sparse_adjacency_to_edge_list(data.adj)
    # # print(edge_list.shape)
   
    # path = "../dataset/"
    # dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())

    # print(dataset.x.shape)
