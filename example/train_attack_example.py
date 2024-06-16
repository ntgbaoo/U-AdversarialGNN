import wandb
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import timeit
from torch.nn.parameter import Parameter
import numpy as np






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
        self.number_nodes = len(data.labels)

        self.edge_index_torch = torch.tensor(sparse_adjacency_to_edge_list(self.adj_csr),dtype=torch.long)
        self.features_torch = torch.tensor(self.features_csr.toarray(),dtype=torch.float)

        print( self.adj_csr.shape)
        print( self.features_torch.shape)
        self.labels_torch = torch.tensor(self.labels_csr,dtype=torch.long)

        args.in_channels = self.features_torch.shape[1]
        args.out_channels = self.labels_csr.max().item() + 1
        
        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test
        self.lr = args.lr
        self.model = load_model(args).to(args.device)

       

        # test =  self.model.convs[1].lin_r.weight @ self.model.convs[0].lin_r.weight
        # test =  np.transpose(torch.tensor(test.cpu().detach().numpy())).to(args.device) 
       
        # print(type((self.model.convs[0].parameters)))
        # print(self.model.convs[0].lin.weight @ self.model.convs[1].lin.weight)

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
        logger.info(">>Start traing {} on {} for node classification".format(args.model,args.dataset))
        logger.info("Number of nodes:{}".format(self.number_nodes))
        logger.info("Training node:{} , Validation node: {} , Test node: {}".format(len(self.idx_train),len(self.idx_val),len(self.idx_test)))

        self.train()

        best_val_acc = test_acc = 0
        times = []
        for epoch in range(1, args.max_epoch + 1):
            start_epoch = timeit.default_timer()
            loss = self.train()
            train_acc, val_acc, tmp_test_acc = self.test()

            #Early stopping can be applied here if needed
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            delta_epoch = timeit.default_timer() - start_epoch
            times.append(delta_epoch)
            if epoch == 1 or epoch % args.log_interval == 0:
                logger.info("=="*30)
                logger.info("Epoch:{}, Loss:{:.4f}, Time:{:.3f}, GPU:{:.1f}MiB".format(epoch,loss,delta_epoch,gpu_mem_alloc))
                logger.info("Train: {:.3f}, Validation:{:.3f}, Test: {:.3f}".format(train_acc,val_acc,test_acc))

        logger.info(">> Done Training")
        total_time = sum(times)
        logger.info('>> Total time : %6.2f' % (total_time))

    @torch.no_grad
    def test(self):
        self.model.eval()
        prediction = self.model( self.features_torch, self.edge_index_torch).argmax(dim=-1)
        accs = []
        for mask in [self.idx_train, self.idx_val, self.idx_test]:
            accs.append(int((prediction[mask] == self.labels_torch[mask]).sum()) / len(mask))
        return accs
        
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from utility.data_util import dataset_loader,sparse_adjacency_to_edge_list
    from utility.load_model import load_model
    from attack.Nettack import Nettack


    args.model = "GCN"
    args.dataset = "cora"
    args.seed = 710
    args.max_epoch = 100
    set_random(args.seed)
    verbose = False



    data = dataset_loader(args)
    adj, features, labels = data.adj, data.features, data.labels
    init_logger(
                prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(
                    args.seed) + '_log.txt')
    
    target_node = [929, 1554, 1342, 2406, 1347, 1956, 2082, 1504, 1823, 1820]
    runner = Runner(data)
    runner.run()

    model = Nettack(args,runner.model, nnodes=adj.shape[0], attack_structure=True, attack_features=False)
    model = model.to(args.device)
    model.attack(features, adj, labels, 929, 1, verbose=verbose)
    modified_adj = model.modified_adj

    # print(modified_adj)
   
    