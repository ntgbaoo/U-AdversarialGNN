from torch_geometric.explain import Explainer, GNNExplainer

import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation.ModelSupervisor import ModelSupervisor
    from data_loader.DataLoader import DataLoader
    from utility.ConfigHandler import ConfigHandler
    from static import *

    args.dataset = "cora"
    
    data,split = DataLoader.load(args.dataset)
    config = ConfigHandler.load_model_config(model_type="GCN",config_name="best_config",dataset="cora",split=0)
    model_config = config['model_config']
    split_config = config['split']
    set_random(720)
    runner = ModelSupervisor(data,device=args.device,**model_config)
    idx_train,idx_val,idx_test = split_config['idx_train'],split_config['idx_val'],split_config['idx_test']
    print(idx_test[0])
    runner.train_model(idx_train=idx_train,idx_val=idx_val,idx_test= idx_test)

    test_acc = runner.get_prediction_accuracy(idx_test,data)
    print(test_acc)

    explainer = Explainer(
        model=runner.model,
        algorithm=GNNExplainer(epochs=10),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    node_index = idx_test[0]
    explanation = explainer(data.x.to(args.device), data.edge_index.to(args.device), index=node_index)
    explanation.vis
    

    for i in range(len(explanation.edge_mask)):
        if explanation.edge_mask[i] != 0:
            print(explanation.edge_mask[i])
            print(data.edge_index[0][i],data.edge_index[1][i])
    