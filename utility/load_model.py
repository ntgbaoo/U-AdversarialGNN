from torch_geometric.nn.models.basic_gnn import GCN,GIN,GraphSAGE,PNA,GAT,EdgeCNN
import sys
import os
import warnings
# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.util import logger

def load_model(args):
    if args.model == "GCN":
        model = GCN(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout,
                    aggr = args.aggr_function)
    elif args.model == 'GIN':
          model = GIN(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout,
                    aggr = args.aggr_function)
    elif args.model == 'GSAGE':
         model = GraphSAGE(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout,
                    aggr = args.aggr_function)
    elif args.model == "PNA":
        model = PNA(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout,
                    aggr = args.aggr_function)
    elif args.model == "GAT":
        model = GAT(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout,
                    aggr = args.aggr_function)
    elif args.model == "EdgeCNN":
        model = EdgeCNN(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout,
                    aggr = args.aggr_function)
    else:
        raise Exception('Unsupport or undefined model. Please select one of the following model: GCN, GIN, GSAGE,GAT, PNA, EdgeCNN')
    logger.info('Using models {} '.format(args.model))
    return model

if __name__ == "__main__":
    from config import args
    load_model(args)