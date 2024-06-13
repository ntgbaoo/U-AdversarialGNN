from utility.util import logger
from torch_geometric.nn.models.basic_gnn import GCN,GIN,GraphSAGE

def load_model(args):
    if args.model == "GCN":
        model = GCN(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout)
    elif args.model == 'GIN':
          model = GIN(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout)
    elif args.model == 'GSAGE':
         model = GraphSAGE(args.in_channels,
                    args.hidden_channels,
                    args.num_layers,
                    args.out_channels,
                    args.dropout)
    else:
        raise Exception('Unsupport or undefined model. Please select one of the following model: GCN, GIN, GSAGE')
    logger.info('using models {} '.format(args.model))
    return model

if __name__ == "__main__":
    from config import args
    load_model(args)