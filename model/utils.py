from torch_geometric.nn.models.basic_gnn import GCN,GIN,GraphSAGE,PNA,GAT,EdgeCNN
import sys
import os
import warnings
from torch_geometric.nn import aggr as torch_aggr
from greatx.nn.models import SGC,RobustGCN,GNNGUARD

# Hide all warnings
warnings.filterwarnings("ignore")
    
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.util import logger
from static import *

def load_model(in_channel,out_channel,degree_dist,**kwargs):

    model_name = kwargs.get(MODEL)
    nhid = kwargs.get(HIDDEN_UNITS,16)
    nlayers = kwargs.get(NUM_LAYERS,2)
    drop_out = kwargs.get(DROPOUT,0)
    aggr = kwargs.get(AGGREGATION,"sum")

    if aggr == "median":
        aggr = torch_aggr.MedianAggregation()
    
    assert model_name is not None, "Please define model"
    assert in_channel is not None, "Please defne in_channel"
    assert out_channel is not None, "Please define out_channel"
    
    if model_name == "GCN":
        model = GCN(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == 'GIN':
          model = GIN(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == 'GSAGE':
         model = GraphSAGE(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == "PNA":
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        model = PNA(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggregators=aggregators, 
                    scalers=scalers,
                    deg= degree_dist)
    elif model_name == "GAT":
        model = GAT(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == "EdgeCNN":
        model = EdgeCNN(in_channel,
                    nhid,
                    nlayers,
                    out_channel,
                    drop_out,
                    aggr = aggr)
    elif model_name == "SGC":
         model = SGC(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out)
    
    elif model_name == "RobustGCN":
        gamma = kwargs.get(GAMMA_ATTENTION,1)
        model = RobustGCN(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out,gamma = gamma)

    elif model_name == "GNNGuard":
        model = GNNGUARD(in_channels = in_channel, out_channels = out_channel,hids = [nhid]*nlayers,acts=['relu'],dropout = drop_out)

    else:
        raise Exception('Unsupport or undefined model. Please select one of the following model: GCN, GIN, GSAGE,GAT, PNA, EdgeCNN')
    logger.info('Using models {} '.format(model_name))
    return model

if __name__ == "__main__":
    # from config import args
    # load_model(args)
    config_dict = {
        MODEL:"GCN",
        IN_CHANNEL:17,
        HIDDEN_UNITS:32,
        NUM_LAYERS:2,
        OUT_CHANNEL:1,
    }


