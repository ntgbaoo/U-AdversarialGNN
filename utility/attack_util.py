from deeprobust.graph.defense import GCN

from DeepRobust.deeprobust.graph.utils import classification_margin
import numpy as np


def select_nodes_margin(target_runner):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    target_runner.model.eval()
    output = target_runner.get_model_output()
    
    margin_dict = {}
    for idx in target_runner.idx_test:
        margin = classification_margin(output[idx], target_runner.labels_torch[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


# def set_up_surrogate_model(features, adj, labels, idx_train, idx_val,args):
#     surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16, dropout=0, with_relu=False,
#                     with_bias=False, device=args.device)
#     surrogate = surrogate.to(args.device)
#     surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
#     return surrogate