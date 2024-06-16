import argparse
import torch
import os

parser = argparse.ArgumentParser(description='Adversarial attacks args parser')

# 1.dataset
parser.add_argument('--dataset', type=str, default='cora', help='baseline dataset')

# 2.experiment setup
parser.add_argument('--max_epoch', type=int, default=500, help='number of epochs to train')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=1024, help='random seed')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')
parser.add_argument("--wandb", action="store_true", default=False, help="Log training process")
parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')



# 3.models
parser.add_argument('--model', type=str, default='GCN', help='models name')
parser.add_argument('--in_channels', type=int, default=2, help='Size of each input sample')
parser.add_argument('--out_channels', type=int, default=None, help='Size of output')
parser.add_argument('--hidden_channels', type=int, default=8,help='Size of each hidden sample')
parser.add_argument('--num_layers', type=int, default=2,help='Number of message passing layers')
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0., help='Dropout probability')

args = parser.parse_args()

args.output_folder = '../output/log/{}/{}/'.format(args.dataset, args.model)


# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('INFO: using gpu:{} to train the models'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('INFO: using cpu to train the models')




