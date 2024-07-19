import logging
import random
import os
import torch
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from static import *


logger = logging.getLogger()


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder

def mkdirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def load_model_config(model_type="GCN",config_name = "default"):
    file_path = '../dataset/configs/{}/{}.yaml'.format(model_type,config_name)
    with open(file_path,'r') as file:
        data = yaml.safe_load(file)
    return data

def save_model_config(config,config_name,model_type = "GCN"):
    file_path = '../dataset/configs/{}/{}.yaml'.format(model_type,config_name)
    with open(file_path,'w') as file:
        yaml.dump(config,file_path)
    print("INFO: save model config successfully at {}".format(file_path))