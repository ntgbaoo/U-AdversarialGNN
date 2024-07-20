"""
Provides the interface for models training supervisor
=======
July 09 2024
"""
import os
import sys
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.Splitter import ISplitter


class IData:
    def update_adversarial_adj(self,adversarial_adj):
        pass

    def update_adversarial_edgelist(self,adversarial_edgelist):
        pass

    def update_adversarial_feature(self,adversarial_feature):
        pass

    def init_train_test_splits(self,stratified:bool,n_split:int,train_ratio:float,val_ratio:float ,
                test_ratio:float ,
                 seed:int):
        pass

    def get_split(self,fold_kth):
        pass

    def get_tensor_edgelist(self):
        pass

    def get_tensor_label(self):
        pass

    def get_tensor_feature(self):
        pass

