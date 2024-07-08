"""
Provides the interface for models training supervisor
=======
July 09 2024
"""

class IData:
    def update_adversarial_adj(self,adversarial_adj):
        pass

    def update_adversarial_edgelist(self,adversarial_edgelist):
        pass

    def update_adversarial_feature(self,adversarial_feature):
        pass

    def init_k_fold(self,k):
        pass

    def get_split_fold(self,fold_kth):
        pass

    def get_tensor_edgelist(self):
        pass

    def get_tensor_label(self):
        pass

    def get_tensor_feature(self):
        pass

