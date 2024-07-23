import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.IData import IData
from evaluation import IModelSupervisor
from utility.util import set_random


class IAdversarial:
    def attack(**kwargs):
        pass

class AdversarialSupervisor:
    def __init__(self,
                 adversarial, 
                 model_supervisor:IModelSupervisor,
                 data:IData,
                 surrogate_model,
                 target_nodes,
                 attack_structure = True,
                 attack_feature = False,
                 **kwargs
                 ) -> None:
        self.attack_structure = attack_structure
        self.attack_feature = attack_feature
        self.adversarial = adversarial
        self.model_supervisor = model_supervisor
        self.data = data
        self.surrogate_model = surrogate_model
    
    def attack(self,node,seed,**kwargs):
        set_random(seed)
        device = kwargs.get('device')
        self.adversarial = self.adversarial.to(device)
        self.adversarial.attack(target_node = node,**kwargs)
        adj = self.data.adj
        feature = self.data.feature
        if self.attack_structure:
            adj = self.adversarial.modified_adj
            self.model_supervisor.update_structure_with_adj(adj)
        if self.attack_feature:
            feature = self.adversarial.modified_features
            self.model_supervisor.update_node_feature(feature)
        
        self.model_supervisor.train_model()

        prediction = self.model_supervisor.get_model_prediction()
        labels = self.data.label
        return (prediction[node] == labels[node]).item()

