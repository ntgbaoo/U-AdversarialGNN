
"""
Provides the interface for models training supervisor
=======
July 08 2024
"""

class IModelSupervisor:
    def train_model(self):
        pass

    def get_model_output(self):
        pass

    def get_model_prediction():
        pass

    def get_prediction_accuracy(self):
        pass

    def update_structure_with_edgelist(self,adversarial_edgelist):
        pass

    def update_node_feature(self,adversarial_feature):
        pass

    def to_device(device):
        pass


