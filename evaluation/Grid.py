from copy import deepcopy 
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from static import *
class Grid:

    '''
    NUM_LAYERS = "num_layers"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "lr"
    HIDDEN_UNITS ="hidden_unit"
    EPOCHS = "num_epochs"
    DROPOUT = "drop_out"
    PATIENCE = "patience"
    SCHEDULER_STATE = "scheduler_state"
    AGGREGATION = "aggregation"
    OPTIMIZER_STATE = "optimizer_state"
    '''
    def __init__(self,config_dict:dict) -> None:
        self.config_dict = config_dict
        self.num_layers = config_dict.get(NUM_LAYERS)
        self.lr = config_dict.get(LEARNING_RATE)
        self.nhid = config_dict.get(HIDDEN_UNITS)
        self.epoch = config_dict.get(EPOCHS)
        self.drop_out = config_dict.get(DROPOUT)
        self.patience = config_dict.get(PATIENCE)
        self.scheduler_state = config_dict.get(SCHEDULER_STATE)
        self.optimizer_state = config_dict.get(OPTIMIZER_STATE)
        self.aggregation = config_dict.get(AGGREGATION)
        self._assert_config_dict()
        self.all_configs = []
    
    def _assert_config_dict(self):
        assert self.num_layers is not None and len(self.num_layers) > 0, "Number of layers must be at least 1 possible value"
        assert self.lr is not None and len(self.lr) > 0, "Learning rate must be at least 1 possible value"
        assert self.nhid is not None and len(self.nhid) > 0, "Learning rate must be at least 1 possible value"
        assert self.epoch is not None and len(self.epoch) > 0, "Learning rate must be at least 1 possible value"
        assert self.aggregation is not None and len(self.aggregation) > 0, "Learning rate must be at least 1 possible value"
        assert self.optimizer_state is not None, "optimizer_state must be at least 1 possible value"

    def init_grid(self):
        self.all_configs = self._update_config(self.all_configs,self.num_layers,NUM_LAYERS)
        self.all_configs = self._update_config(self.all_configs,self.lr,LEARNING_RATE)
        self.all_configs = self._update_config(self.all_configs,self.epoch,EPOCHS)
        if self.drop_out is not None and len(self.drop_out) > 0:
            self.all_configs = self._update_config(self.all_configs,self.drop_out,DROPOUT)

        if self.patience is not None and len(self.patience) > 0:
            self.all_configs = self._update_config(self.all_configs,self.patience,PATIENCE)
        if self.scheduler_state is not None:
            self._update_configs_scheduler()
        self._update_config_optimizer()
        self.all_configs = self._update_config(self.all_configs,self.aggregation,AGGREGATION)
        return self.all_configs
    
        
    
    def _update_config_optimizer(self):
        if self.optimizer_state.get(OPTIMIZER) == ADAM:
            optimizer = [ADAM]
            self.all_configs = self._update_config(self.all_configs,optimizer,OPTIMIZER)
            opti_state_configs = []
            for key in self.optimizer_state:
                if key != OPTIMIZER:
                    schedule_state_configs = self._update_config(schedule_state_configs,self.optimizer_state.get(key),key)
            if len(opti_state_configs) > 0:
                self.all_configs = self._update_config(self.all_configs,opti_state_configs,OPTIMIZER_STATE)

        else:
            raise Exception("Adam {} is not supported".format(self.optimizer_state.get(OPTIMIZER)))

        
    def _update_configs_scheduler(self):
        if self.scheduler_state.get(SCHEDULER) == STEP_LR:
            scheduler_class = [STEP_LR]
            self.all_configs = self._update_config(self.all_configs,scheduler_class,SCHEDULER)
            schedule_state_configs = []
            for key in self.scheduler_state:
                if key != SCHEDULER:
                    schedule_state_configs = self._update_config(schedule_state_configs,self.scheduler_state.get(key),key)
            
            if len(schedule_state_configs) >0 :
                self.all_configs = self._update_config(self.all_configs,schedule_state_configs,SCHEDULER_STATE)
        else:
            raise Exception("Scheduler {} is not supported".format(self.scheduler_state.get(SCHEDULER)))
        
    def _update_config(self,
                    all_configs:list,
                    possible_values: list,
                    hyperparameter: str
                    ):
        new_configs_list = []
        if len(all_configs) == 0:
            for possible_value in possible_values:
                new_config = {hyperparameter:possible_value}
                new_configs_list.append(new_config)
        else:
            old_configs_list = deepcopy(all_configs)
            for possible_value in possible_values:
                for old_config in old_configs_list:
                    new_config = deepcopy(old_config)
                    new_config[hyperparameter] = possible_value
                    new_configs_list.append(new_config)

        return new_configs_list

if __name__ == "__main__":
   config = {
       NUM_LAYERS:[3,5],
       BATCH_SIZE:[16],
       LEARNING_RATE :[0.01,0.001,0.0001],
       HIDDEN_UNITS : [32,64],
       EPOCHS:[10000],
       PATIENCE:[500],
       OPTIMIZER_STATE:{
           OPTIMIZER:ADAM
       },
       SCHEDULER_STATE:{
           SCHEDULER:STEP_LR,
           STEP_SIZE:[50],
           GAMMA:[0.5]
       },
       AGGREGATION: ["sum","average"]
   }

   all_configs = Grid(config).init_grid()
   print(len(all_configs))
   for config in all_configs:
       print(config)


    
