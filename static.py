
#Model set-up
DEVICE = "device"
IN_CHANNEL = "in_channel"
OUT_CHANNEL = "out_channel"
MODEL = "model"

#Hyper=parameters
NUM_LAYERS = "num_layers"
BATCH_SIZE = "batch_size"
LEARNING_RATE = "lr"
HIDDEN_UNITS ="hidden_unit"
EPOCHS = "num_epochs"
DROPOUT = "drop_out"
PATIENCE = "patience"
SCHEDULER_STATE = "scheduler_state"
AGGREGATION = "aggr"
OPTIMIZER_STATE = "optimizer_state"
SCHEDULER = "scheduler"
OPTIMIZER = "optimizer"
GAMMA_LIST = "gamma_list"
STEP_SIZE_LIST = "step_size_list"
STEP_SIZE = "step_size"
WEIGHT_DECAY = "weight_decay"


#Hyper-parameters' options
SGD ="SGD"
ADAM = "ADAM"
STEP_LR = "StepLR"
GAMMA = "gamma"

#PATH
PATH_DATA = "data/"
PATH_CONFIG_MODEL = "{}configs/models/".format(PATH_DATA)
PATH_CONFIG_SPLIT = "{}configs/split/".format(PATH_DATA)
PATH_DATASET = "{}dataset".format(PATH_DATA)
PATH_OUTPUT = "output/"
PATH_LOG = "{}log/".format(PATH_OUTPUT)