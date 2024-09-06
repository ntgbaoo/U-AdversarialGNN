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
GAMMA_ATTENTION ="gamma_attention"
PROPA_STEP_K="propa_step_k"
LAMBDA1 ="lambda1"
LAMBDA2 ="lambda2"
DROP_NODE_RATE="drop_node_rate"
ORDER = "order"
TEMP = "temp"
LAMBDA ="lambda"


IDX_TRAIN = "idx_train"
IDX_VAL = "idx_val"
IDX_TEST = "idx_test"
#Hyper-parameters' options
SGD ="SGD"
ADAM = "ADAM"
STEP_LR = "StepLR"
GAMMA = "gamma"

#PATH
BEST_CONFIG = "best_config_split_"
PATH_DATA = "data/"
PATH_CONFIG_MODEL = "{}configs/models/".format(PATH_DATA)
PATH_CONFIG_SPLIT = "{}configs/split/".format(PATH_DATA)
PATH_DATASET = "{}dataset".format(PATH_DATA)
PATH_OUTPUT = "output/"
PATH_LOG = "{}log/".format(PATH_OUTPUT)
PATH_RESULT = "{}results/".format(PATH_DATA)
PATH_CACHED_PRED = "{}pred_cache/".format(PATH_DATA)

#Adversarial
NETTACK_NAME = "nettack"
GOTTACK_NAME = "gottack"
FGA_NAME= "fga"
SGA_NAME= "sga"
RND_NAME = "rnd"
GFATTACK_NAME = "gfattack"
IGATTACK_NAME = "igattack"
PGDATTACK_NAME = "pgdattack"
DICE_NAME = "dice"
PRBCD_NAME = "prbcdattack"
EVASION = "evasion"
POSION = "poision"
GOTTACK ="GOttack"




