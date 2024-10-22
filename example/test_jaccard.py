if __name__ == '__main__':
    import sys
    import os
    
    # Edit path to import from different module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utility.config import args
    from utility.util import set_random, logger, init_logger,mkdirs,prepare_dir
    from evaluation import ModelSupervisor
    from data_loader import DataLoader
    from utility import ConfigHandler
    from static import *
    from defense_model.improving_graph import JaccardPurification

    args.dataset = "cora"
    
    data,split = DataLoader.load(args.dataset)
    print(data)
    purification = JaccardPurification()
    modified_data = purification(data=data,inplace=False)
    print(modified_data)
