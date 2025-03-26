import sys
import os
import statistics as stats

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility import ConfigHandler

results = []
model = "SGC"
config = ConfigHandler.load_model_config(model_type=model,config_name="best_config",dataset="squirrel",split=0)
for index in range(5):
    config = ConfigHandler.load_model_config(model_type=model,config_name="best_config",dataset="squirrel",split=index)
    results.append(config['performance'])

print(f"Mean:{ stats.mean(results)}")
print(f"std:{ stats.stdev(results)}")