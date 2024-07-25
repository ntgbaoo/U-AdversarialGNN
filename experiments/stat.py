import statistics as stat
import numpy as np
data_latest = [0.825,0.95,0.825,0.875,0.975]
mean_latest = np.mean(data_latest)
std_latest = np.std(data_latest)

print(mean_latest, std_latest)