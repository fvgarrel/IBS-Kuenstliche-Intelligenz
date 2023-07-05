import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource

import seaborn as sns
from scipy import stats
import tensorflow_datasets as tfds
import tensorboard as tb

data = pd.read_csv("./PPO_tensorboard/csv/run-PPO_1-tag-rollout_ep_rew_mean.csv")

# Boxplot
data['Value'].plot(kind='box')
plt.title("...")
plt.xlabel("xxx")
plt.ylabel("yyy")
plt.show()

# Scatter
x_scatter = np.array(data['Step'])
y_scatter = np.array(data['Value'])
plt.scatter(x_scatter, y_scatter)
plt.title("...")
plt.xlabel("xxx")
plt.ylabel("yyy")
plt.show()

# Histogram
x_histo = np.array(data['Value'])
plt.hist(x_histo)
plt.title("...")
plt.show()

# Hillshade
