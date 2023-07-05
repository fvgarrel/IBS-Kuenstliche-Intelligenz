import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

firstData = pd.read_csv("./csv/run-PPO_71-tag-rollout_ep_rew_mean.csv")
secondData = pd.read_csv("./csv/run-PPO_72-tag-rollout_ep_rew_mean.csv")

# Ausreißer nach Interquartilsabstandsmethode entfernen
firstQ1 = firstData.quantile(q=.25)
firstQ3 = firstData.quantile(q=.75)


# Boxplot
plt.subplot(1, 2, 1)
firstData['Value'].plot(kind='box')
plt.title("Hoch zu niedrig")

plt.subplot(1, 2, 2)
secondData['Value'].plot(kind='box')
plt.title("Niedrig zu hoch")

plt.figure()
data = [firstData['Value'], secondData['Value']]
plt.boxplot(data)
plt.xticks([1, 2], ['Hoch zu niedrig', 'Niedrig zu hoch'])
plt.title("Dynamische Lernrate")

plt.show()

# Scatter
x_scatter1 = np.array(firstData['Step'])
y_scatter1 = np.array(firstData['Value'])

x_scatter2 = np.array(secondData['Step'])
y_scatter2 = np.array(secondData['Value'])

plt.figure()
plt.scatter(x_scatter1, y_scatter1, label='Hoch zu niedrig')
plt.scatter(x_scatter2, y_scatter2, label='Niedrig zu hoch')

plt.legend()
plt.show()

# Histogram
histo1 = np.array(firstData['Value'])
histo2 = np.array(secondData['Value'])

plt.figure()  # Neue Figure erstellen
plt.hist(histo1, label='Hoch zu niedrig')
plt.hist(histo2, label='Niedrig zu hoch')

plt.legend()
plt.show()