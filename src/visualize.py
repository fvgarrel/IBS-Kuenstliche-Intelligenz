import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

firstData = pd.read_csv("./csv/run-PPO_71-tag-rollout_ep_rew_mean.csv")
secondData = pd.read_csv("./csv/run-PPO_72-tag-rollout_ep_rew_mean.csv")

# Ausreißer nach Interquartilsabstandsmethode entfernen
firstQ1 = np.percentile(firstData['Value'], 25)
firstQ3 = np.percentile(firstData['Value'], 75)
firstIQR = stats.iqr(firstData['Value'])

firstData_clean = firstData[~((firstData['Value'] < firstQ1 - 1.5 * firstIQR) | (firstData['Value'] > firstQ3 + 1.5 * firstIQR))]

secondQ1 = np.percentile(secondData['Value'], 25)
secondQ3 = np.percentile(secondData['Value'], 75)
secondIQR = stats.iqr(secondData['Value'])

secondData_clean = secondData[~((secondData['Value'] < secondQ1 - 1.5 * secondIQR) | (secondData['Value'] > secondQ3 + 1.5 * secondIQR))]

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
x_scatter1 = np.array(firstData_clean['Step'])
y_scatter1 = np.array(firstData_clean['Value'])

x_scatter2 = np.array(secondData_clean['Step'])
y_scatter2 = np.array(secondData_clean['Value'])

plt.figure()
plt.scatter(x_scatter1, y_scatter1, label='Hoch zu niedrig', color='orange')
plt.scatter(x_scatter2, y_scatter2, label='Niedrig zu hoch', color='blueviolet')
plt.xlabel('Episoden Reward')
plt.ylabel('Anzahl in Tsd.')
plt.title('Dynamische Lernrate')
plt.legend()
plt.show()

# Histogram
histo1 = np.array(firstData_clean['Value'])
histo2 = np.array(secondData_clean['Value'])

plt.figure()  # Neue Figure erstellen
plt.xlabel('Episoden Reward')
plt.ylabel('Anzahl in Tsd.')
plt.title('Dynamische Lernrate')
plt.hist(histo1, label='Hoch zu niedrig', color='orange')
plt.hist(histo2, label='Niedrig zu hoch', color='blueviolet')
plt.legend()
plt.show()
