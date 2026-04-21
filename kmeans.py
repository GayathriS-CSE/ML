import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("your_dataset.csv")

X = pd.get_dummies(df)

model = KMeans(n_clusters=3)
clusters = model.fit_predict(X)

print("Cluster Labels:", clusters)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=clusters)
plt.show()
