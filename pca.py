import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("your_dataset.csv")

X = pd.get_dummies(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained Variance:", pca.explained_variance_ratio_)

plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()
