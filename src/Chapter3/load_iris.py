import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# irisデータの読み込み
iris = load_iris()

X = iris.data[:, :2]  # 特徴量データの抽出
y = iris.target # ラベルデータの抽出

# プロットするグラフデータの範囲を決めるために、特徴量の最大値・最小値を求める
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# データをプロット
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,edgecolor='k')

# プロット範囲の定義
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 軸のラベルの定義
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()