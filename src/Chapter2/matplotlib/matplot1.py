import matplotlib.pyplot as plt
import numpy as np

# 2次関数の定義域
# xは-10から10まで0.1ごとに定義
x = np.arange(-5, 5, 0.1)

# 描画する2次関数の方程式
# y = x^2
y = x**2

# グラフをプロット
plt.plot(x, y , label="y = x^2")

# x軸とy軸のラベルを表示
plt.xlabel("X")
plt.ylabel("Y")

# グラフを作成した際に label 引数で指定名前を表示
plt.legend(loc='best')

# プロットしたグラフを表示
plt.show()
