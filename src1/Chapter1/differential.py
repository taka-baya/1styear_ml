import matplotlib.pyplot as plt
import numpy as np

def deltax(x):
	delta = 0.0001
	pdx = x + delta
	mdx = x
	return pdx ,mdx

# 6×4サイズのFigureを作成
fig = plt.figure()

# FigureにAxesを１つ追加
ax = fig.add_subplot(111)

x = 1

x1 ,x2 = deltax(x)
y1 = x1**2
y2 = x2**2

print("x1:{0}".format(x1))
print("x2:{0}".format(x2))
print("y1:{0}".format(y1))
print("y2:{0}".format(y2))

a = (y2 - y1)/(x2 - x1)
b = ((y2 - y1)/(x2 - x1))*x1*(-1) + y1

print(a)
print(b)

ax.grid()

ax.set_xlabel("x", fontsize = 14)
ax.set_ylabel("y", fontsize = 14)

# x軸の範囲を設定
ax.set_xlim(-1, 2)
ax.set_ylim(0, 2)


# -5～5まで1刻みのデータを作成
x = np.arange(-1, 2,0.1)

# 直線の式を定義
y = a*x + b

ax.plot(x, y, color = "blue")

y = x**2

ax.plot(x, y, color = "red")

plt.show()
