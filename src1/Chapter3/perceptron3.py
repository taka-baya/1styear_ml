# -*- coding: utf-8 -*-

# 演算用にNumPyを、プロット用にmatplotlibをimport
import numpy as np
import matplotlib.pyplot as plt
import random

#内積計算
def f(w, x):
	return np.dot(w, x)

# 誤差計算
def error(out, label):
	err = label - out
	return err

# 重み更新式
def update(weights, err ,x):
	for i in range(len(x)):
		#print("weights :{0} ,err :{1} ,x{2} :{3}".format(weights[i],float(err),i,x[i]))
		weights[i] += 0.0005 * float(err) * weights[i]

# 正答率を調べる
def accuracy(z,l,w):
	act = 0
	for i in range(len(z)):
		ans = np.round(f(z[i], weights))
		if(int(ans) == int(l[i])):
			act= act + 1
	
	return (act/len(z))

# データをシャッフル
def shuffle_data(z,l):
	s = np.array(np.c_[z,l])
	np.random.shuffle(s)
	z, l = np.array_split(s, 2, 1)
	return z ,l

# データの生成
def make_data():
	# A_1のテストデータ
	# x1とy1はそれぞれ平均𝑥=4,𝑦=4の標準偏差1のランダムな数
	x1=np.random.normal(4,sigma,data_num)
	y1=np.random.normal(4,sigma,data_num)

	# A_2のテストデータ
	# x2とy2はそれぞれ平均𝑥=1,𝑦=1の標準偏差1のランダムな数
	x2=np.random.normal(1,sigma,data_num)
	y2=np.random.normal(1,sigma,data_num)

	# 学習データの整形
	# 作成したA_1とA_2の学習データを配列として結合
	z1 = np.c_[x1,y1]
	z2 = np.c_[x2,y2]
	train_z=np.array(np.r_[z1,z2])
	
	# 正解のラベル付け
	label1=np.ones(data_num)
	label2=np.zeros(data_num)
	label_z=np.array(np.r_[label1,label2])

	return train_z, label_z

if __name__ == '__main__':
	data_num = 50 #学習させるデータ数
	sigma = 1 #分散
	act_data=[] #正答率

	# 初期の重み
	weights = np.random.normal(0.07,0.01,2)
	print("重み初期値[w1 w2] : {0}".format(weights))

	for epoch in range(100):
		sumE = 0

		# 新しい学習データの生成
		train_z, label_z =make_data()

		# 学習データの順番をシャッフル
		train_z ,label_z = shuffle_data(train_z ,label_z)

		for p in range(len(train_z)):
			# 誤差値を求める
			e = error(f(train_z[p], weights), label_z[p])
			#print('{0} train_z:{1}  ,weights:{2} ,label_z:{3} ,e:{4}'.format(p, train_z[p], weights, label_z[p], e))

			# 重みの更新
			update(weights, e, train_z[p])

			# 二乗誤差を求める
			sumE += e**2

		# 学習した重みに対して正解率を求める為のテストデータの生成
		train_z, label_z =make_data()
		train_z ,label_z = shuffle_data(train_z ,label_z)

		# 正解率を求める
		act = accuracy(train_z,label_z,weights)
		print('weights :{0}'.format(weights))
		print('epoch : {0}  ,  Square error : {1}  ,  actuary : {2} '.format(epoch ,sumE ,act))

		# 正解率の保存
		act_data.append(act)

		# もし二乗誤差が0ならば学習終了
		if sumE == 0:
			break

	# 学習結果の表示
	plt.xlabel("epoch")
	plt.ylabel("Accuracy(%)")
	plt.plot(act_data)
	plt.show()