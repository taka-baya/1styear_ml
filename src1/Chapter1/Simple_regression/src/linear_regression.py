# coding: utf-8
# python version 3.6.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def file_read():
	lst = pd.read_csv("./example_data/orders_date_test.csv").values.tolist()
	return lst

def calc_RA(data):
	x_total = 0
	y_total = 0
	xx_dsp = 0
	yy_dsp = 0
	xy_dsp = 0
	a = [0,0]

	length = len(data)
	data = np.delete(data, 0, 1)

	for i in range(0,length):
		x_total += int(data[i][0])
		y_total += int(data[i][1]) 
	x_ave = x_total/length
	y_ave = y_total/length

	for i in range(0,length):
		xx_dsp += pow(int(data[i][0])-x_ave,2)
		yy_dsp += pow(int(data[i][1])-y_ave,2)
		xy_dsp += (int(data[i][0]) - x_ave) * (int(data[i][1]) - y_ave)

	a[0] = xy_dsp/xx_dsp
	a[1] = y_ave - x_ave * a[0]

	return a

def plot(data,a):
	data = np.delete(data, 0, 1)
	x1 = [0]*(len(data))
	x2 = [0]*(len(data))
	a0 = float(a[0])
	a1 = float(a[1])
	for i in range(0,len(data)):
		x1[i] = int(data[i][0])
		x2[i] = int(data[i][1])
		
	print(a)
	data = np.delete(data, 0, 1)
	print(x1)
	x = np.linspace(24,34)
	#y = np.linspace(50,100,50)
	plt.scatter(x1,x2)
	y1 = a0 * x + a1
	plt.plot(x,y1,"r-")
	plt.xlabel("degree(℃)")
	plt.ylabel("orders(num)")
	plt.show()

def MCC(data,a):
	data = np.delete(data, 0, 1)
	length = len(data)
	y_pre = [0]*length
	y_pre_total = 0
	y_total = 0
	Syy = 0
	Sy_y_ = 0
	Syy_ = 0

	for i in range(0,length):
		y_pre[i] = a[0]*int(data[i][0]) + a[1]
		y_pre_total += a[0]*int(data[i][0]) + a[1]
	y_pre_ave = y_pre_total/len(data)

	for i in range(0,length):
		y_total += int(data[i][1]) 
	y_ave = y_total/length

	for i in range(0,length):
		Syy +=  pow(int(data[i][1])-y_ave,2)
		Sy_y_ += pow(y_pre[i]-y_pre_ave,2)
		Syy_ += (int(data[i][1])-y_ave) * (y_pre[i]-y_pre_ave)

	print(Sy_y_)
	R = Syy_/pow(Syy * Sy_y_,1/2)
	C = pow(R,2)

	print("重相関係数　＝",R)
	print("寄与率　＝",C)

	
def write_csv(index):
	length = len(index)
	data = np.delete(index, 0, 1)
	x = [0]*(length + 2)
	y = [0]*(length + 2)
	x_total = 0
	y_total = 0
	xx_dsp = [0]*(length + 2) #xの偏差平方和
	yy_dsp = [0]*(length + 2) #yの偏差平方和
	xy_dsp = [0]*(length + 2) #xとyの偏差積和
	xx_dsp_total = 0 #xの偏差平方和の和
	yy_dsp_total = 0 #yの偏差平方和の和
	xy_dsp_total = 0 #xとyの偏差積和の和
	a = [0,0] #回帰式の第一次項と切片

	y_pre = [0]*(length + 2) 
	y_pre_total = 0
	y_total = 0
	Syy = [0]*(length + 2)
	Sy_y_ = [0]*(length + 2)
	Syy_ = [0]*(length + 2)
	Syy_total = 0
	Sy_y__total = 0
	Syy__total = 0

	for i in range(0,length):
		x[i] = int(data[i][0])
		y[i] = int(data[i][1]) 
		x_total += int(data[i][0])
		y_total += int(data[i][1]) 
	x_ave = x_total/length
	y_ave = y_total/length

	x.append(x_total)
	y.append(y_total)
	x.append(x_ave)
	y.append(y_ave)

	for i in range(0,length):
		xx_dsp[i] = pow(int(data[i][0])-x_ave,2)
		yy_dsp[i] = pow(int(data[i][1])-y_ave,2)
		xy_dsp[i] = (int(data[i][0]) - x_ave) * (int(data[i][1]) - y_ave)
		xx_dsp_total += pow(int(data[i][0])-x_ave,2)
		yy_dsp_total += pow(int(data[i][1])-y_ave,2)
		xy_dsp_total += (int(data[i][0]) - x_ave) * (int(data[i][1]) - y_ave)

	a[0] = xy_dsp_total/xx_dsp_total
	a[1] = y_ave - x_ave * a[0]
	print("a:{0} b{1}".format(a[0],a[1]))
	
	xx_dsp.append(xx_dsp_total)
	yy_dsp.append(yy_dsp_total)
	xy_dsp.append(xy_dsp_total)
	xx_dsp.append(0)
	yy_dsp.append(0)
	xy_dsp.append(0)

	for i in range(0,length):
		y_pre[i] = a[0]*int(data[i][0]) + a[1]
		y_pre_total += a[0]*int(data[i][0]) + a[1]
	y_pre_ave = y_pre_total/len(data)

	y_pre.append(y_pre_total)
	y_pre.append(y_pre_ave)

	for i in range(0,length):
		Syy[i] =  pow(int(data[i][1])-y_ave,2)
		Sy_y_[i] = pow(y_pre[i]-y_pre_ave,2)
		Syy_[i] = (int(data[i][1])-y_ave) * (y_pre[i]-y_pre_ave)

		Syy_total +=  pow(int(data[i][1])-y_ave,2)
		Sy_y__total += pow(y_pre[i]-y_pre_ave,2)
		Syy__total += (int(data[i][1])-y_ave) * (y_pre[i]-y_pre_ave)

	Syy.append(Syy_total)
	Sy_y_.append(Sy_y__total)
	Syy_.append(Syy__total)
	Syy.append(0)
	Sy_y_.append(0)
	Syy_.append(0)

	data = np.c_[x, y, xx_dsp, yy_dsp, y_pre, Syy, Sy_y_ ,Syy_]
	print(data)
	df = pd.DataFrame(np.c_[x, y, xx_dsp, yy_dsp, xy_dsp, y_pre, Syy, Sy_y_ ,Syy_])
 
	# CSV ファイル (employee.csv) として出力
	df.to_csv("calculate.csv")


if __name__ == '__main__':
	data = file_read()
	a = calc_RA(data)
	MCC(data,a)
	plot(data,a)
	write_csv(data)