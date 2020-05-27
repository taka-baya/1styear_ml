import numpy as np
import chainer
from keras.datasets import mnist
from sklearn.model_selection import train_test_split # データセットを分割するモジュールの読み込み
import chainer.links as L # パラメータを持つ関数 (層)：リンク
import chainer.functions as F # パラメータを持たない関数：ファンクション(ReLU,Sigmoid)
from chainer import Sequential
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages #PDFとして保存

# print chainer version infomation
print(chainer.print_runtime_info())


# net としてインスタンス化
NETWORK_INPUT = 784
NETWORK_HIDDEN = 100
NETWORK_OUTPUT = 10
# ネットワーク層の定義
net = Sequential(
    L.Linear(NETWORK_INPUT, NETWORK_HIDDEN), F.relu,
    L.Linear(NETWORK_HIDDEN, NETWORK_HIDDEN), F.relu,
    L.Linear(NETWORK_HIDDEN, NETWORK_OUTPUT)
)

# 学習率の定義
lr=0.01

# 更新する学習率
lr_add = 0.01

# 学習率の更新回数
lr_num = 10

# 学習データとテストデータの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) # 2次元配列を1次元に変換
x_test = x_test.reshape(10000, 784) # 2次元配列を1次元に変換
x_train = x_train.astype('float32') # int型をfloat32型に変換
x_test = x_test.astype('float32') # int型をfloat32型に変換
t_train = t_train.astype('int32') # 一応int型に
t_test = t_test.astype('int32') # 一応int型に
x_train /= 255 # [0-255]の値を[0.0-1.0]に変換
x_test /= 255 # [0-255]の値を[0.0-1.0]に変換

# 最適化手法としてAdamを使用、netに宣言
optimizer = chainer.optimizers.Adam(lr)
optimizer.setup(net)

NUM_EPOCH = 100 #エポック数
EPOCH_TRAIN_NUM = 1000 #1エポックに対しての学習データ数
NUM_BATCHSIZE = 16 #バッチサイズ
iteration = 0 #試行回数

# 学習ログの保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}


for epoch in range(NUM_EPOCH):

    # 全データセットからランダムにEPOCH_TRAIN_NUM個抽出
    order = np.random.choice(x_train.shape[0], EPOCH_TRAIN_NUM)

    # 各バッチ毎の目的関数の出力と分類精度の保存用
    loss_list = []
    accuracy_list = []

    # n_batchsizeごとにバッチ処理を実行する
    for i in range(0, len(order), NUM_BATCHSIZE):
        # バッチを準備
        # x_train_batchには各入力データ(784)が二次元配列として格納されている
        # t_train_batchには正解ラベル(0〜9)が格納
        index = order[i:i+NUM_BATCHSIZE]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]

        # 0〜9のラベルのうちどの分類に属するかを予測し出力する
        y_train_batch = net(x_train_batch)

        # softmax_cross_entropy()を用いて予測値と正解値の誤差を求める
        # accuracy()で分類精度を求める。
        loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        # 正解率と損失率をデータとして保持
        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)

        # 各更新対象モデルのcleargrad()による勾配の初期化
        # lossからのbackward() による誤差逆伝搬
        net.cleargrads()
        loss_train_batch.backward()

        # optimizer.update()によるw各モデルの勾配更新
        optimizer.update()

        # カウントアップ
        iteration += 1

    # 訓練データに対する目的関数の出力と分類精度を集計
    # np.meanは配列要素の平均をとる
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # 1エポック終えたら、検証データで評価を行う
    # 検証データで予測値を出力
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = net(x_test)

    # 目的関数を適用し、分類精度を計算
    # softmax_cross_entropy()を用いて予測値と正解値の誤差を求める
        # accuracy()で分類精度を求める。
    loss_val = F.softmax_cross_entropy(y_val, t_test)
    accuracy_val = F.accuracy(y_val, t_test)

    # 結果の表示
    print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
        epoch, iteration, loss_train, loss_val.array))

    # 学習過程のログを保存
    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)

# PDFとして出力
pdf = PdfPages('mnist.pdf')

# 目的関数の出力 (loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(results_train['loss'], label='train')  # label で凡例の設定
plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
# plt.show()  # 凡例の表示
pdf.savefig()
plt.clf()

# 分類精度 (accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(results_train['accuracy'], label='train')  # label で凡例の設定
plt.plot(results_valid['accuracy'], label='valid')  # label で凡例の設定
plt.legend(loc="upper left")
# plt.show()  # 凡例の表示
pdf.savefig()
plt.clf()

# 出力したPDFを閉じる
pdf.close()