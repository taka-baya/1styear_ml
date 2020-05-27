import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# ログ出力するための定義
from logging import getLogger
import logging.config
import os

## About the logging
#     """detail infomation please show https://docs.python.org/3/howto/logging.html
# The numeric values of logging levels are given in the following table.
# 
#   LEVEL  | Numeric value |  
# ---------+---------------+
# CRITICAL | 50            |
# ERROR    | 40            |
# WARNING  | 30            |
# INFO     | 20            |
# DEBUG    | 10            |
# NOTSET   | 0             |
#
# Please see "logging.conf" for output log settings

# ログ設定ファイルからログ設定を読み込み
OUT_LOG_FOLDER = "./log"
if not os.path.exists(OUT_LOG_FOLDER):
        os.makedirs(OUT_LOG_FOLDER)

logging.config.fileConfig('./log/logging.conf')
logger = getLogger(__name__)

# torch.nn.Moduleによるネットワークモデルの定義
class Mnist_net(nn.Module):
    def __init__(self, INPUT_FEATURES=784, OUTPUT_RESULTS=10):
        # 定数（モデル定義時に必要となるもの）
        self.INPUT_FEATURES = INPUT_FEATURES# 入力（特徴）の数： 2
        self.LAYER1_NEURONS = 256      # ニューロンの数： 256
        self.LAYER2_NEURONS = 128      # ニューロンの数： 128
        self.OUTPUT_RESULTS = OUTPUT_RESULTS# 出力結果の数： 10

        super(Mnist_net, self).__init__()
        # 入力層→中間層①：1つ目のレイヤー（layer）
        self.layer1 = nn.Linear(
            self.INPUT_FEATURES,                # 入力ユニット数：784
            self.LAYER1_NEURONS)                # 次のレイヤーの出力ユニット数：256

        # 中間層①→中間層②：2つ目のレイヤー（layer）
        self.layer2 = nn.Linear(
            self.LAYER1_NEURONS,                # 入力ユニット数：256
            self.LAYER2_NEURONS)                # 次のレイヤーへの出力ユニット数：128

        # 中間層②→出力層：2つ目のレイヤー（layer）
        self.layer_out = nn.Linear(
            self.LAYER2_NEURONS,                # 入力ユニット数：128
            self.OUTPUT_RESULTS)                # 出力結果への出力ユニット数：10

    def forward(self, x):
        # フィードフォワードを定義
        # 「出力＝活性化関数（第n層（入力））」の形式で記述する
        x = x.view(-1, 784) # 28×28の配列を784の一次元配列に変換
        x = F.relu(self.layer1(x))  # 活性化関数はreluとして定義
        x = F.relu(self.layer2(x))  # 同上 
        x = F.softmax(self.layer_out(x), dim=1)  # 出力する際はsoftmaxを使用
        return x

"""
init_parameters関数
> パラメーター（重みやバイアス）の初期化を行う関数の定義
"""
def init_parameters(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight) # 重みを「一様分布のランダム値」に初期化
        layer.bias.data.fill_(0.0)            # バイアスを「0」に初期化

"""
Get_mnist_data関数
> Mnistのデータセットをダウンロードしてくる
batch_size          # バッチサイズ
"""
def Get_mnist_data(batch_size):
    # Totenser関数でtensor化
    # Normalise関数で正規化(-1.0〜1.0をとるような値として変換)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))])
    
    # 変数datasetにMNISTのデータセットを格納
    # ダウンロードしたデータセットは、./dataフォルダに配置される。
    # 訓練用のデータセットを変数train_datasetに格納
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform, target_transform=None)
    # 訓練用のデータセットをバッチサイズごとに分割して変数trainloaderに格納
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # テスト(検証)用のデータセットを変数test_datasetに格納
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform, target_transform=None)
    # テスト(検証)用のデータセットをバッチサイズごとに分割して変数testloaderに格納
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # バッチサイズに分割した訓練用のデータセットとテスト(検証)用のデータセット返す
    return trainloader, testloader

"""
・train_step関数
> 学習途中のモデルに対し学習データを用いて損失値と正答率を計算する。
> 損失値から学習モデルのパラメータ(重み)などの更新を行う。
train_X                # 入力画像データ(訓練データ)
train_y                # 正解ラベル(訓練データ)
model                  # ネットワークモデル
optimizer              # 定義した最適化手法
criterion              # 定義した損失関数
"""
def train_step(train_X, train_y, model, optimizer, criterion):
    # 訓練モードに設定
    model.train()

    # フィードフォワードで出力結果を取得
    pred_y = model(train_X) # 出力結果

    logger.debug("学習中のモデルの出力結果(訓練データ) : \n {}".format(pred_y))

    # 出力結果と正解ラベルから損失を計算し、勾配を求める
    optimizer.zero_grad()   # 勾配を0で初期化（※累積してしまうため要注意）
    loss = criterion(pred_y, train_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得
    loss.backward()   # 逆伝播の処理として勾配を計算（自動微分）

    logger.debug("training loss : \n {}".format(loss.item()))

    # 勾配を使ってパラメーター（重みとバイアス）を更新
    optimizer.step()  # 指定されたデータ分の最適化を実施

    # softmaxの出力結果をonehotからintegerに戻す。(正解かどうか検証するため)
    _, pred_y_ = torch.max(pred_y.data, 1)
    pred_y_np = pred_y_.numpy() #integerに戻した
    train_y_np = train_y.numpy()

    with torch.no_grad(): # 勾配は計算しないモードにする
        acc = (pred_y_np == train_y_np).sum()     # 正解数を計算する
        
    # 損失と正解数をタプルで返す
    return (loss.item(), acc.item())  # ※item()=Pythonの数値

"""
・valid_step関数
学習途中のモデルに対してテストデータを用いて損失値と正答率を計算する。
valid_X                # 入力画像データ(テストデータ)
valid_y                # 正解ラベル(テストデータ)
model                  # ネットワークモデル
criterion              # 定義した損失関数
"""
def valid_step(valid_X, valid_y, model, criterion):
    # 評価モードに設定（※dropoutなどの挙動が評価用になる）
    model.eval()
    
    # フィードフォワードで出力結果を取得
    pred_y = model(valid_X) # 出力結果

    logger.debug("学習中のモデルの出力結果(検証データ) : \n {}".format(pred_y))
    
    # 出力結果と正解ラベルから損失を計算
    loss = criterion(pred_y, valid_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得
    # ※評価時は勾配を計算しない

    logger.debug("test loss : \n {}".format(loss.item()))

    # onehotラベルをintegerに戻す
    _, pred_y = torch.max(pred_y.data, 1)
    pred_y = pred_y.numpy()
    valid_y = valid_y.numpy()

    # 正解数を算出
    with torch.no_grad(): # 勾配は計算しないモードにする
        acc = (pred_y == valid_y).sum() # 正解数を合計する

    # 損失と正解数をタプルで返す
    return (loss.item(), acc.item())  # ※item()=Pythonの数値

"""
・training関数
学習を実行する関数
loader_train                # 入力データ
loader_valid                # 正解ラベル
model                       # ネットワークモデル
optimizer                   # 定義した最適化手法
criterion                   # 定義した損失関数
"""
def training(loader_train, loader_valid, model, optimizer, criterion):
    # 学習の前にパラメーター（重みやバイアス）を初期化する
    model.apply(init_parameters)

    # 定数（学習／評価時に必要となるもの）
    EPOCHS = 10             # エポック数： 100

    # 変数（学習／評価時に必要となるもの）
    train_loss = 0.0           # 「訓練」用の平均「損失値」
    train_acc = 0.0            # 「訓練」用の平均「正解率」
    val_loss = 0.0             # 「評価」用の平均「損失値」
    val_acc = 0.0              # 「評価」用の平均「正解率」

    # 損失の履歴を保存するための変数
    train_history = []
    valid_history = []

    for epoch in range(EPOCHS):
        # forループ内で使う変数と、エポックごとに値リセット
        total_loss = 0.0     # 「訓練」時における累計「損失値」
        total_acc = 0.0      # 「訓練」時における累計「正解数」
        total_val_loss = 0.0 # 「評価」時における累計「損失値」
        total_val_acc = 0.0  # 「評価」時における累計「正解数」
        total_train = 0      # 「訓練」時における累計「データ数」
        total_valid = 0      # 「評価」時における累計「データ数」

        # train_x：28×28の訓練画像データの値
        # train_y：28×28の訓練画像データに対する正解のラベル。
        for train_x, train_y in loader_train:
            
            # 【重要】1ミニバッチ分の「訓練」を実行
            loss, acc = train_step(train_x, train_y, model, optimizer, criterion)

            # 取得した損失値と正解率を累計値側に足していく
            total_loss += loss          # 訓練用の累計損失値
            total_acc += acc            # 訓練用の累計正解数
            total_train += len(train_y) # 訓練データの累計数
        
        # valid_x：28×28のテスト画像データの値
        # tvalid_y：28×28のテスト画像データに対する正解のラベル。
        for valid_x, valid_y in loader_valid:

            # 【重要】1ミニバッチ分の「評価（精度検証）」を実行
            val_loss, val_acc = valid_step(valid_x, valid_y, model, criterion)

            # 取得した損失値と正解率を累計値側に足していく
            total_val_loss += val_loss  # 評価用の累計損失値
            total_val_acc += val_acc    # 評価用の累計正解数
            total_valid += len(valid_y) # 訓練データの累計数

        # ミニバッチ単位で累計してきた損失値や正解率の平均を取る
        train_loss = total_loss                 # 訓練用の損失値
        train_acc = total_acc / total_train         # 訓練用の正解率
        val_loss = total_val_loss        # 訓練用の損失値
        val_acc = total_val_acc / total_valid # 訓練用の正解率

        logger.info("{}EPOCH , train loss:{} , train acc:{} , test loss:{} , test acc:{}".format(epoch,train_loss,train_acc,val_loss,val_acc))

        # グラフ描画のために損失の履歴を保存する
        train_history.append(train_loss)
        valid_history.append(val_loss)
    
    # 本ソースと同じディレクトリ上にmnist_modelという名前で、学習したモデルを保存
    torch.save(model.state_dict(), "./mnist_model")

if __name__ == "__main__":
    batch_size = 128 # バッチサイズ: 128
    LEARNING_RATE = 0.001   # 学習率： 0.001
    REGULARIZATION = 0  # 正則化率： 0.0

    logger.info("---------start training------------")

    # 学習データをダウンロード
    # loader_train：学習用データ
    # loader_valid：テスト用データ
    loader_train, loader_valid = Get_mnist_data(batch_size)

    # 変数modelにMnist_netクラスのネットワークモデルを定義
    model = Mnist_net()

    logger.info(f'model shape :' \
        f'{model}')

    # オプティマイザを作成（パラメーターと学習率も指定）
    optimizer = optim.AdamW(         # 最適化アルゴリズムに「AdamW」を選択
        model.parameters(),          # 最適化で更新対象のパラメーター（重みやバイアス）
        lr=LEARNING_RATE)            # 更新時の学習率
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()  # 損失関数：交差エントロピー

    # 学習の実行
    training(loader_train, loader_valid, model, optimizer, criterion)

