import sys
from keras.datasets import mnist
from PIL import Image
 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
#X_trainの画像化とy_trainの値
#1番目のトレーニングの画像データについて調べてみる
train_no = 8
 
print('訓練画像')
for xs in X_train[train_no]:
    for x in xs:
        sys.stdout.write('%03d ' % x)
    sys.stdout.write('\n')
    
outImg = Image.fromarray(X_train[train_no].reshape((28,28))).convert("RGB")
outImg.save("train.png")
 
print('訓練ラベル(y_train) ＝ %d' % y_train[train_no])
 
 
#X_testの画像化とy_testの値
#1番目のテストの画像データについて調べてみる
test_no = 8
 
print('テスト画像')
for xs in X_test[test_no]:
    for x in xs:
        sys.stdout.write('%03d ' % x)
    sys.stdout.write('\n')
    
outImg = Image.fromarray(X_test[test_no].reshape((28,28))).convert("RGB")
outImg.save("test.png")
 
print('テストラベル(y_test) ＝ %d' % y_test[test_no])