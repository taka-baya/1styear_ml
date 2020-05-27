import pandas as pd

# データセットの読み込み
df = pd.read_csv("orders_date_test.csv")

# 読み込んだcsvファイルの表示
print(df)

# 読み込んだデータの最初の3件を表示
print(df.head(3))

# dateの列のみを取り出し
print(df['date'])

# 行と列の型を取得
print(df.shape)

# test.csvファイルとしてファイルを保存
df.to_csv('test.csv')

# 要素degreeが30以上で、かつ要素ordersが60以下のデータを抽出
squeeze = (df['degree'] < 30) & (df['orders'] < 60)
print(df[squeeze])

# degree 列の値を昇順に並べ替え
df_up = df.sort_values(by='degree',ascending=True)
print(df_up)

# degree 列の値を降順に並べ替え
df_down = df.sort_values(by='degree',ascending=False)
print(df_down)

# 平均
print(df.mean())

# 分散
print(df.var())

# 相関係数の算出
print(df.corr())