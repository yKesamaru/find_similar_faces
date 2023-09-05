

![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/顔データセットの間違い探し_2.png)


## はじめに
深層学習におけるモデル学習において、データセットのクレンジングは重要な作業です。
顔認証システムにおいてのデータセットのクレンジングとは、「人物Aの顔画像ファイルが、間違いなく人物Aのフォルダーに存在しているか」と定義できます。
このクレンジング作業は、ある程度自動化していますが、最終的には目視で確認する必要があります。
なかには知っている人物もありますが、大部分は知らない人物です。
スクレイピング対象の人物名がマイナーな場合（仮に人物Aとします）、同じ名字の有名人（人物B）がヒットしてしまうこともあります。
有名人と言っても私は知らないので、顔画像枚数の多い人物Bを、人物Aのフォルダーに配置してしまうかもしれません。
人物Aのフォルダーには人物Aの顔画像ファイルが存在し、人物Bのフォルダーにも人物Aの顔画像ファイルが存在することになってしまいます。
この状態は、モデル学習において、大きな悪影響を及ぼします。

そこで既存の顔学習モデルを使用して、各フォルダーの顔画像ファイルと、他のフォルダーに存在する顔画像ファイルとのコサイン類似度を計算し、類似度が高いものを抽出します。

### 例
![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/g869.png)
この写真は佐藤真知子アナウンサーと佐藤梨那アナウンサーです。お互いのフォルダーに、お二人の写真が混在していました。自動化の網からこぼれ落ちたこのような間違いを発見することが大事です。

# 手順
最初に、ソースコードを確認しましょう。
```python
import csv
import os
import sys
from typing import List

import cupy as cp
import dlib
import numpy.typing as npt
from tqdm import tqdm

# モジュールのパスを追加
sys.path.append("/home/terms/bin/FACE01_IOT_dev")
from face01lib.api import Dlib_api
from face01lib.utils import Utils  # type: ignore

Utils_obj = Utils()
api_obj = Dlib_api()

def calculate_cosine_similarity(api_obj, file_path1, file_path2):
    encoding_list = []
    for face_path in [file_path1, file_path2]:
        img = dlib.load_rgb_image(face_path)  # type: ignore
        face_locations: List = api_obj.face_locations(img, mode="cnn")
        face_encodings: List[npt.NDArray] = api_obj.face_encodings(
            deep_learning_model=1,
            resized_frame=img,
            face_location_list=face_locations
        )
        encoding_list.append(face_encodings[0])

    emb0 = encoding_list[0].flatten()
    emb1 = encoding_list[1].flatten()
    cos_sim = cp.dot(emb0, emb1) / (cp.linalg.norm(emb0) * cp.linalg.norm(emb1))
    return cos_sim

def generate_combinations(api_obj):
    os.chdir("/media/terms/2TB_Movie/face_data_backup/")
    parent_dir = "data"
    sub_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    CPU_cnt = 0
    # 保存先のCSVファイルを開く
    with open('output.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 各サブディレクトリに対して処理（tqdmで進捗を表示）
        for i, sub_dir1 in enumerate(tqdm(sub_dirs, desc="Processing directories")):
            png_files1 = [os.path.join(sub_dir1, fname) for fname in os.listdir(sub_dir1) if fname.endswith('.png')]
            # 残りのサブディレクトリに対して処理
            for sub_dir2 in sub_dirs[i+1:]:
                png_files2 = [os.path.join(sub_dir2, fname) for fname in os.listdir(sub_dir2) if fname.endswith('.png')]
                # サブディレクトリが異なる場合のみ組み合わせを作成
                for file1 in png_files1:
                    for file2 in png_files2:
                        # CPU温度が上がるので、対策
                        CPU_cnt += 1
                        if CPU_cnt % 2 == 0:
                            # CPU温度が72度を超えていたら待機
                            Utils_obj.temp_sleep()
                            CPU_cnt = 0
                        try:
                            cos_sim = calculate_cosine_similarity(api_obj, file1, file2)
                            # CSVに出力
                            if cos_sim >= 0.4:
                                csv_writer.writerow([file1, file2, cos_sim])
                        except Exception as e:
                            print(e)
                            continue

if __name__ == "__main__":
    generate_combinations(api_obj)
```

この`generate_combinations`関数は、顔画像のペアに対するコサイン類似度（cosine similarity）を計算し、CSVファイルに結果を保存します。


- **サブディレクトリの取得**: `parent_dir`が"data"として設定され、その下のサブディレクトリ（顔のカテゴリー/クラス）を`sub_dirs`リストに格納。

- **サブディレクトリのループ**: 各サブディレクトリ`sub_dir1`に対して、そのサブディレクトリ内の`.png`ファイル（顔画像）をリスト`png_files1`に格納。

- **顔画像ペアの生成**: さらに`sub_dir1`と異なる他のサブディレクトリ（`sub_dir2`）をループし、その中の`.png`ファイルを`png_files2`に格納し、`png_files1`と`png_files2`から顔画像のペアを作成。

- **コサイン類似度の計算**: 各顔画像ペアに対して、`calculate_cosine_similarity`関数を用いてコサイン類似度を計算。
  
- **CPU温度の管理**: CPUの温度が上がるので、5回ごとに`Utils_obj.temp_sleep()`関数を呼び出し、CPU温度を管理。

- **CSVへの出力**: 計算したコサイン類似度が0.4以上の場合、その顔画像ペアと類似度をCSVファイルに書き込み。

# 出力結果
```bash
data/風間杜夫/風間杜夫_0uGH.jpg.png_default.png_0.png_0_align_resize.png,data/高橋光/高橋光_fssJ.jpg.png_default.png_0.png_0_align_resize.png,-0.0017361037
data/風間杜夫/風間杜夫_0uGH.jpg.png_default.png_0.png_0_align_resize.png,data/高橋光/高橋光_7ySb.jpg.png_default.png_0.png_0_align_resize.png,-0.076728545
data/風間杜夫/風間杜夫_0uGH.jpg.png_default.png_0.png_0_align_resize.png,data/高橋光/高橋光_JGlT.jpg.png_default.png_0.png_0_align_resize.png,0.026699446
（...略）
```
これを上位からソートして取り出します。
```bash
sort -t',' -k3,3 -g -r output.csv | head -n 1000 > sorted_output.csv
```
`pandas`を用いた場合は以下になります。
```python
import pandas as pd

# CSVを読み込む（必要な場合は、dtypeオプションでデータ型を指定する）
df = pd.read_csv("output.csv", header=None)

# 3列目で降順ソート
df.sort_values(by=2, ascending=False, inplace=True)

# 上位1000行を新しいCSVに保存
df.head(1000).to_csv("sorted_output.csv", index=False, header=None)
```
ただし、非常に行数が大きいため、`sort`や`head`コマンドを用いたほうがメモリ不足になりません。

たとえば、先頭10件を確認すると、以下のようになります。
```bash
data/風間杜夫/風間杜夫_qA6x.jpg_default.png.png_0.png_0_align_resize.png,data/砂川啓介/砂川啓介_Tc71.jpg.png.png_0_align_resize_default.png,0.3586308
data/風間杜夫/風間杜夫_yee0.jpg_default.png.png_0.png_0_align_resize.png,data/塚本晋也/塚本晋也_YhoA.jpeg.png.png_0.png_0_align_resize.png,0.3468275
data/風間杜夫/風間杜夫_qA6x.jpg_default.png.png_0.png_0_align_resize.png,data/砂川啓介/砂川啓介_Mrfe..png.png.png_0_align_resize_default.png,0.34644827
data/風間杜夫/風間杜夫_qA6x.jpg_default.png.png_0.png_0_align_resize.png,data/砂川啓介/砂川啓介_V5lB..png_0_align_resize_default.png,0.34484732
data/風間杜夫/風間杜夫_5gTs.jpg.png_default.png_0.png_0_align_resize.png,data/生田智子/生田智子_gUE1.jpg.png_0.png_0_align_resize.png,0.3317447
data/風間杜夫/風間杜夫_5gTs.jpg.png_default.png_0.png_0_align_resize.png,data/生田智子/生田智子_0bIT.jpg.png.png.png_0.png_0_align_resize.png,0.32970083
data/風間杜夫/風間杜夫_yee0.jpg_default.png.png_0.png_0_align_resize.png,data/塚本晋也/塚本晋也_knVt.jpg.png.png_0.png_0_align_resize.png,0.32865226
data/風間杜夫/風間杜夫_5gTs.jpg.png_default.png_0.png_0_align_resize.png,data/生田智子/生田智子_LLKl.jpg.png.png.png_0.png_0_align_resize.png,0.31950635
data/風間杜夫/風間杜夫_qA6x.jpg_default.png.png_0.png_0_align_resize.png,data/砂川啓介/砂川啓介_lfo5.jpg_default..png.png.png_0.png_0_align_resize.png,0.31697136
data/風間杜夫/風間杜夫_yFg1.jpg_default.png.png_0.png_0_align_resize.png,data/トータス松本/トータス松本_default.png.png.png_0_align_resize.png,0.31625932
```
# まとめ
このように、顔画像のペアに対するコサイン類似度を計算することで、間違いがあるかどうかを確認できます。
実際には非常に処理時間が長いため、今現在も処理中です。コサイン類似度が高い組み合わせについては目視で確認し、WEBで誰が誰なのかを調べて、間違いがあれば訂正します。

以上です。ありがとうございました。
