import os
import time
import faiss
import numpy as np

# 処理開始時刻を記録
start_time = time.time()

# FAISSインデックスの設定
dimension = 512  # ベクトルの次元数
nlist = 100  # クラスタ数
# 量子化器を作成（内積を使用）
quantizer = faiss.IndexFlatIP(dimension)
# IVFフラットインデックスを作成
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# データのルートディレクトリ
root_dir = "/media/terms/2TB_Movie/face_data_backup/data"
# カレントディレクトリを変更
os.chdir(root_dir)

# サブディレクトリのリストを作成
sub_dir_path_list = [
    os.path.join(root_dir, sub_dir)
    for sub_dir in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, sub_dir))
]

# データ格納用のリスト
all_model_data = []
all_name_list = []
all_dir_list = []  # ディレクトリ情報も保存

# 各サブディレクトリからデータを読み込む
for dir in sub_dir_path_list:
    npz_file = os.path.join(dir, "npKnown.npz")
    with np.load(npz_file) as data:
        model_data = data['efficientnetv2_arcface']
        name_list = data['name']
        # データの形状を変更し、L2正規化を行う
        model_data = model_data.reshape((model_data.shape[0], -1))
        faiss.normalize_L2(model_data)
        # データをリストに追加
        all_model_data.append(model_data)
        all_name_list.extend(name_list)
        all_dir_list.extend([dir] * len(name_list))

# データをnumpy配列に変換
all_model_data = np.vstack(all_model_data)

# 量子化器を訓練し、データを追加
index.train(all_model_data)
index.add(all_model_data)

# 類似度が高い要素を検索
k = 10
D, I = index.search(all_model_data, k)

# 結果を保存
with open("output.csv", "a") as f:
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            # コサイン類似度が0.5以上で、同じディレクトリでない場合に出力
            if D[i, j] >= 0.5 and all_dir_list[i] != all_dir_list[I[i, j]]:
                f.write(f"{all_name_list[i]},{all_name_list[I[i, j]]},{D[i, j]}\n")

# 処理時間を計算して出力
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"処理時間: {int(minutes)}分 {seconds:.2f}秒")
