import os
import time
import faiss
import numpy as np
import sys
# FACE01ライブラリのインポート
sys.path.insert(1, '/home/user/bin/FACE01_IOT_dev')
from face01lib.api import Dlib_api

api = Dlib_api()

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
root_dir = "/media/user/2TB_Movie/face_data_backup/data"
# カレントディレクトリを変更
os.chdir(root_dir)

# 顔写真をロード
face_image = api.load_image_file("/home/user/ドキュメント/find_similar_faces/assets/woman2.png")
# face_image = api.load_image_file("/home/user/ドキュメント/find_similar_faces/assets/woman.png")
face_location = api.face_locations(face_image, mode="cnn")
face_encoding = api.face_encodings(
    deep_learning_model=1,
    resized_frame=face_image,
    face_location_list=face_location,
)
face_encoding = np.array(face_encoding[0][0]).reshape(1, 512)

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
# # 量子化器を訓練し、データを追加
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)  # 追加; IVFインデックスを作成

# 量子化器を訓練し、データを追加
index.train(all_model_data)
index.add(all_model_data)

# 類似度を検索
k = 10
D, I = index.search(face_encoding, k)

# 類似度が高い順に結果を表示
for i in range(len(I[0])):
    index_i = I[0][i]
    distance_i = D[0][i]
    # コサイン類似度に変換（オプション）
    length_query = np.linalg.norm(face_encoding)
    length_result = np.linalg.norm(all_model_data[index_i])
    cos_similarity = distance_i / (length_query * length_result)
    name_i = all_name_list[index_i]
    dir_i = all_dir_list[index_i]
    print(f"類似度: {cos_similarity:.4f}, 名前: {name_i}, ディレクトリ: {dir_i}")

# 処理時間を計算して出力
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"処理時間: {int(minutes)}分 {seconds:.2f}秒")