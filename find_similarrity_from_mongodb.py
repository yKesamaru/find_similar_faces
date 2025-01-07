import os
import sys
import time

import faiss
import numpy as np
from pymongo import MongoClient

# FACE01ライブラリのインポート
sys.path.insert(1, '/home/terms/bin/FACE01_IOT_dev')
from face01lib.api import Dlib_api

api = Dlib_api()

# 処理開始時刻を記録
start_time = time.time()

# 顔写真をロード
face_image = api.load_image_file("/home/terms/ドキュメント/find_similar_faces/assets/woman2.png")
# face_image = api.load_image_file("/home/terms/ドキュメント/find_similar_faces/assets/woman.png")
face_location = api.face_locations(face_image, mode="cnn")
face_encoding = api.face_encodings(
    deep_learning_model=1,
    resized_frame=face_image,
    face_location_list=face_location,
)
face_encoding = np.array(face_encoding[0][0]).reshape(1, 512)

# MongoDBに接続
client = MongoClient('localhost', 27017)
db = client['face_recognition_db']
collection = db['known_faces']

# MongoDBから全てのドキュメントを取得
documents = collection.find({})

# MongoDBから取得した特徴量を格納するリスト
db_features = []
# 各特徴量に対応するドキュメントIDを格納するリスト
db_ids = []

for doc in documents:
    feature = np.array(doc['vector']).reshape(1, 512)
    db_features.append(feature)
    db_ids.append(doc['_id'])

# NumPy配列に変換し、データ型をfloat32に変換
db_features = np.vstack(db_features).astype('float32')

# 特徴量ベクトルをL2正規化
db_features = db_features / np.linalg.norm(db_features, axis=1)[:, np.newaxis]
face_encoding = face_encoding / np.linalg.norm(face_encoding, axis=1)[:, np.newaxis]  # 追加; face_encodingも正規化

# FAISSインデックスを作成（内積を使用）
index = faiss.IndexFlatIP(512)
index.add(db_features)

# 類似度検索（コサイン類似度）
D, I = index.search(face_encoding, 5)  # 5つの最も類似した特徴量を検索

# 類似した特徴量のドキュメントIDと類似度（コサイン類似度）を表示
for i, d in zip(I[0], D[0]):
    similar_doc_id = db_ids[i]
    print(f"Similar document ID: {similar_doc_id}")

    # 類似度（コサイン類似度）を表示
    print(f"Similarity Score (Cosine Similarity): {d}")

    # IDに基づいてMongoDBからドキュメントを取得
    similar_doc = collection.find_one({"_id": similar_doc_id})

    # ドキュメントからfile_nameを取得して表示
    if similar_doc and 'file_name' in similar_doc:
        print(f"Similar file name: {similar_doc['file_name']}")

# 処理時間を計算して出力
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"処理時間: {int(minutes)}分 {seconds:.2f}秒")
