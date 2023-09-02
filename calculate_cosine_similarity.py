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

