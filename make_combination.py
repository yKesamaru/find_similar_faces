import os
from tqdm import tqdm  # tqdmをインポート

# dataディレクトリ以下のサブディレクトリを探索
os.chdir("/media/terms/2TB_Movie/face_data_backup/")
parent_dir = "data"
sub_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

# 保存先のcsvファイル名
output_csv = "combination_file.csv"

# csvファイルに書き込むための準備
with open(output_csv, "w") as f:
    # 各サブディレクトリに対して処理（tqdmで進捗を表示）
    for i, sub_dir1 in enumerate(tqdm(sub_dirs, desc="Processing directories")):
        # サブディレクトリ内のPNGファイルを探してリスト化
        png_files1 = [os.path.join(sub_dir1, fname) for fname in os.listdir(sub_dir1) if fname.endswith('.png')]

        # 残りのサブディレクトリに対して処理
        for sub_dir2 in sub_dirs[i+1:]:
            # サブディレクトリ内のPNGファイルを探してリスト化
            png_files2 = [os.path.join(sub_dir2, fname) for fname in os.listdir(sub_dir2) if fname.endswith('.png')]

            # サブディレクトリが異なる場合のみ組み合わせを作成
            for file1 in png_files1:
                for file2 in png_files2:
                    f.write(f"{file1},{file2}\n")



