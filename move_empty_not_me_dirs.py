import os
import shutil  # ディレクトリ移動用


def move_empty_not_me_dirs(base_dir, destination_dir):
    """
    指定したディレクトリ内で、`not_me`という名前の空ディレクトリを持つ
    サブディレクトリを、別のディレクトリに移動します。

    Args:
        base_dir (str): 処理対象のルートディレクトリ。
        destination_dir (str): サブディレクトリを移動する先のディレクトリ。

    Returns:
        None
    """
    # ルートディレクトリ内のサブディレクトリを探索
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)  # サブディレクトリの完全パスを生成

        # サブディレクトリであるかどうかを確認
        if os.path.isdir(subdir_path):
            not_me_path = os.path.join(subdir_path, "not_me")  # not_meディレクトリのパスを生成

            # not_meが存在し、かつ空ディレクトリの場合
            if os.path.exists(not_me_path) and os.path.isdir(not_me_path) and not os.listdir(not_me_path):
                print(f"Moving {subdir_path} to {destination_dir}...")  # 移動の確認用出力

                # サブディレクトリを目的のディレクトリに移動
                shutil.move(subdir_path, destination_dir)
                print(f"Moved {subdir} successfully.")

# スクリプトのエントリーポイント
if __name__ == "__main__":
    # ルートディレクトリ（処理対象）
    base_dir = "/media/terms/2TB_Movie/face_data_backup/woman"
    # 移動先ディレクトリ
    destination_dir = "/media/terms/2TB_Movie/face_data_backup/data"

    # 移動先ディレクトリが存在しない場合は作成
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)  # 必要なディレクトリを作成
        print(f"Created destination directory: {destination_dir}")

    # 関数を実行
    move_empty_not_me_dirs(base_dir, destination_dir)
