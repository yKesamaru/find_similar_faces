# 必要なライブラリをインポート
import csv
from collections import defaultdict

def read_csv_file(filename):
    """
    CSVファイルを読み込む関数
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        return [row for row in reader]

def extract_name_pairs_and_average_values(rows):
    """
    各行から名前のペアとその数値を取得し、同じ名前のペアの数値の平均を計算する関数
    """
    pairs_values = defaultdict(float)  # 各ペアの数値の合計を保存する辞書
    pairs_counts = defaultdict(int)     # 各ペアの出現回数を保存する辞書

    for row in rows:
        name1 = row[0].split('_')[0]  # 1つ目の名前を抽出
        name2 = row[1].split('_')[0]  # 2つ目の名前を抽出
        pair = tuple(sorted([name1, name2]))  # 名前のペアをアルファベット順にソート
        value = float(row[2])  # 数値を抽出

        pairs_values[pair] += value  # 同じ名前のペアの数値を合計
        pairs_counts[pair] += 1      # ペアの出現回数をカウント

    # 各ペアの数値の合計をそのペアの出現回数で割って平均を計算
    for pair in pairs_values:
        pairs_values[pair] /= pairs_counts[pair]

    return pairs_values


def main():
    rows = read_csv_file('assets/output.csv')  # CSVファイルのパスを指定
    pairs_values = extract_name_pairs_and_average_values(rows)

    # 結果を出力
    for pair, value in pairs_values.items():
        with open('assets/output2.csv', 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([pair[0], pair[1], value])
        print(f"{pair[0]} と {pair[1]}: 合計値 = {value}")

if __name__ == "__main__":
    main()  # メインの処理を実行
