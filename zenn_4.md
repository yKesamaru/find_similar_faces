# データセットから似ている顔を検索する

![](assets/eye_catch.png)

## はじめに
モデルの学習に使われていない顔データセットを用い、用意した写真と似ている顔を検索します。

今回の記事では、顔写真を1枚用意し、`faiss`を用いて直接データセットから検索します。

### 顔データセット
```bash
find . -maxdepth 2 -type f -name *.png | wc -l
57637
```
約5万7千枚の顔写真が含まれるデータセットを用意しました。

### 顔写真
![](assets/2023-09-28-17-31-34.png)

## 実装