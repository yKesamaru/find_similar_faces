# [faiss] データセットから似ている顔を検索する

![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/eye_catch.png)

- [\[faiss\] データセットから似ている顔を検索する](#faiss-データセットから似ている顔を検索する)
  - [はじめに](#はじめに)
    - [顔データセット](#顔データセット)
    - [顔写真](#顔写真)
  - [実装](#実装)
  - [実行結果](#実行結果)
  - [まとめ](#まとめ)

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
![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/woman.png)

## 実装

https://github.com/yKesamaru/find_similar_faces/blob/4eb0c208138791ce0898cee70e4250bcf5e5307f/find_similarrity.py#L1-L94

## 実行結果
```bash
類似度: 0.4285, 名前: 大谷直子_vixw.jpg.png_align_resize_default.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/大谷直子
類似度: 0.3852, 名前: 岸本加世子_QIp1.jpg_default.png.png_0.png_0_align_resize.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/岸本加世子
類似度: 0.3835, 名前: 岸本加世子_9iBC.jpg.png_align_resize_default.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/岸本加世子
類似度: 0.3834, 名前: 岸本加世子_vByP.jpg.png_align_resize_default.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/岸本加世子
類似度: 0.3480, 名前: 宮崎美子_cdzk.jpg.png.png_0.png_0_align_resize.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/宮崎美子
類似度: 0.3460, 名前: 愛華みれ_rU0K.jpg..png_align_resize_default.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/愛華みれ
類似度: 0.3454, 名前: 愛華みれ_Trha.jpg..png_align_resize_default.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/愛華みれ
類似度: 0.3394, 名前: 宮崎美子_l4MT.jpg.png.png_0.png_0_align_resize.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/宮崎美子
類似度: 0.3333, 名前: 岸本加世子_qjSm.jpg_default.png.png_0.png_0_align_resize.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/岸本加世子
類似度: 0.3306, 名前: 宮崎美子_neJS.jpg.png.png.png_0.png_0_align_resize.png, ディレクトリ: /media/user/2TB_Movie/face_data_backup/data/宮崎美子
処理時間: 0分 41.79秒
```
![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/大谷直子_vixw.jpg.png_align_resize_default.png)

大谷直子さん。
コサイン類似度は0.4285でした。Maxが1.0ですので、大して似ていないですが、データセットの中ではもっとも類似度が大きく出ました。

![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/woman.png)
![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/2023-09-28-19-31-38.png)

## まとめ
実は今回検索対象とした女性の顔は、生成AIによって生成されたものです。
ですので、検索の結果、ヒットしなかったのはむしろ成功と言えます。

ただしもととなったLoRA（Low-Rank Adaptation）は[average face of Japanese MILFs 日本人熟女平均顔](https://www.seaart.ai/models/detail/64412adea99838a2740557673f2066ae)です。

LoRAに実際の人物写真が使われた場合、その特徴を学習します。そのような場合、生成AIではどんな画像でも作れてしまいますから、マズい画像も出回ってしまうわけです。

今回の実験では、そのへんの調査のための下地作りといえます。

以上です。ありがとうございました。
