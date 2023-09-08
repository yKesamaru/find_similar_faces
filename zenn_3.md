前回の記事で、顔データセットの類似度の計算には**組み合わせが15億8600通りあり、総当りで計算すると1年以上**かかる事が分かりました。
そのため、`faiss`を用いて多次元ベクトル類似度計算の高速化を試みましたが、前回の記事では、`faiss`を用いても計算時間が長くなってしまいました。
そのため、以下の`faiss`チュートリアルを参考に、計算時間を短縮することができるかを試してみました。

https://github.com/facebookresearch/faiss

https://github.com/facebookresearch/faiss/wiki/Getting-started

Faiss解説シリーズ（第一回）基本編（Go言語）
https://crumbjp.hateblo.jp/entry/2021/05/05/Faiss%E8%A7%A3%E8%AA%AC%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA%EF%BC%88%E7%AC%AC%E4%B8%80%E5%9B%9E%EF%BC%89

結果として、すべての組み合わせに対する類似度計算が**わずか9秒で完了しました**。

この記事では、実際のfaissコードと、コードの解説をします。

![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/%E9%A1%94%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%81%AE%E9%96%93%E9%81%95%E3%81%84%E6%8E%A2%E3%81%97_4.png)

今回は、以下の記事の続きです。
https://zenn.dev/ykesamaru/articles/d4b78ea53b02e2
https://zenn.dev/ykesamaru/articles/83b171b045809c

- [はじめに](#はじめに)
- [faissコード](#faissコード)
  - [コード解説](#コード解説)
    - [インデックスの設定](#インデックスの設定)
    - [コサイン類似度の比較](#コサイン類似度の比較)
    - [インデックス追加](#インデックス追加)
    - [検索](#検索)
- [出力結果](#出力結果)
- [まとめ](#まとめ)


## はじめに
> 深層学習におけるモデル学習において、データセットのクレンジングは重要な作業です。
> 顔認証システムにおいてのデータセットのクレンジングとは、「人物Aの顔画像ファイルが、間違いなく人物Aのフォルダーに存在しているか」と定義できます。
> このクレンジング作業は、ある程度自動化していますが、最終的には目視で確認する必要があります。
> なかには知っている人物もありますが、大部分は知らない人物です。
> スクレイピング対象の人物名がマイナーな場合（仮に人物Aとします）、同じ名字の有名人（人物B）がヒットしてしまうこともあります。
> 有名人と言っても私は知らないので、顔画像枚数の多い人物Bを、人物Aのフォルダーに配置してしまうかもしれません。
> 人物Aのフォルダーには人物Aの顔画像ファイルが存在し、人物Bのフォルダーにも人物Aの顔画像ファイルが存在することになってしまいます。
> この状態は、モデル学習において、大きな悪影響を及ぼします。
> 
> そこで既存の顔学習モデルを使用して、各フォルダーの顔画像ファイルと、他のフォルダーに存在する顔画像ファイルとのコサイン類似度を計算し、類似度が高いものを抽出します。

ここまでが、これまでの記事の共通な内容です。

それでは、実際のコードを見ていきましょう。

<参考>
- Building an index and adding the vectors to it
https://github.com/facebookresearch/faiss/wiki/Getting-started#building-an-index-and-adding-the-vectors-to-it

- Faster search
https://github.com/facebookresearch/faiss/wiki/Faster-search

## faissコード
https://github.com/yKesamaru/find_similar_faces/blob/780fe23d16890a24e83fc8d70281aa9410e2eb77/faiss_6_combination_similarity_copy.py#L1-L71

### コード解説

#### インデックスの設定
<参考>
- Cell-probe methods (IndexIVF* indexes)
https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#cell-probe-methods-indexivf-indexes

https://github.com/yKesamaru/find_similar_faces/blob/780fe23d16890a24e83fc8d70281aa9410e2eb77/faiss_6_combination_similarity_copy.py#L9-L15

1. **dimension = 512**: ベクトルの次元数を設定。ここでは、各ベクトルが512次元であると指定しています。

2. **nlist = 100**: クラスター数を指定。FAISSのIVF（Inverted File）アルゴリズムでは、全体のデータセットをいくつかのクラスターに分割します。この`nlist`は、そのクラスター数を指定しています。

3. **quantizer = faiss.IndexFlatIP(dimension)**: 量子化器（Quantizer）を作成。量子化器は、クラスタリングの際に各ベクトルがどのクラスターに属するかを決定する役割を果たします。`IndexFlatIP`は内積（Inner Product）を使用する量子化器です。

4. **index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)**: IVFフラットインデックスを作成。このインデックスは、上で作成した量子化器（`quantizer`）、ベクトルの次元数（`dimension`）、クラスター数（`nlist`）、そして距離計算のメトリック（この場合は内積）をパラメーターとして受け取っています。

`nlist`については、通常試行錯誤して値をつめていきますが、今回は`100`で十分な精度と性能が得られました。値を大きくすると、より精度が上がりますが、計算時間が増えます。

#### コサイン類似度の比較
<参考>
- How can I index vectors for cosine similarity?
https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity

https://github.com/yKesamaru/find_similar_faces/blob/780fe23d16890a24e83fc8d70281aa9410e2eb77/faiss_6_combination_similarity_copy.py#L34-L46


1. **データの整形**: `model_data = model_data.reshape((model_data.shape[0], -1))` によって、データを2次元配列に整形します。これにより、各行が1つの特徴ベクトルとなります。

2. **L2正規化**: `faiss.normalize_L2(model_data)` によって、各特徴ベクトルをL2正規化します。L2正規化されたベクトル間の内積は、そのベクトル間のコサイン類似度と等しくなります。

3. **データの集約**: `all_model_data.append(model_data)` と `all_name_list.extend(name_list)` によって、すべてのサブディレクトリからのデータを1つの大きな配列に集約します。これにより、後のステップで一度に多くの比較が可能になります。

4. **ディレクトリ情報の保存**: `all_dir_list.extend([dir] * len(name_list))` によって、どの特徴ベクトルがどのディレクトリから来たのかを追跡します。同じディレクトリ内の要素同士の比較を避けるためにです。（各フォルダーがそれぞれクラスであるからです。）


https://github.com/yKesamaru/find_similar_faces/blob/780fe23d16890a24e83fc8d70281aa9410e2eb77/faiss_6_combination_similarity_copy.py#L48-L49

この部分で `np.vstack(all_model_data)` を使用しているのは、Pythonのリストに保存された複数のNumPy配列（`all_model_data`）を1つの大きなNumPy配列に縦方向（行方向）に結合するためです。

`all_model_data` リストには各サブディレクトリから読み込んだ特徴ベクトルが格納されています。これらを1つのNumPy配列に結合することで、後続の処理（FAISSによる高速な検索や類似度計算など）を一度に行えるようにします。

#### インデックス追加

https://github.com/yKesamaru/find_similar_faces/blob/780fe23d16890a24e83fc8d70281aa9410e2eb77/faiss_6_combination_similarity_copy.py#L51-L53


1. `index.train(all_model_data)`: 量子化器（`quantizer`）を訓練します。この訓練プロセスでは、データセット（`all_model_data`）のに基づいて、量子化器がどのようにデータをクラスタリング（分割）するかを学習します。

2. `index.add(all_model_data)`: 訓練された量子化器を用いて、実際のデータ（`all_model_data`）をFAISSインデックスに追加します。これにより、インデックスが検索可能な状態になります。

#### 検索
https://github.com/yKesamaru/find_similar_faces/blob/780fe23d16890a24e83fc8d70281aa9410e2eb77/faiss_6_combination_similarity_copy.py#L55-L57

この部分で、FAISSインデックスを用いて類似度が高い要素を検索しています。
各データポイントに対して、もっとも類似度が高い`k`個のデータポイントと、その類似度（距離）を取得します。

1. `k = 10`: ここで`k`は、各クエリポイントに対して検索する**もっとも類似度の高い要素の数**を指定します。ここでは、各データポイントに対してもっとも類似度が高い10個の要素を検索しています。

2. `D, I = index.search(all_model_data, k)`: `index.search`メソッドを用いて実際に検索を行います。
    - `all_model_data`: 検索対象のデータセット。
    - `k`: 上で設定した、検索するもっとも類似度の高い要素の数。
    - `D`: 返される距離（この場合は内積）の配列。`D[i]`は、`i`番目のクエリポイントに対するもっとも類似度の高い`k`個の要素の距離を含みます。
    - `I`: 返されるインデックスの配列。`I[i]`は、`i`番目のクエリポイントに対するもっとも類似度の高い`k`個の要素のインデックスを含みます。


## 出力結果
```bash
処理時間: 0分 8.98秒
```
![](https://raw.githubusercontent.com/yKesamaru/find_similar_faces/master/assets/2023-09-07-13-02-01.png)

出力されたcsvファイルについて、3列目の数値でソートし、類似度が高い順に並べ替えます。

```bash
sort -t, -k3,3 -n -r output.csv > sorted_output.csv
```
> <補足>
> `-t`: フィールド区切り文字
> `-k`: キーとなるフィールド（列）
> `-n`: は数値として並べ替えることを指示。これがないと、文字列としての比較になる。
> `-r`:「逆順」でソート。通常、sortコマンドは昇順（小さいものから大きいものへ）。


```bash
森本龍太郎_0euH.png_default.png.png_0.png_0_align_resize.png,森本慎太郎_FQuR.png_default.png.png_0.png_0_align_resize.png,1.0000004768371582
森本慎太郎_FQuR.png_default.png.png_0.png_0_align_resize.png,森本龍太郎_0euH.png_default.png.png_0.png_0_align_resize.png,1.0000004768371582
夏子_dt71.jpg.png.png.png_0.png_0_align_resize.png,横澤夏子_Df6H.jpg.png.png.png_0.png_0_align_resize.png,1.0000003576278687
横澤夏子_Df6H.jpg.png.png.png_0.png_0_align_resize.png,夏子_dt71.jpg.png.png.png_0.png_0_align_resize.png,1.0000003576278687
夏子_ZkZ1.jpg.png.png.png_0.png_0_align_resize.png,横澤夏子_tof4.jpg.png.png.png_0.png_0_align_resize.png,1.000000238418579
夏子_TXdE.jpg.png.png.png_0.png_0_align_resize.png,横澤夏子_jY2j.jpg.png.png.png_0.png_0_align_resize.png,1.000000238418579
夏子_3Rju.jpg.png.png.png_0.png_0_align_resize.png,横澤夏子_OH0A.jpg.png.png.png_0.png_0_align_resize.png,1.000000238418579
横澤夏子_tof4.jpg.png.png.png_0.png_0_align_resize.png,夏子_ZkZ1.jpg.png.png.png_0.png_0_align_resize.png,1.000000238418579
横澤夏子_jY2j.jpg.png.png.png_0.png_0_align_resize.png,夏子_TXdE.jpg.png.png.png_0.png_0_align_resize.png,1.000000238418579
横澤夏子_OH0A.jpg.png.png.png_0.png_0_align_resize.png,夏子_3Rju.jpg.png.png.png_0.png_0_align_resize.png,1.000000238418579
森本龍太郎_XXhz.jpg_default.png.png_0.png_0_align_resize.png,森本慎太郎_9RHi.jpg_default.png.png_0.png_0_align_resize.png,1.0000001192092896
森本慎太郎_9RHi.jpg_default.png.png_0.png_0_align_resize.png,森本龍太郎_XXhz.jpg_default.png.png_0.png_0_align_resize.png,1.0000001192092896
夏子_c7os.jpg.png.png.png_0.png_0_align_resize.png,横澤夏子_MCPB.jpg.png.png.png_0.png_0_align_resize.png,1.0000001192092896
横澤夏子_MCPB.jpg.png.png.png_0.png_0_align_resize.png,夏子_c7os.jpg.png.png.png_0.png_0_align_resize.png,1.0000001192092896
(...後略)
```
狙い通りの結果を得られました。

「森本龍太郎」フォルダーと「森本慎太郎」フォルダーに同一人物の顔画像ファイルが混在しています。また、「夏子」フォルダーと「横澤夏子」フォルダーにも同一人物の顔画像ファイルが混在しています。
この他に、「おぎやはぎ小木」フォルダーと「小木博明」、「高橋光」フォルダーと「髙橋ひかる」フォルダーにも同一人物の顔画像ファイルが混在しています。

これらはそれぞれ、同じフォルダーにまとめる必要があります。（同じクラスにする、ということです。）

## まとめ
`faiss`の威力を痛感しました。