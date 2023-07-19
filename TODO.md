## TODO

### ハイパーパラメータチューニング系

- [ x ] arcface のハイパーパラメータチューニング

  - arcface で精度計算ができるようにしておく(学習済みのモデルで train を検索結果候補, valid をクエリとして精度計算してみる。)

  - cross validation 内に、↑ の処理をいれる。

  - arcface をハイパーパラメータチューニングする(resnet32, 128 x 128)

  流れは、以下を各ハイパーパラメータごとに評価して一番性能が高いものを採用する。

  1. 1fold ごとに学習用データ, 検証用データをベクトル化して、faiss で近傍探索を実施(ベクトル化するモデルは val_loss が一番低いモデル => 保存しておく)

  2. 上位 20 件から精度指標を計算

  3. 全 fold の精度指標の平均を計算

- [ x ] データ拡張の選定をする。

  - cutoff

  - 色変える系

  - mixup

- [ x ] チューニングしたデータ拡張などを本番のモデルで実施する

### 別モデル使う系 => 一旦こちらを先に実施してもいいかも(ハイパーパラメータチューニング系は実装に時間がかかりそうなので。) => これが一番聞きそう

- [ ] 別モデルも試してみる(swinformer とか。https://github.com/anyai-28/nishika_jpo_2nd_solution/blob/main/train_swin.py#L453 が参考になりそう。)

  個人的にはこれやりたい

- [ ] 別モデルの特徴量を concat(hstack)する

- [ ] triplet loss, circle loss 試す

- [ ] SubCenterArcface(https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/SubCenterArcFaceMNIST.ipynb 参照)

### 後処理系(これはいつでもできるので一番最後でもいいかも)

- [ x ] α weighted query expansion => あんまりよくなさそう

- [ ] TTA

## モデルの課題

- 学習用データにないっぽいやつが上手く判定できないっぽい(以下の 2 つを確認) => Triplet loss(ContrastiveLoss が楽そう)とか使うとよい?

  - 瀬戸のさざなみ(https://www.isochidori.co.jp/index.php?main_page=product_info&products_id=13)

  - 一茶の里(里がついている別の銘柄がヒットした。https://www.saketime.jp/brands/4279/)

  - 学習用データにはないがラベルの色が同じようなものは当てられる

- 文字が入っていないやつもミスる? => モデルを大きくするとか?

- 回転しているっぽいやつが弱い(データ拡張するとある程度ましになる) => TTA で回転を加えるとマシになるかも

  - 千代の亀(200110806)とか
