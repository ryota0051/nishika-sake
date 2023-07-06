## TODO

- [ ] arcfaceのハイパーパラメータチューニング

    - arcfaceで精度計算ができるようにしておく(学習済みのモデルでtrainを検索結果候補, validをクエリとして精度計算してみる。)

    - cross validation内に、↑の処理をいれる。

    - arcfaceをハイパーパラメータチューニングする(resnet32, 128 x 128)

- [ ] データ拡張の選定をする。

    - cutoff

    - 色変える系

    - mixup

- [ ] チューニングしたデータ拡張などを

- [ ] 別モデルも試してみる(swinformerとか。https://github.com/anyai-28/nishika_jpo_2nd_solution/blob/main/train_swin.py#L453 が参考になりそう。)

- [ ] α weighted query expansion

- [ ] k-foldの結果をhstackしてみる => 普通に平均するより性能が上がるみたい

- [ ] triplet loss, circle loss試す

- [ ] TTA

- [ ] 別モデルの特徴量をconcat(hstack)する
