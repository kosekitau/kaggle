# スモールデータにどう対応するべきか
## playground s3 e12(スモールデータでの尿管結石予測)
  - 7位解法(https://www.kaggle.com/competitions/playground-series-s3e12/discussion/402385)
    - GBDTは使わない、cvのスプリットは3、特徴量は一部pd.cutなどで簡略化してる、モデルはNNとかLogisticRegressionをブレンディングしてる。
    - ブレンディングのやり方(https://www.kaggle.com/competitions/playground-series-s3e12/discussion/402385#2226337)
  - 病状系はデータ集めづらいのでしょうがない

## NestedCV
  - (https://blog.amedama.jp/entry/2018/07/23/084500)
    - 外側と内側のforループで構成され、外側のループ回数分ハイパラが選択される。通常の交差検証と違ってハイパラをひとつに定めない。
  - (https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/420823#2327513)
  - 実装はここが参考になった(https://github.com/sergeyf/SmallDataBenchmarks)

## smallデータのcv
  - https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/420823
  - https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/414944

## SMOTE UnderSampling
  - は微妙というのをここのコメントで見た(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/414944)
    - 今回はプロットした時に0と1が混ざりすぎてる(t-sneとかを見て？)から、そういう時はSMOTEは向かんとのこと
  - SMOTE(https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)

## 特徴量削減
  - PCAなど圧縮系(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/411466)
  - Boruta algorithm、ノイズより重要度などが低いか(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/417773#2306201)
  - PFI
  - ロジスティック回帰の実験だと、PFIだけで特徴量選択した方がよいかも。Borutaは逆に精度落ちたので。
  - sklearnのドキュメント(https://scikit-learn.org/stable/modules/feature_selection.html)
  - Information Value (IV) and Weights of evidence (WoE) 
    - この例だとIV>0.05で閾値を設定している(https://www.kaggle.com/code/tatudoug/logistic-regression-baseline/notebook)
    - 日本語(https://lazdera.hatenablog.com/entry/2019/11/14/151453)

## 欠損値の処理
  - 一番右下のグラフを見るとNaNを含む箇所はLightGBMでうまく予測をできている(青い点が多い)から特別処理をしなくて良いだろうという主張、回帰モデルとかだと処理した方がええんよろか(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/410843)

# 外れ値の処理
  - np.log1pで正規分布に近付ける。これだけでロジスティック回帰はかなり精度よくなる。

# Logistic Regression
  - 公式にsolverの選択について書かれている、データセットが小さい時はliblinearらしい(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)。

# scipy.minimize()
  - (https://www.kaggle.com/code/tilii7/cross-validation-weighted-linear-blending-errors)
  - (https://www.kaggle.com/code/pourchot/stacking-with-scipy-minimize)

# 分類モデルの評価方法と、アンサンブルのヒント
  - CDFプロット(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/423931)
    - OOFを予測確率でソートして横軸をデータ数縦軸を1の確率でプロットしたカーブ。trainは83%がクラス0なので、横軸83%くらいは縦軸0くらいに張り付いてて欲しいし、横軸の残り17%は縦軸1付近に張り付いた緩急のつかないプロットになってると嬉しい。
    - モデルによって、縦軸0に張り付き型、縦軸1に張り付き型などあるのでそういうのをアンサンブルして平均プロット的になると良くなるのでは
    - あと、GBDTだとモデルは使わん特徴量を勝手に選ぶから特徴量削除はなくても良いかもしれない。線形モデルは知らん。

  - CDFプロット続き(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/426536)

# ハードポイント
  - ロジスティック回帰の結果から
    - 509と313と267はclass1だが当てるのめっちゃむずい
    - 586, 26, 468, 102, 462, 367, はその逆 
  - XGBの結果から
    - 509と313と267はclass1だが当てるのめっちゃむずい(LRと同じ)、408は(25%以下)
    - 292, 102, 556, 195, 386, 462, 468, 367, 190, 498はその逆(90%以上の確率で1と予測)
  - LGBMの結果から
    - 509と313と479がclass1だがむずい(1の確率10%以下)、193, 145, 408, 267は(25%以下)、31、434、274、229は(50%以下)
      - XGBと違い、267が1である確率はちょっと高い
  - MultiRepeatedStratifiedCVのために激ムズclass1を選ぶなら
    - 509, 313, 479, 267, 408, 193, 145, (229, 31, 434)

# アンサンブル
  - アンサンブルのまとめ記事(https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/51058)
  - 多様性の話はわかるが、混ぜ方がわからない

# 検討事項
  - LRとlgbmのCDFプロット
  - 欠損値処理+LR
  - ハードポイントと時間の関係性
----



- 大事そう：AB, AF, BC, BQ, CR, DA, DU, EU, FR, GL

- BQが重要度高いのはNaNあるところが0になるから
  - BQのNaNとELのNaNには関係性がありそう
  - クソでかBQは半分(19)が1
- クソでかDUは1になりがち、RFもLGBMもLRも重要度とか回帰係数がでっかくなる
- クソでかABは1多め？
- FLデカめは1多め
- 外れ値取ってるやつはだいたい1になることが多い。
- BNは年齢
- 特徴量削減→外れ値処理、欠損値処理の順番で検討するべきか
- PFIで精度が上がる特徴量は弾く、ノイズに重要度負ける特徴量は弾く、そもそもノイズの重要度が高くなるモデルは使わない
- 時間で分布変わってる可能性あるか？(見つけやすくなってるとか、)(これ以上データ削るの？、、)
- raddar氏によるexcel閲覧(https://www.kaggle.com/code/raddar/icr-competition-analysis-and-findings/comments#2324689)
- t-SNE + knnによる分析(https://www.kaggle.com/code/raddar/icr-competition-analysis-and-findings/comments#2324775)



- RandomForestでノイズより重要度低いカラム
  - AH, AR, AZ, BD, BP, BZ, CF, CH, CL, CW, DV, EJ, GB, GE, GF, GI
- LightGBMでノイズより重要度低いカラム
  - ['AH', 'AR', 'AX', 'AY', 'AZ', 'BD ', 'BN', 'BP', 'BR', 'BZ', 'CB', 'CL', 'CS', 'CU', 'CW ', 'DF', 'DV', 'EG', 'EJ', 'EL', 'FC', 'FS', 'GB', 'GE', 'GF', 'GI']
- LogisticRegressionでノイズより回帰係数小さいカラム
  - ['AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BD ', 'BN', 'BP', 'BR', 'BZ', 'CB', 'CC', 'CL', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DL', 'DU', 'DY', 'EB', 'EJ', 'EU', 'FC', 'FE', 'FL', 'FS', 'GB', 'GF', 'GH', 'GI', 'GL']
  - ['AM', 'AR', 'AX', 'AY', 'AZ', 'BD ', 'BP', 'BR', 'CB', 'CS', 'CU', 'CW ', 'DA', 'DE', 'DF', 'DH', 'DY', 'EG', 'EJ', 'EL', 'EU', 'FC', 'FD ', 'FE', 'GB', 'GL']
- 上記の結果から自分が重要ではないかと思ったカラム
  - ['AB', 'AF', 'BC', 'BQ', 'CC', 'CD ', 'CR', 'DI', 'DL', 'DN', 'DU', 'EB', 'EE', 'EH', 'EP', 'FI', 'FL', 'FR', 'GH']


- 20230709 LogisticRegressionでnestedcvしてPFIが0未満になっただめなカラム
  - ['AF', 'AH', 'AX', 'AY', 'BR', 'BZ', 'CL', 'CS', 'CW ', 'DA', 'DF', 'EG', 'EL', 'EU', 'FC', 'FD ', 'GL']
  - 第2次波：['AZ', 'BD ', 'BN', 'BP', 'CF', 'CU', 'DE', 'FE', 'GB']
  - 3 : ['AR', 'EH', 'GF']
  - 4 : ['FL']


- 20230711 LightGBMでnestedcvしえtPFIが0未満だったカラム
  -  ['AX', 'AY', 'BC', 'BN', 'BR', 'BZ', 'CB', 'CL', 'DF', 'DH', 'DV', 'EJ', 'FC', 'FE', 'GB', 'GE', 'GF']
  - 2:['AH', 'AR', 'AZ', 'CW ', 'EG', 'GH']　ここでスコア下がった


- 20230712
  - ['AH' 'AX' 'AY' 'BR' 'BZ' 'CL' 'CS' 'CW ' 'DF' 'EG' 'EL' 'FC']
　　- こいつらはマジモンのゴミかもしれないこいつらを抜いたlgbmとlrはcvもpublicも上がってる
  - ロジスティック回帰とPFIで特徴量削りまくったらcv上がるがPublicは爆上がりした。どこかでtrain.csvにfitしすぎてしまうのかもしれない。明日PFIでゴリゴリに削ったlgbm試してだめなら削るのは最低限にする

- 20230713
  -  ['AX', 'AY', 'BC', 'BN', 'BR', 'BZ', 'CB', 'CL', 'DF', 'DH', 'DV', 'EJ', 'FC', 'FE', 'GB', 'GE', 'GF',]
  - cv=0.24, LB=0.23なのであんまりかな。削っちゃいかんのがあったみたい。lgbm+PFIで削るのは微妙か？明日はXGB+NestedCVやる

- 20230714
  - XBGで特徴量全部使ってcv=0.17, LB=0.17だった。スモールデータだがGBDT系が強いんかな
  - PFIの平均が0以下だったもの['AH', 'AR', 'AY', 'AZ', 'BC', 'BN', 'BP', 'BR', 'BZ', 'DE', 'DF', 'DH', 'DV', 'FC', 'FE', 'FL', 'GE', 'GF', 'GI']
  - Catのコードの書き方わからん

- 20230715
  - RFのcv0.33くらいしかいかん。

- 20230716
  - XGBで特徴量削ったらLBだけ精度下がる現象起きた

- 20230717
  - Catだめっぽい。cv0.25くらいしかいかん

- 20230727
  - score0.30まで混ぜるか(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/426536#2357198)

- 20230729
  - ensembleの結果、probaが0とか1には張り付いてないように見える(https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/427446)
  - (https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/426751)
  - モデルによってclass1の予測が得意なもの、class0の予測が得意なもの、個性がある。TabPFNはCDFプロットの結果、class0の予測が得意らしい。