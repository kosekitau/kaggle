# 序盤
- 線形回帰やARIMA、N時点前をそのまま予測に持ってくるだけでもcvやLBの成績は良さそうと分かる。
  - Linear Regression Baseline - [LB 1.092]
    - https://www.kaggle.com/code/cdeotte/linear-regression-baseline-lb-1-092
  - GDMBF: AR | MA | ARMA | ARIMA | SARIMA| AUTO ARIMA
    - https://www.kaggle.com/code/tanmay111999/gdmbf-ar-ma-arma-arima-sarima-auto-arima
    - 終盤で自分でもARIMAでモデリング実験をしてみたが、cfipsの3135地域中、(p,d,q)のパラメータが(0,1,0)となるのが一番多く2454地域あった。
  - New Last Value Baseline is LB 1.4631 not LB 3.2776 - Read Why Here
    - https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/389215

- LightGBMなどを使った手法もあり回してみたが、1ヶ月先予測の時点で1時点前の目的変数に若干の上方修正をかけた感じの予測値になり、機械学習モデルでも予測は難しそうに感じた。
  - GoDaddy: Tune Stacking
    - https://www.kaggle.com/code/batprem/godaddy-tune-stacking

- そもそも予測をすること自体が無理(privateは3~5ヶ月先予測なので尚更無理)。

# 中盤
- 上記よりN時点前予測が多く提出されると思ったので、N時点前より正解に近い値を提出すればコンペには勝てるのではと思った(予測としては全然当たってないけども)。
- LightGBMでNヶ月先の目的変数が今より上昇するか減少するかの2値分類器を作り、上昇する場合一定の上昇補正値をかける減少ならその逆という手法を考えた。しかし3ヶ月先分類器の時点で正解率が43%くらいのものしか作れず断念。
- ここで、list(range(1, len(train_data)+1))という特徴量を加えて分類器を作った瞬間、検証データの正解率がほぼ100%を達成することを確認。
- 学習データの後半は目的変数が上昇傾向にあることが分かった。

# 終盤
- 以下のnotebookを見つける。データ全体に上昇傾向があり、予測せずN時点前の目的変数に1.0045をかけてそれを提出するもの。このやり方を改造することにした。
  - 21 Lines of Code
    - https://www.kaggle.com/code/vitalykudelya/21-lines-of-code
- 全て1.0045をかけるのではなく、1(public用)、3、4、5ヶ月先予測ごとに最適な補正値を求めることにした。

```python

ACT_THR = 150
lags = [1, 3, 4, 5] # Nヶ月先予測、3~5はprivate
mults = list(np.round(np.arange(0.9960, 1.0200, 0.0005), 4)) # 補正値の候補
result = []
TS = list(range(35, 41)) # 2022年7月~2022年12月を検証用のデータとした。

for lag in lags:
  for mult in mults:　# グリッドサーチ
    TS_pred = [t-lag for t in TS] # 2022年7月~2022年12月のlagヶ月前を求める
    y = df_train.query("dcount==@TS & lastactive>@ACT_THR")["active"].to_numpy() # 2022年7月~2022年12月の目的変数を取り出す
    y_pred = df_train.query("dcount==@TS_pred & lastactive>@ACT_THR")["active"].to_numpy() # 2022年7-lag月~2022年12-lag月の目的変数を取り出す
    y_pred = y_pred * mult # 候補補正値multをかけて予測を作る
    a = pd.DataFrame({"y":y, "y_pred":y_pred}).dropna()
    score = smape(a["y"], a["y_pred"])
    result.append([lag, mult, score])

result = pd.DataFrame(result, columns=["lag", "mult", "SMAPE"])
display(result.query("lag==1").sort_values("SMAPE").head(1))
display(result.query("lag==3").sort_values("SMAPE").head(1))
display(result.query("lag==4").sort_values("SMAPE").head(1))
display(result.query("lag==5").sort_values("SMAPE").head(1))

# それぞれ一番良かったもの→{1:1.0025, 3:1.0105, 4:1.0135, 5:1.017}

```
- {1:1.0025, 3:1.0105, 4:1.0135, 5:1.017}をそれぞれ2022年12月の目的変数にかけたものを提出した。