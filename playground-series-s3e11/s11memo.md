# 特徴量
- ベースの特徴量からいくつか削除した以下を使用。codeから持ってきた。なんで下のやつに絞ったか言及してるnotebookを見つけられなかったが、LightGBMでpermutation importanceの実験をしたら以下をシャッフルすると精度が落ちると分かり個人的には納得。
  - permutation importanceの実装参考：https://www.kaggle.com/code/cdeotte/lstm-feature-importance/notebook
```python
['total_children', 'num_children_at_home', 'avg_cars_at home(approx).1', 'store_sqft', 'coffee_bar', 'video_store', 'prepared_food', 'florist', 'children_ratio']
```
- codeから以下もパクって追加。なぜ良いのか分かってない。
```python
df['children_ratio'] = df['total_children']/df['num_children_at_home']
```
- 以下も考えてみて試したがcvもpublicもスコア落ちたので捨てた。
```python
# 場所の利用数
place = ['coffee_bar', 'video_store', 'salad_bar']
df_train["place_counts"] = df_train[place].sum(axis=1)
df_test["place_counts"] = df_test[place].sum(axis=1)
# 食品系のカウント数
food = ['low_fat', 'prepared_food', 'recyclable_package']
df_train["food_counts"] = df_train[food].sum(axis=1)
df_test["food_counts"] = df_test[food].sum(axis=1)
# 場所の利用数2
place = ['coffee_bar', 'video_store']
df_train["place_counts2"] = df_train[place].sum(axis=1)
df_test["place_counts2"] = df_test[place].sum(axis=1)
# (来店した子供数 - 地域の平均子供数)で大きいと子供向け施策?
df_train["for_child"] = (df_train["total_children"] - df_train["num_children_at_home"]).clip(lower=0)
df_test["for_child"] = (df_test["total_children"] - df_test["num_children_at_home"]).clip(lower=0)
```

# モデル
- 以下を試した。結局RandomForestで1つ、LightGBMで2つcvスコアが良い感じのハイパラを見つけられたので採用。他のモデルはcvがあまり良くなかったので使わないことにした。
  - LightGBM
  - CatBoost
  - XGBoost
  - RandomForest
  - ExtraTrees
  - (TabNet)←ちょっと触っただけ
- ハイパラ探索はoptunaを使った。探索範囲はThe Kaggle Bookを参考にした。

# アンサンブル
- 上記3つの(モデル+ハイパラ)の組み合わせに対して、random_stateだけ変更してモデルを学習させる作業を繰り返し、モデル数を合計30個までかさ増しさせた。
  - 参考：https://speakerdeck.com/rsakata/santander-product-recommendationfalseapurotitoxgboostfalsexiao-neta?slide=39
- 最終的な予測は単純に全モデルの予測の平均で出した。random_state変更かさ増しのおかげで、cvもpublicも0.00001くらいスコアが良くなった。

# 反省点
- 目的変数が離散値だったので予測値を一番近い値に近づける処理を行なった。最終日夜に突然思いつき、cvでスコアがどれくらい変動するか検証せずなぜか確信を持って、それを最終提出に採用してた。結局修正しない方がprivateのスコアは良かった。cvで検証せずsubmitは大変に良くないと感じた。
```python
sub = pd.read_csv("submissions/submission.csv")

df_train = pd.read_csv("data/train.csv")
df_train["cost_copy"] = df_train["cost"].tolist()
d = pd.merge_asof(sub.sort_values("cost"), df_train.sort_values("cost")[["cost", "cost_copy"]], on="cost", direction="nearest") # trainの一番近いcostに予測を修正する

submission = pd.read_csv("data/sample_submission.csv")
submission = pd.merge(submission, d, on="id")
submission = submission.drop(["cost_x", "cost_y"], axis=1).rename({"cost_copy":"cost"}, axis=1)

now = datetime.now().astimezone(timezone('Asia/Tokyo'))
submission.to_csv(f"submissions/{now}-submission.csv", index=None)
```

- 特徴量作成の勘みたいなのは相変わらず分からない。ランダム生成とかを試してみようか。
- originalデータを追加すると良かったらしい。trainデータでかいし変わらんだろwって考えてたけど、横着せずにまずは検証しよう。