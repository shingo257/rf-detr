# COCO17 拡張体幹コントロール

RF-DETR の COCO 17 キーポイントから、キャラクターアニメーション用の体幹制御点と姿勢パラメータを生成する。
追加学習前の段階で腰の傾き、肩と骨盤の逆回転、簡易的な S 字変形を利用可能にするためのモジュールである。

## 実装範囲

- 入力: `sv.KeyPoints` または 1 人分の `(17, 2)` 座標
- 出力: 追跡・平滑化後の COCO 17 点と 13 個の体幹制御点
- 人物ごとの時系列平滑化
- 最大 4 フレームの短時間欠損保持
- 制御点の信頼度と生成元
- フレーム単位の JSON 変換
- 欠損率と正規化ジッターの評価

生成する制御点は `neck_base`, `sternum`, `left_lower_rib`, `right_lower_rib`, `navel`,
`left_waist`, `right_waist`, `pelvis_center`, `spine_upper`, `spine_mid`, `spine_lower`,
`chest_center`, `hip_axis_center` である。
各人物の`coco17`には顔、肩、肘、手首、腰、膝、足首を残すため、後段の全身リターゲットで腕、脚、頭を制御できる。

## 動画処理への接続

追跡処理と既存のキーポイント時間フィルターを適用した後の `sv.KeyPoints` を渡す。

```python
from rfdetr_demo.animation.pipeline import TorsoControlPipeline, torso_frame_as_dict

torso_pipeline = TorsoControlPipeline()
animation_frames = []

for frame_index, key_points in enumerate(filtered_keypoint_frames):
    tracked_controls = torso_pipeline.process(key_points)
    animation_frames.append(
        torso_frame_as_dict(
            tracked_controls,
            frame_index=frame_index,
            timestamp_sec=frame_index / fps,
        )
    )
```

追跡 ID が `key_points.data` に含まれる場合は人物ごとに独立した状態を保持する。追跡 ID がない場合は検出行番号を代用するため、
複数人の動画では先に人物追跡を適用する必要がある。

実動画から JSON と比較動画を直接生成する場合は次を実行する。

```powershell
$env:RF_HOME = "C:\work\code\ER-FlowScan\rf-detr\artifacts\models"
$env:PYTHONPATH = "src;.venv\Lib\site-packages"
python -m rfdetr_demo.animation.video_export `
  --source sample\mzoo.mov `
  --json artifacts\fukkachan_torso\mzoo_controls.json `
  --overlay artifacts\fukkachan_torso\mzoo_torso.mp4
```

`--frame-stride`で推論間隔、`--max-frames`で検証フレーム数、`--threshold`で人物検出しきい値、
`--keypoint-threshold`で体幹制御に採用する点の信頼度を指定できる。

## ふっかちゃんへのリターゲット

全身制御JSONから自動分割リグを動かすプレビュー動画を生成できる。

```powershell
python -m rfdetr_demo.animation.puppet_cli `
  --controls artifacts\fukkachan_torso\mzoo_fullbody_controls_15f.json `
  --rig artifacts\fukkachan_yukata_animator_package `
  --output artifacts\fukkachan_animation\fukkachan_mzoo.mp4
```

先頭フレームを基準姿勢として、以降の差分をキャラクターへ割り当てる。

- 骨盤中心: キャラクター全体の移動
- 胴体傾斜: 胴体と帯の回転
- 背骨曲率: 胴体レイヤーのS字ワープ
- 目の線: 頭の傾き
- 肩から手首: 左右腕の回転
- 腰から足首: 左右脚の回転
- 膝から足首: 足先の補助回転

人物の左右は解剖学的な左右であるのに対し、自動分割素材の`Left`と`Right`は画面上の左右なので、
正面人物では左右を反転して割り当てている。

プレビューと同時に`*.poses.json`を生成する。ここにはキャラクター側へ変換済みの移動量、角度、伸縮、
S字ワープ量が保存されるため、別のレンダラーやAfter Effectsへ渡すことができる。

### 二次動作

レンダラーは既定でキャラクター制御に減衰ばねを適用する。頭、胴体、腕、脚、足先で剛性と減衰率を分け、
腕と桃袋は胴体より遅く追従させる。比較用に無効化する場合は`--no-dynamics`を指定する。

先頭15フレームでの二階差分RMSは次のように改善した。

- 左腕角度: `1.06583`から`0.07948`
- 頭角度: `0.55305`から`0.03896`
- 水平移動: `0.79986`から`0.07571`

60フレームの実モデル検証結果は次の通り。

- 人物追跡数: 1
- 体幹点欠損率: `0.0`
- 体幹正規化ジッター: `0.00753`
- 左腕角度範囲: `0.0`から`23.63`度
- 右腕角度範囲: `0.0`から`15.46`度
- 頭角度範囲: `-4.04`から`1.28`度
- 水平重心移動: `-21.58`から`10.71`px
- S字ワープ量: `-28.15`から`0.0`px

検証成果物は`artifacts/fukkachan_animation/fukkachan_mzoo_rig_v2d_dynamics_60f.mp4`である。

### 足接地と足上げ

各足首の骨盤相対Y座標を基準フレームと比較し、キャラクター側の足上げ量へ変換する。
足上げ量が3px以下の足は接地と判定し、その脚レイヤーでは全身の上下移動を相殺して床位置を固定する。
遊脚は足上げ量を減衰ばねで平滑化して持ち上げる。

60フレーム検証では次の結果になった。

- 右足接地: 56フレーム
- 右足接地中の上下変動範囲: `7.13px`から`0.0px`
- 左足遊脚: 40フレーム
- 左足最大上げ量: `16.29px`
- 足接地版動画: `artifacts/fukkachan_animation/fukkachan_mzoo_rig_v2e_footlock_60f.mp4`

現状の接地判定は先頭フレームを床基準にする2D推定である。ジャンプを含む動画では両足が上がった場合に接地固定を解除し、
床位置推定を別状態として持つ拡張が必要になる。

### 間引き推論と元fps補間

長尺動画では`video_export`の`--frame-stride`でRF-DETRの推論回数を削減できる。
JSONには`source_frame_count`, `source_frames_read`, `complete_source`を保存し、元動画を最後まで読み切ったJSONだけを
自動的に元fpsへ補間する。途中検証JSONは全尺へ引き伸ばさない。

```powershell
python -m rfdetr_demo.animation.video_export `
  --source sample\mzoo.mov `
  --json artifacts\fukkachan_torso\mzoo_fullbody_controls_full_stride6.json `
  --overlay artifacts\fukkachan_torso\mzoo_fullbody_torso_full_stride6.mp4 `
  --frame-stride 6

python -m rfdetr_demo.animation.puppet_cli `
  --controls artifacts\fukkachan_torso\mzoo_fullbody_controls_full_stride6.json `
  --rig artifacts\fukkachan_rig_v2\package `
  --output artifacts\fukkachan_animation\fukkachan_mzoo_rig_v2h_full_40fps.mp4
```

補間を行わず推論キーフレームだけを確認する場合は`--native-keyframes`を指定する。
角度パラメータは最短角度方向、接地状態は近い側のキーフレーム、その他の値は線形補間し、補間後に減衰ばねを適用する。

全編検証結果:

- 元動画: 1270フレーム、40.638fps、31.252秒
- RF-DETR推論: 212フレーム、stride 6
- 人物追跡: 1、欠損率0%
- 完成動画: 1270フレーム、40.638fps、31.252秒
- 左右腕角度: 約`-38.6`から`38.8`度
- 頭角度: 約`-20.3`から`20.5`度
- 胴体角度: 約`-16.0`から`4.1`度
- S字ワープ: 約`-18.1`から`0.0`px
- 左足接地: 466フレーム
- 右足接地: 652フレーム
- 完成動画: `artifacts/fukkachan_animation/fukkachan_mzoo_rig_v2h_full_40fps.mp4`

### 自動分割素材の制約

自動分割で覆われない領域は残差レイヤーで補い、各パーツ境界も元画像の色で拡張している。
ただし、元の1枚絵で隠れている肩、首、袖の内側は復元できない。頭や腕を大きく回すと白い継ぎ目が露出するため、
完成版では次の部分を手修正した透明レイヤーへ差し替える必要がある。

- 頭の背面と首の接続部
- 左右の袖と肩の下地
- 手の付け根
- 帯の後ろに隠れた胴体
- 左右の脚と足の重なり部分

## ニュートラル姿勢リグ v2

単一画像で隠れていた接続部を減らすため、元画像を厳密な外観参照として、腕と脚が重ならない正面ニュートラル姿勢を
組み込み画像編集で生成した。マゼンタ背景をクロマキー除去し、9個の専用レイヤーとピボットを持つmanifestへ変換している。

- 透明ニュートラル素材: `artifacts/fukkachan_rig_v2/fukkachan_neutral_v2.png`
- レイヤーパッケージ: `artifacts/fukkachan_rig_v2/package`
- 分割スクリプト: `scripts/fukkachan_prepare_neutral_rig.py`
- 検証動画: `artifacts/fukkachan_animation/fukkachan_mzoo_rig_v2c_15f.mp4`

```powershell
python scripts\fukkachan_prepare_neutral_rig.py `
  --input artifacts\fukkachan_rig_v2\fukkachan_neutral_v2.png `
  --output artifacts\fukkachan_rig_v2\package
```

この素材は首・肩・脚の独立性を優先した生成補助素材であり、元画像と輪郭や細部が完全一致する保証はない。
公開・納品前には元キャラクターとのデザイン照合が必要である。

### 連続メッシュレンダリング

初期の剛体レイヤー方式では、自動マスクが角、顔輪郭、袖、帯など本来連続すべき線を横断し、回転時に切断面が露出した。
`rig_v2`では`render_mode="continuous_mesh"`を指定し、表示面を全身透明PNG 1枚へ変更した。

頭、胴体、左右肩、左右脚のピボットは切断点ではなく、周辺画素へ滑らかに影響する変形ハンドルとして扱う。
各ハンドルの影響はガウス重みで減衰し、複数ハンドルの変位を合成して全身画像を1回だけ`cv2.remap`する。
このため次の線を切らずに変形できる。

- 頭と角、耳、顔輪郭
- 首と浴衣の襟
- 袖と胴体
- 帯と浴衣
- 胴体と脚

剛体レイヤーより腕の独立可動量は小さくなるが、キャラクターの輪郭連続性を優先する。

- 連続メッシュ全編動画: `artifacts/fukkachan_animation/fukkachan_mzoo_mesh_full_40fps.mp4`
- フレーム数: 1270
- fps: 40.638
- 尺: 31.252秒
- 解像度: 1254×1254

旧`fukkachan_mzoo_rig_v2*.mp4`は比較用であり、最終候補としては使用しない。

## 姿勢パラメータ

- `body_lean_deg`: 胴体中心線の左右傾斜
- `shoulder_roll_deg`: 肩線角度
- `pelvis_roll_deg`: 骨盤線角度
- `torso_twist_2d_deg`: 肩線と骨盤線の角度差
- `spine_curvature`: アニメーション用 S 字曲率
- `waist_compression_left/right`: 左右脇腹の相対圧縮量
- `torso_height_px`: 首付け根から骨盤中心までの距離
- `shoulder_width_px`, `pelvis_width_px`: 肩幅と骨盤幅

`spine_curvature` は肩と骨盤の逆回転から生成した演出用の推定値であり、実際に観測した背骨位置ではない。
独自 25 点モデルが完成した後は、胸郭、へそ、左右腰の実測点を優先し、この仮想曲率を補助値へ切り替える。

## 品質評価

`evaluate_torso_sequence()` は、指定した制御点について欠損率、平均信頼度、体幹長で正規化した二階差分ジッターを返す。
同じ評価クリップに対して平滑化前後を計測し、動作遅延を増やさずジッターが下がる設定を選ぶ。

`sample/mzoo.mov`の先頭15フレームをCPUで処理した初期基準は次の通り。

- 追跡人物数: 1
- 体幹点欠損率: 0.0
- 正規化ジッター: 0.00667
- 平均体幹信頼度: 0.99938
- 2Dひねり推定範囲: 0.33度から14.38度

この値は短い正面映像での配線確認用であり、モデル精度の最終評価値ではない。
S字動作、側面、遮蔽を含む固定評価クリップで再計測する必要がある。

## 次の段階

1. 実際のダンス動画から体幹制御 JSON と比較オーバーレイを出力する。
2. 300 から 500 フレームの固定評価セットを作る。
3. COCO17 と拡張制御点の欠損率、ジッター、角度差を記録する。
4. 体幹 8 点を追加した COCO 25 点アノテーション仕様を確定する。
5. RF-DETR の追加学習後、`source="derived"` を `source="detected"` へ段階的に置き換える。
