# ふっかちゃん浴衣キーポイントアニメーション計画

## 目的

RF-DETR の keypoint タスクで検出した COCO 17点姿勢を、`ふっかちゃん浴衣.jpg` の 2D パペットへ割り当て、手足、腰、顔角度を入力人物の動きに追従させる。

## 現在の成果物

- 入力画像: `C:\work\code\ER-FlowScan\rf-detr\ふっかちゃん浴衣.jpg`
- 自動分割パッケージ: `C:\work\code\ER-FlowScan\rf-detr\artifacts\fukkachan_yukata_animator_package`
- 確認用シート: `C:\work\code\ER-FlowScan\rf-detr\artifacts\fukkachan_yukata_animator_package\contact_sheet.png`

この自動分割は初期リグ確認用であり、高品質版の最終素材ではない。元画像が 1枚絵なので、腕や足の裏側など「画像に存在しない隠れ部分」は自動復元できない。完全再現に近づけるには、最終的に手修正または生成補完したレイヤー素材が必要。

## 推奨パイプライン

1. RF-DETR keypoint 動画解析
   - 入力動画から 17点 COCO キーポイントをフレーム単位で出力する。
   - 1人だけを対象にする場合は、`track_id` を固定して中央または最大信頼度の人物を選ぶ。
   - 低信頼度点は前後フレームで補間し、急な飛びを temporal filter で抑える。

2. キーポイント正規化
   - 肩幅、腰幅、首-腰距離でスケールを正規化する。
   - 入力人物の重心を avatar root に合わせる。
   - 左右反転が必要な映像では、左肩/右肩の x 座標で自動判定する。

3. ふっかちゃん用リターゲット
   - 顔角度: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear` から頭の回転と傾きを推定する。
   - 胴体/腰: `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip` から body root、腰回転、上下揺れを出す。
   - 腕: 肩、肘、手首から左右袖と手の回転を出す。
   - 足: 腰、膝、足首から左右足の回転を出す。元絵の足は短いため、角度は大きくしすぎず 0.45-0.65 倍程度に抑える。

4. レイヤー描画
   - 最低レイヤー: `body`, `head`, `left_arm`, `right_arm`, `left_hand`, `right_hand`, `left_leg`, `right_leg`, `left_foot`, `right_foot`
   - 追加推奨: 葉っぱ/角、耳、うちわ、帯、口、頬を別レイヤーにする。
   - 高品質化では、各パーツを元画像から切るだけでなく、関節で露出する欠け部分を描き足す。

5. レンダリング
   - 量産/自動化: Python + Pillow/OpenCV でパーツを affine 変換して PNG/MP4 を出力する。
   - 最高品質: After Effects、Adobe Character Animator、Live2D、Spine のいずれかにレイヤー PSD を渡し、キーポイント JSON/CSV を制御トラックとして読み込む。

## 中間 JSON 形式

```json
{
  "fps": 30,
  "width": 1920,
  "height": 1080,
  "frames": [
    {
      "index": 0,
      "track_id": 1,
      "keypoints": {
        "nose": [960.0, 210.0, 0.94],
        "left_shoulder": [850.0, 420.0, 0.91],
        "right_shoulder": [1065.0, 425.0, 0.90]
      }
    }
  ]
}
```

COCO 17点名は `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle` を使う。

## 品質基準

- 関節の回転中心が絵の線からずれない。
- 頭と胴体の前後関係が破綻しない。
- 袖、手、足が胴体から離れすぎない。
- キーポイント欠落時に一瞬で飛ばず、補間またはホールドする。
- 最終動画は 30fps 以上、透明背景または任意背景合成に対応する。

## 次に実装するもの

1. RF-DETR keypoint 出力を JSON に保存する exporter。
2. `manifest.json` のパーツ中心を pivot として使う簡易 Python renderer。
3. 手修正済みレイヤー素材を差し替えて再レンダリングする workflow。
