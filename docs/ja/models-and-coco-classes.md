# RF-DETR モデル構成と対応クラス解説

本ドキュメントは、RF-DETR フォーク（ER-FlowScan）向けの技術リファレンスです。  
**どの推論モデルが何を検出できるか**、**事前学習チェックポイントの違い**、**COCO 80 クラスの一覧**をまとめています。

- 公式 API リファレンス: [rfdetr.roboflow.com](https://rfdetr.roboflow.com)
- 日本語ドキュメント索引: [docs/ja/README.md](README.md)
- Vast.ai 外部 GPU 組み込み: [docs/ja/vast-ai-integration-guide.md](vast-ai-integration-guide.md)
- コード上のクラス定義: [`src/rfdetr/assets/coco_classes.py`](../src/rfdetr/assets/coco_classes.py)
- モデルバリアント: [`src/rfdetr/variants.py`](../src/rfdetr/variants.py)

---

## 1. 重要な前提

### 1.1 推論モデルはタスクごとに別物

RF-DETR は **1 つの万能モデルをモード切替する方式ではありません**。  
検出・セグメンテーション・キーポイントは **別 Python クラス・別チェックポイント（`.pth` / `.pt`）・別出力形式** です。

```
共通基盤（DINOv2 系バックボーン + DETR 系）
        │
        ├── 検出モデル      RFDETRNano … RFDETRLarge     → Detections（矩形 + クラス）
        ├── セグモデル      RFDETRSegNano … RFDETRSeg2XL → Detections + マスク
        └── キーポイント    RFDETRKeypointPreview        → KeyPoints（17 関節）
```

| 誤解 | 正しい理解 |
|------|------------|
| 人検知専用モデル | **検出版**は COCO 80 クラス全体向け。`person` はその 1 つ |
| セグ版は検出版の「機能追加」 | **別重み**で学習された **別モデル**（マスク用ヘッド付き） |
| キーポイント版も COCO 80 物体を検出 | **人の 17 関節**が主目的（Preview）。汎用物体検出向けではない |
| 何でもそのまま検出できる | **事前学習済みクラス以外**（例: ジャガイモ）は **ファインチューニング必須** |

### 1.2 事前学習データセット

| タスク | 事前学習データ |
|--------|----------------|
| 物体検出 | [Microsoft COCO 2017 Detection](https://cocodataset.org/)（80 クラス） |
| インスタンスセグメンテーション | COCO 2017 Instance Segmentation（検出と同じ 80 クラス + マスク） |
| キーポイント（Preview） | COCO 2017 Person Keypoints（**person** の 17 点） |

### 1.3 `num_classes = 90` について

設定ファイル（`ModelConfig`）では `num_classes: int = 90` です。  
これは **COCO の category ID が 1〜90 の範囲で欠番がある**（12, 26, 29, 30 … など未使用 ID あり）ため、モデル内部の分類次元が 90 スロットになる technical な事情です。

**ユーザーが参照すべきラベル一覧は 80 クラス**（下表）であり、推論結果の `class_id` は [`COCO_CLASSES`](../src/rfdetr/assets/coco_classes.py) のキー（COCO category ID）に対応します。

---

## 2. モデルバリアント一覧

### 2.1 物体検出（Object Detection）

| Size | Python クラス | 既定重みファイル | 解像度 | 出力 | ライセンス |
|:----:|---------------|------------------|--------|------|------------|
| N | `RFDETRNano` | `rf-detr-nano.pth` | 384×384 | 矩形 + クラス | Apache 2.0 |
| S | `RFDETRSmall` | `rf-detr-small.pth` | 512×512 | 同上 | Apache 2.0 |
| M | `RFDETRMedium` | `rf-detr-medium.pth` | 576×576 | 同上 | Apache 2.0 |
| L | `RFDETRLarge` | `rf-detr-large.pth` | 704×704 | 同上 | Apache 2.0 |
| XL | `RFDETRXLarge` △ | `rfdetr_plus` 経由 | 700×700 | 同上 | PML 1.0 |
| 2XL | `RFDETR2XLarge` △ | `rfdetr_plus` 経由 | 880×880 | 同上 | PML 1.0 |

```python
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRMedium()
detections = model.predict("image.jpg", threshold=0.5)
# detections.class_id → COCO category ID
# COCO_CLASSES[detections.class_id[i]] → クラス名（英語）
```

### 2.2 インスタンスセグメンテーション

| Size | Python クラス | 既定重みファイル | 解像度 | 出力 |
|:----:|---------------|------------------|--------|------|
| N | `RFDETRSegNano` | `rf-detr-seg-n.pt` | 312×312 | 矩形 + クラス + **マスク** |
| S | `RFDETRSegSmall` | `rf-detr-seg-s.pt` | 384×384 | 同上 |
| M | `RFDETRSegMedium` | `rf-detr-seg-m.pt` | 432×432 | 同上 |
| L | `RFDETRSegLarge` | `rf-detr-seg-l.pt` | 504×504 | 同上 |
| XL | `RFDETRSegXLarge` | `rf-detr-seg-xl.pt` | 624×624 | 同上 |
| 2XL | `RFDETRSeg2XLarge` | `rf-detr-seg-xxlarge.pt` | 768×768 | 同上 |

- 検出対象クラスは **COCO 80 クラスと同じ**（ラベル空間は検出版と共通）。
- `predict()` の戻り値は `supervision.Detections` で、マスク情報を含む。

### 2.3 キーポイント検出（Preview）

| モデル | Python クラス | 既定重み | 解像度 | 出力 |
|--------|---------------|----------|--------|------|
| Keypoint Preview | `RFDETRKeypointPreview` | `rf-detr-keypoint-preview-xlarge.pth` | 576×576 | `supervision.KeyPoints` |

- スキーマ: `num_keypoints_per_class = [0, 17]` → **person（COCO ID=1）に 17 関節**
- 姿勢推定・人数カウントの補助など **人体向け**
- API は Preview のため将来変更の可能性あり

```python
from rfdetr import RFDETRKeypointPreview

model = RFDETRKeypointPreview()
key_points = model.predict("image.jpg", threshold=0.5)
# key_points.xy → (N, 17, 2)  ピクセル座標
```

---

## 3. COCO 80 クラス一覧（検出・セグメンテーション）

以下が **そのまま（ファインチューニングなし）** で名前付き検出できるクラスです。  
`class_id` は COCO category ID（`COCO_CLASSES` のキー）です。

### 3.1 人・動物

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 1 | person | 人 |
| 16 | bird | 鳥 |
| 17 | cat | 猫 |
| 18 | dog | 犬 |
| 19 | horse | 馬 |
| 20 | sheep | 羊 |
| 21 | cow | 牛 |
| 22 | elephant | 象 |
| 23 | bear | 熊 |
| 24 | zebra | シマウマ |
| 25 | giraffe | キリン |

### 3.2 乗り物

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 2 | bicycle | 自転車 |
| 3 | car | 自動車 |
| 4 | motorcycle | オートバイ |
| 5 | airplane | 飛行機 |
| 6 | bus | バス |
| 7 | train | 電車 |
| 8 | truck | トラック |
| 9 | boat | ボート |

### 3.3 屋外・道路

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 10 | traffic light | 信号機 |
| 11 | fire hydrant | 消火栓 |
| 13 | stop sign | 一時停止標識 |
| 14 | parking meter | パーキングメーター |
| 15 | bench | ベンチ |

### 3.4 バッグ・スポーツ用品

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 27 | backpack | リュック |
| 28 | umbrella | 傘 |
| 31 | handbag | ハンドバッグ |
| 32 | tie | ネクタイ |
| 33 | suitcase | スーツケース |
| 34 | frisbee | フリスビー |
| 35 | skis | スキー |
| 36 | snowboard | スノーボード |
| 37 | sports ball | スポーツボール |
| 38 | kite | 凧 |
| 39 | baseball bat | 野球バット |
| 40 | baseball glove | 野球グローブ |
| 41 | skateboard | スケートボード |
| 42 | surfboard | サーフボード |
| 43 | tennis racket | テニスラケット |

### 3.5 食器・食べ物

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 44 | bottle | 瓶 |
| 46 | wine glass | ワイングラス |
| 47 | cup | カップ |
| 48 | fork | フォーク |
| 49 | knife | ナイフ |
| 50 | spoon | スプーン |
| 51 | bowl | ボウル |
| 52 | banana | バナナ |
| 53 | apple | りんご |
| 54 | sandwich | サンドイッチ |
| 55 | orange | オレンジ |
| 56 | broccoli | ブロッコリー |
| 57 | carrot | にんじん |
| 58 | hot dog | ホットドッグ |
| 59 | pizza | ピザ |
| 60 | donut | ドーナツ |
| 61 | cake | ケーキ |

> **注意:** **potato（ジャガイモ）は COCO に含まれません。** コンベア上のジャガイモを COCO 事前学習モデルにかけると、検出漏れや `apple` / `sports ball` / `bowl` などへの **誤ラベル** が起き得ます。これはモデル故障ではなく **クラス定義の不一致** です。

### 3.6 家具・家電

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 62 | chair | 椅子 |
| 63 | couch | ソファ |
| 64 | potted plant | 鉢植え |
| 65 | bed | ベッド |
| 67 | dining table | ダイニングテーブル |
| 70 | toilet | トイレ |
| 72 | tv | テレビ |
| 73 | laptop | ノート PC |
| 74 | mouse | マウス |
| 75 | remote | リモコン |
| 76 | keyboard | キーボード |
| 77 | cell phone | 携帯電話 |
| 78 | microwave | 電子レンジ |
| 79 | oven | オーブン |
| 80 | toaster | トースター |
| 81 | sink | シンク |
| 82 | refrigerator | 冷蔵庫 |

### 3.7 その他

| ID | 英語名 | 日本語の目安 |
|:--:|--------|-------------|
| 84 | book | 本 |
| 85 | clock | 時計 |
| 86 | vase | 花瓶 |
| 87 | scissors | はさみ |
| 88 | teddy bear | ぬいぐるみ |
| 89 | hair drier | ヘアドライヤー |
| 90 | toothbrush | 歯ブラシ |

### 3.8 プログラムからクラス名を取得

```python
from rfdetr.assets.coco_classes import COCO_CLASSES, COCO_CLASS_NAMES

# ID → 名前
label = COCO_CLASSES[18]  # "dog"

# ソート済み 80 クラス名リスト（ID 順）
all_names = COCO_CLASS_NAMES
```

---

## 4. COCO 17 キーポイント（キーポイント版のみ）

`RFDETRKeypointPreview` が推定する人体関節（COCO Person Keypoints 標準）。  
インデックス `0..16` は以下の順序です（[COCO keypoints 仕様](https://cocodataset.org/#format-data) に準拠）。

| Index | 英語名 | 日本語 |
|:-----:|--------|--------|
| 0 | nose | 鼻 |
| 1 | left_eye | 左目 |
| 2 | right_eye | 右目 |
| 3 | left_ear | 左耳 |
| 4 | right_ear | 右耳 |
| 5 | left_shoulder | 左肩 |
| 6 | right_shoulder | 右肩 |
| 7 | left_elbow | 左肘 |
| 8 | right_elbow | 右肘 |
| 9 | left_wrist | 左手首 |
| 10 | right_wrist | 右手首 |
| 11 | left_hip | 左腰 |
| 12 | right_hip | 右腰 |
| 13 | left_knee | 左膝 |
| 14 | right_knee | 右膝 |
| 15 | left_ankle | 左足首 |
| 16 | right_ankle | 右足首 |

評価指標は **OKS ベースの AP<sub>50:95</sub>**（COCO キーポイント標準）。

---

## 5. 出力形式（Supervision）

| タスク | 戻り値型 | 主なフィールド |
|--------|----------|----------------|
| 検出 | `sv.Detections` | `xyxy`, `class_id`, `confidence`, `metadata["source_image"]` |
| セグメンテーション | `sv.Detections` | 上記 + マスク（`mask`） |
| キーポイント | `sv.KeyPoints` | `xy` (N,K,2), `keypoint_confidence`, `detection_confidence`, `data["xyxy"]` |

`threshold` は **インスタンス（検出）信頼度** のしきい値です。キーポイント版では各関節の `keypoint_confidence` を別途フィルタできます。

---

## 6. カスタムクラス（ファインチューニング）

COCO 80 クラス **以外**（ジャガイモ、異物、自社部品など）を検出する手順:

1. **COCO 形式** または **YOLO 形式** でデータセットを用意（[dataset-formats](https://rfdetr.roboflow.com/learn/train/dataset-formats/)）
2. `num_classes` をデータセットのクラス数に合わせて学習
3. 学習後のチェックポイントを `RFDETRMedium(pretrain_weights="path/to/checkpoint.pth")` のように指定

Roboflow プラットフォーム、[Google Colab チュートリアル](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)、または `rfdetr` CLI で学習可能。

---

## 7. ER-FlowScan Suite との使い分け

| 用途 | 推奨モジュール | 理由 |
|------|----------------|------|
| コンベア上のジャガイモ計数 | **Detect**（YOLO 系） | 自社クラス向けに学習済み想定 |
| 「〇〇を探して」（自由文） | **Search**（FlashFind / VLM） | ゼロショット探索 |
| COCO 物体（人・車など）の高速検出 | **rf-detr 検出版** | COCO 事前学習そのまま |
| 輪郭マスクが必要 | **rf-detr セグ版** | ピクセル単位領域 |
| 人体姿勢・関節 | **rf-detr キーポイント版** | 17 点スケルトン |
| 自社固定クラスの高速 DETR | **rf-detr + FT** | 精度・速度のバランス |

---

## 8. 参考リンク

| 資料 | URL |
|------|-----|
| 検出の実行 | [docs/learn/run/detection.md](../docs/learn/run/detection.md) |
| セグメンテーション | [docs/learn/run/segmentation.md](../docs/learn/run/segmentation.md) |
| キーポイント | [docs/learn/run/keypoints.md](../docs/learn/run/keypoints.md) |
| 学習・データセット形式 | [docs/learn/train/index.md](../docs/learn/train/index.md) |
| ローカル動画デモ | [README.md — Local Video Demo](../../README.md#local-video-demo--ローカル動画デモ) |
| 動画デモ GUI / Vast.ai | [README.md — Video Demo GUI](../../README.md#video-demo-gui--動画デモ-gui) |
| Vast.ai 組み込みガイド | [vast-ai-integration-guide.md](vast-ai-integration-guide.md) |

---

*最終更新: ER-FlowScan フォーク向けローカルドキュメント（upstream: [roboflow/rf-detr](https://github.com/roboflow/rf-detr)）*
