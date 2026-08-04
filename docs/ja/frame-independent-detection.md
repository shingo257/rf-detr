# フレーム独立推論と検出人数の点滅

RF-DETR 動画デモ（`rfdetr_demo`）において、画面上の人数と GUI 上の検出数が一致しない・フレームごとに 4〜6 と揺れる問題について、原因・仕様上の制約・対策の方向性をまとめた資料です。

---

## 1. 概要

| 項目 | 内容 |
|------|------|
| 現象 | 画面上は 5 人いるのに、検出数が 4〜6 とフレームごとに変動する |
| 典型例 | 中央付近（x ≈ 500）の人物がフレームによって消える／6 件になるときは端（x ≈ 130）の二重検出 |
| 対象タスク | 主にキーポイント推論（`make_keypoint_callback`） |
| 結論 | **RF-DETR 本体はフレーム独立推論が前提**。人数の安定化は **アプリ層（`rfdetr_demo`）で後処理する設計** が現実的 |

---

## 2. 観測された問題

### 2.1 GUI 上の検出数

動画デモ GUI の「解析状況」では **累積** `total_detections`（処理済み全フレームの検出件数の合計）が表示されます。  
キーポイントタスクでは各フレームで `key_points_for_display()` 後の件数が加算されるため、GUI 数字は単調増加し、**4〜6 の点滅そのものは表示されません**。

点滅は **フレーム単位のインスタンス数**（監査ログ `[機密] #NN … N 人`、注釈付きプレビュー、`probe_person_count.py`）で確認します。

### 2.2 機密フレーム監査（先頭 20 フレーム）

`confidential/media/input/mn1-2.mov` を閾値 0.6・キーポイント・ヒートマップ表示で解析した監査結果（`confidential/audit/`）:

| 指標 | 値 |
|------|-----|
| 監査フレーム数 | 20 |
| 検出数の範囲 | 4〜6 |
| 平均 | 5.0 |
| 0 件フレーム | なし |
| ヒューリスティック評価 | 概ね正常（ただし人数の不安定さは残る） |

視覚的には 5 人が常に存在するシーンでも、モデル出力上は **4 人または 6 人** になるフレームがある。

### 2.3 診断スクリプト

`uv run rfdetr-demo probe-count` を実行すると、フレームごとの bbox 中心 x 座標と件数が `artifacts/person_count_probe.json` に記録されます。

閾値 0.6 での傾向:

- **安定して検出される 4 位置**: x ≈ 130, 365, 779, 1049
- **点滅する中央**: x ≈ 500〜514（4 人フレームでは欠落、5 人フレームでは出現）
- **6 人になる原因**: 端（x ≈ 130）または中央（x ≈ 511）付近の **二重検出**

---

## 3. なぜ「フレーム独立」なのか

### 3.1 RF-DETR 本体（upstream）

RF-DETR の公開 API `RFDETR.predict()` は **静止画（または互いに独立した画像リスト）** 向けです。

- 1 回の forward pass = 1 枚（またはバッチ内の各枚）の画像
- フレーム間の hidden state・track ID・時系列メモリは **モデル内部に存在しない**
- キーポイントモデルも、そのフレームの `KeyPoints` を返すだけ

これは RF-DETR 固有のバグではなく、**画像向け DETR 系モデルの一般的な設計** です。  
`predict([f1, f2, ...])` で複数フレームをまとめて投げても、**フレーム間の関連付けは行われません**。

### 3.2 動画デモパイプライン（`rfdetr_demo`）

動画は `process_video()` が OpenCV で 1 フレームずつ読み、コールバック内で毎回 `model.predict()` を呼びます。

```
動画ファイル
  └─ process_video()  … フレーム読み込みループ
       └─ callback(frame_bgr, frame_index)
            └─ model.predict(frame_rgb, threshold=...)
                 └─ （任意）temporal_filter.apply()  … 関節の後処理のみ
```

関連コード:

| ファイル | 役割 |
|----------|------|
| `src/rfdetr_demo/inference/video_io.py` | フレームループ |
| `src/rfdetr_demo/inference/callbacks.py` | フレームごとの `predict()` |
| `src/rfdetr/detr.py` | RF-DETR 推論 API |

**「動画なのにフレーム独立」** という違和感はもっともですが、現状は **モデル仕様 + デモの実装方針** の両方による結果です。

---

## 4. 既存の時間方向処理と限界

フォークにはすでに時間方向のコンポーネントがありますが、**検出インスタンス数（人数）の安定化が主目的ではありません**。

| コンポーネント | ファイル | 役割 | 人数点滅への効果 |
|----------------|----------|------|------------------|
| `PersonAssociator` | `src/rfdetr_demo/tracking/person_associator.py` | IoU + ハンガリアン法でフレーム間 bbox を対応付け | △ トラック ID 用。検出の追加・削除はしない |
| `KeypointTemporalFilter` | `src/rfdetr_demo/inference/temporal_filter.py` | 関節の速度・共分散・振動を抑制 | × インスタンス数は変えない |
| `TunePreviewCache` | 自動チューニング試走用 | 試走結果の再描画 | × 推論自体はフレーム独立のまま |
| `auto_tune_proposer` | `src/rfdetr_demo/tuning/auto_tune_proposer.py` | 不安定時に閾値を提案 | △ 人数不足時に閾値を **上げる** 方向だと悪化しうる |

キーポイントコールバックの流れ（簡略）:

1. `model.predict()` … そのフレームだけの生検出
2. `temporal_filter.apply()` … 関節位置の平滑化（人数は不変）
3. `key_points_for_display()` … 表示閾値でフィルタ
4. `stats["total_detections"] += len(display_points)` … GUI 表示件数

---

## 5. 原因の整理

### 5.1 構造的原因（必ず起きうる）

1. **フレーム独立推論** … 前フレームの検出結果をモデル入力に使わない
2. **DETR の query 競合** … 重なり・類似姿勢で query が別物体に割り当たる
3. **閾値境界** … 中央人物の confidence が 0.6 前後で揺れると、1 フレームの差で in/out

### 5.2 件数が「多く」見える原因

- 同一人物に対する **二重 bbox**（IoU-NMS 未適用）
- 端と中央で別 ID としてカウント

### 5.3 件数が「少なく」見える原因

- 中央人物の confidence が一時的に閾値未満
- オクルージョン・重なりで bbox / keypoint head のスコア低下
- auto-tune が閾値を上げ、ギリギリの検出を落とす

---

## 6. 「仕方ない」の範囲

### RF-DETR 本体だけを使う場合

| 質問 | 答え |
|------|------|
| フレーム独立は避けられるか | **現 API・学習範囲では避けられない** |
| バッチ推論で時間一貫性は得られるか | **得られない**（各画像は独立処理） |
| 動画向け API は upstream にあるか | **ない**（画像 DETR として提供） |

### パイプライン全体として

| 質問 | 答え |
|------|------|
| 人数のブレは完全に不可避か | **いいえ**。後処理・トラッキングで改善可能 |
| upstream を変えずに対策できるか | **はい**（`rfdetr_demo` 層） |

---

## 7. 対策案（未実装・検討候補）

優先度の目安順:

### 7.1 アプリ層（短期・低リスク）

1. **IoU-NMS** … `key_points.data["xyxy"]` に対し、同一人物の二重検出を 6→5 に抑制
2. **閾値の調整** … 0.50〜0.55 付近で中央人物の再現率を確認（トレードオフ: 誤検出増）
3. **トラックベースの hold** … `PersonAssociator` を拡張し、1〜2 フレーム欠落時は前フレーム bbox を補完表示
4. **トラック単位 hysteresis** … 新規検出は高閾値、継続トラックは低閾値で維持
5. **期待人数 N** … ダンス等で定員が分かる場合、トラック数を N に cap / fill

### 7.2 自動チューニング

- 人数 **不足** が主因のときは閾値を **下げる** 方向の提案ロジックを追加
- 不安定さの指標を「件数の分散」だけでなく「トラック途切れ率」にも拡張

### 7.3 モデル・学習（中長期）

- 群衆・重なりシーン向け fine-tune
- Video Instance Segmentation / MOT 向けアーキテクチャ（TrackFormer 系など）への移行検討  
  ※ 現行 RF-DETR には含まれない

---

## 8. 関連ファイル・コマンド

| 種別 | パス / コマンド |
|------|-----------------|
| 動画ループ | `src/rfdetr_demo/inference/video_io.py` |
| 推論コールバック | `src/rfdetr_demo/inference/callbacks.py` |
| RF-DETR API | `src/rfdetr/detr.py` |
| 人物対応付け | `src/rfdetr_demo/tracking/person_associator.py` |
| 関節時間フィルタ | `src/rfdetr_demo/inference/temporal_filter.py` |
| 人数診断 | `uv run rfdetr-demo probe-count` → `artifacts/person_count_probe.json`（`--mode raw\|nms\|stabilize`） |
| 検出安定化 | `src/rfdetr_demo/tracking/pipeline.py`（`PersonTrackPipeline`、既定 ON、`RFDETR_DETECTION_STABILIZER=0` で無効） |
| 機密フレーム監査 | `src/rfdetr_demo/media/frame_audit.py` → `confidential/audit/` |
| 監査無効化 | 環境変数 `RFDETR_FRAME_AUDIT=0` |

---

## 9. まとめ

1. **RF-DETR は画像 DETR として設計されており、動画でもフレームごとに独立推論するのが現状の仕様です。**
2. **`rfdetr_demo` はその API をそのまま動画ループに載せているため、時間一貫性はモデル側では保証されません。**
3. **既存の `KeypointTemporalFilter` / `PersonAssociator` は関節平滑化・ID 対応が中心で、検出人数の安定化には直結しません。**
4. **4〜6 人の点滅は、中央人物の閾値境界と二重検出の組み合わせで説明できます。**
5. **upstream を変えず、NMS・トラック hold・閾値戦略などアプリ層で大幅に改善できる余地があります。**

---

**最終更新**: 2026-06-19  
**関連**: [refactor-boundaries.md](refactor-boundaries.md), [confidential-media-policy.md](confidential-media-policy.md), [ER-FlowScan スイート横断対策](../../../docs/rfdetr-person-count-flicker-cross-app.md)
