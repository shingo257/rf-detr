# リファクタ baseline メトリクス（2026-06-20 Phase 13 更新）

Phase 13 時点の計測値。以降の PR は本ファイルと [refactor-debt-register.md](refactor-debt-register.md) を基準に差分を記録する。

## コード規模

| 領域 | ファイル数 | 行数（概算） |
|------|-----------|-------------|
| `src/rfdetr/` | 111 | ~28,600 |
| `src/rfdetr_demo/` | 65 | ~9,500 |
| `scripts/` | 21 | ~880 |
| `tests/rfdetr_demo/` | 14+ | ~500 |

## God files（400 行超 — rfdetr_demo）

| パス | 行数 | 対象 Phase |
|------|------|------------|
| `media/tracking_audit.py` | 467 | Phase 18 |

### Phase 17 で解消（400 行以下）

| パス | 行数 | 分割先 |
|------|------|--------|
| `gui/panels/compute.py` | 339 | `controllers/vast_controller.py` |
| `gui/panels/io_task.py` | 344 | `controllers/tune_controller.py` |
| `gui/panels/job_runner.py` | 333 | `controllers/run_controller.py` |
| `tracking/detection_stabilizer.py` | 76 | `tracking/pipeline.py` 等（Phase 15） |

### 移行済み（Phase 9）

| 旧モジュール | 分割先 |
|-------------|--------|
| `uncertainty_viz.py` (~428) | `inference/uncertainty/{geometry,heatmap,styles}.py` + facade |
| `auto_tune.py` (~421) | `tuning/auto_tune_{types,metrics,proposer}.py` + facade |

### upstream（フォーク外）

| パス | 行数 |
|------|------|
| `src/rfdetr/detr.py` | ~1,828 |

## エントリポイント

| 用途 | 推奨コマンド |
|------|-------------|
| CLI 動画デモ | `uv run rfdetr-demo` |
| GUI | `uv run rfdetr-demo-gui` |
| Vast orphan 回収 | `uv run rfdetr-vast-cleanup` |
| 人数プローブ | `uv run rfdetr-demo probe-count` |
| 中央トラック監査 | `uv run rfdetr-demo audit-tracking` |
| クリップ品質解析 | `uv run rfdetr-demo analyze-clip` |
| 旧 scripts 診断 | DeprecationWarning 付き thin wrapper |

## テスト

| スイート | 件数 |
|---------|------|
| `tests/rfdetr_demo/` | **62** |
| `tests/scripts/` | 統合済み（削除） |

## 回帰アンカー（手動）

```bash
.venv\Scripts\python.exe -m pytest tests/rfdetr_demo
.venv\Scripts\python.exe -m rfdetr_demo.cli.main probe-count --mode stabilize --frames 20
.venv\Scripts\python.exe -m rfdetr_demo.cli.main audit-tracking --max-frames 60
.venv\Scripts\python.exe scripts\check_import_cycles.py
```

### 監査 baseline（mn1-2.mov 全編）

| 指標 | 値 | run_id |
|------|-----|--------|
| 中央トラック欠落 | 68 / 713 | `20260620T101049Z-ed8dabe5` |
| ID 切替 | 31 回 | 同上 |
| 最大連続欠落 | frame 543–571 | 同上 |

## 目標達成状況

- [x] GUI 400 行超ファイル **0 件**（Phase 17 完了）
- [ ] `rfdetr_demo` 主要モジュール **400行以下**（`tracking_audit.py` が残 — Phase 18）
- [x] 公開 API 正規化（`probe_video_size`, `vast.cli.run_vast_cli`）
- [x] GUI preflight 共通化（`gui/vast_preflight.py`）
- [x] `TuneJobState` 統合
- [x] 検出安定化 PoC（NMS + hold + ゴースト表示）
- [x] 追跡パイプライン統合（Phase 15 — `PersonTrackPipeline`, `TrackStore`, `bbox.py`）
- [ ] 試走 KPI（拒否率 -30%）— チューニング実験トラック

## 関連

- [refactor-master-plan.md](refactor-master-plan.md)
- [refactor-debt-register.md](refactor-debt-register.md)
- [refactor-boundaries.md](refactor-boundaries.md)
