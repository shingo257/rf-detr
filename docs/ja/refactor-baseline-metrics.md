# リファクタ baseline メトリクス

Phase 13–15 計画（tracking 統合）の計測基準。差分比較は本ファイルと [refactor-debt-register.md](refactor-debt-register.md) を使う。

**最終更新**: 2026-08-04

---

## Phase 13 起点スナップショット（計画策定時）

| 指標 | 値 |
|------|-----|
| `tests/rfdetr_demo` | **37** |
| 400 行超 God ファイル | **3**（再発） |
| 399 行付近 | `pipeline.py` 392, `io_task.py` 393 |

| パス | 行数 | 備考 |
|------|------|------|
| `tracking/detection_stabilizer.py` | 491 | Phase 15 分割対象 |
| `media/tracking_audit.py` | 467 | 監査 God ファイル |
| `gui/panels/job_runner.py` | 416 | GUI God ファイル |

監査 baseline（mn1-2.mov 全編）:

| 指標 | 値 | run_id |
|------|-----|--------|
| 中央トラック欠落 | 68 / 713 | `20260620T101049Z-ed8dabe5` |
| ID 切替 | 31 回 | 同上 |
| 最大連続欠落 | frame 543–571 | 同上 |

---

## Phase 15 完了後（現行）

### コード規模（概算）

| 領域 | ファイル数 |
|------|-----------|
| `src/rfdetr/` | 100 |
| `src/rfdetr_demo/` | 116 |
| `scripts/` | 8（shim 削除後; launcher / cycle check / mzoo / animation） |

### Tracking 分割結果

| パス | 行数 | 状態 |
|------|------|------|
| `tracking/detection_stabilizer.py` | — | Phase 9 で削除 |
| `tracking/pipeline.py` | 79 | `PersonTrackPipeline` |
| `tracking/track_store.py` | 195 | NMS / associate オーケストレーション |
| `tracking/track_hold.py` | 161 | hold / motion / ghost / cap |
| `tracking/track_models.py` | 72 | `TrackSnapshot` + geometry |
| `tracking/track_match.py` | 49 | Hungarian IoU match |
| `tracking/bbox.py` | 136 | IoU / NMS |
| `tracking/types.py` | 157 | settings + diagnostics |
| `inference/runner.py` | 238 | `run_demo` I/O |
| `inference/task_callback.py` | 145 | task callback 組立 |
| `media/tracking_audit.py` | 26 | `media/audit/` facade |

### 400 行超（現行）

| パス | 行数 | 状態 |
|------|------|------|
| （なし） | — | `vast/safety.py` は Phase 12 で分割済 |

### Phase 12 Vast safety 分割

| パス | 行数 | 役割 |
|------|------|------|
| `vast/safety.py` | 29 | 互換 facade |
| `vast/safety_settings.py` | 65 | `VastSafetySettings` |
| `vast/safety_lease.py` | 137 | lease persistence |
| `vast/safety_guardrails.py` | 264 | destroy / orphan / emergency |

### テスト

| スイート | 件数 |
|---------|------|
| `tests/rfdetr_demo/` | **166** |

---

## エントリポイント

| 用途 | 推奨コマンド |
|------|-------------|
| CLI 動画デモ | `uv run rfdetr-demo` |
| GUI | `uv run rfdetr-demo-gui` |
| 人数プローブ | `uv run rfdetr-demo probe-count` |
| 中央トラック監査 | `uv run rfdetr-demo audit-tracking` |
| クリップ品質解析 | `uv run rfdetr-demo analyze-clip` |
| Vast orphan 回収 | `uv run rfdetr-vast-cleanup` |
| 旧 scripts | DeprecationWarning 付き thin wrapper |

---

## 回帰アンカー

```bash
.venv/bin/python -m pytest tests/rfdetr_demo
.venv/bin/python -m rfdetr_demo.cli.main probe-count --mode stabilize --frames 20
.venv/bin/python -m rfdetr_demo.cli.main audit-tracking --max-frames 60
.venv/bin/python -m rfdetr_demo.cli.main audit-tracking --sticky-center-track --max-frames 60
.venv/bin/python scripts/check_import_cycles.py
```

Sticky 有効時の目標（全 713 フレーム）: 中央 ID 切替 **31 → ≤15**（手動検証）。

---

## 目標達成状況（Phase 13–15 + 12 + 9）

- [x] Phase 13: baseline / debt register / import cycle checker
- [x] Phase 15a: `types.py` + `bbox.py` 抽出
- [x] Phase 15b: `TrackStore` 統合、`KeypointTemporalFilter` から `_associator` 除去
- [x] Phase 15c: `PersonTrackPipeline` + `RFDETR_MAX_MISSED` / `RFDETR_STICKY_CENTER_TRACK`
- [x] Phase 14: `probe-count` / `audit-tracking` CLI サブコマンド + scripts wrapper
- [x] `detection_stabilizer.py` は 300 行以下の deprecated facade → Phase 9 で削除
- [x] Phase 12: `vast/safety.py` → settings / lease / guardrails
- [x] Phase 9: 6 deprecated facade 削除（tracking / vast / inference / gui）
- [x] Phase 8: scripts shim / thin wrapper 19 本削除
- [x] Phase 12a: `track_store` → models/match/hold、`runner` → `task_callback`
- [x] Phase 12b: GUI panels / vast_controller 分割（すべて &lt;300）
- [ ] 全編 sticky 監査（mn1-2.mov）— 機密動画はローカル実行

## 関連

- [refactor-master-plan.md](refactor-master-plan.md)
- [refactor-debt-register.md](refactor-debt-register.md)
- [refactor-boundaries.md](refactor-boundaries.md)
