# リファクタ負債登録簿

[refactor-master-plan.md](refactor-master-plan.md)（Phase 7–13 体系）の実行トラッキング用。**2026-06-28 実測で全面更新**（旧 Phase 13–20 体系の記録は破棄）。

**最終更新**: 2026-08-04

---

## 1. Deprecation facade（Phase 9 ✅ 完了）

旧 "deprecated facade" 6 本を削除し、正規パスのみを公開 API とした。

| 旧 facade | 移行先 | 状態 |
|-----------|--------|------|
| `tracking/detection_stabilizer.py` | `tracking/pipeline.py` / `stabilizer.py` / `bbox.py` | ✅ 削除 |
| `vast/runner.py` | `vast/cli.py`, `offers.py`, `video_job.py`, `types.py` | ✅ 削除 |
| `vast/compat.py` | `vast/cli.py`, `instance.py`, `remote_io.py` | ✅ 削除 |
| `inference/pipeline.py` | `cli/run_video.py`, `inference/runner.py` | ✅ 削除 |
| `inference/uncertainty_viz.py` | `inference/uncertainty/` | ✅ 削除 |
| `gui/controller.py` | `gui/state/job_state.py` | ✅ 削除 |

`tracking/__init__.py` は `PersonTrackPipeline` 系 API のみを export する。scripts shim（`vast_ai_runner.py`, `keypoint_uncertainty_viz.py`）は正規モジュールへ直接 import する。

### Phase 13–15 構造的重複（解消済）

| 項目 | 旧状態 | 現状態 |
|------|--------|--------|
| `PersonAssociator` 多重保有 | stabilizer / temporal / associator | `TrackStore` 内の単一 associator |
| `DetectionStabilizer._TrackSnapshot` | stabilizer 内 private | `track_store.TrackSnapshot` |
| `KeypointTemporalFilter._associator` | temporal が独自 ID | 削除。pipeline の `track_id` を消費 |

---

## 2. `scripts/` shim（Phase 8 対象）

3行 `from ... import *` shim と 18 行 wrapper。entry point に一本化し削除。

| 種別 | ファイル | 移行先 | 状態 |
|------|---------|--------|------|
| 診断 wrapper | `probe_person_count.py`, `audit_center_tracking.py` | `rfdetr-demo probe-count` / `audit-tracking` | Phase 14 ✅ |
| 18行 wrapper | `run_video_demo.py`, `video_demo_gui.py`, `vast_cleanup_orphans.py` | `[project.scripts]` entry point | DeprecationWarning thin |
| 3行 shim（~12） | `vast_*.py`, `media_guard.py`, … | 対応する `rfdetr_demo/*` | Phase 8 残 |
| 残すロジック | `run_mzoo_benchmark.py`（518） | `rfdetr_demo/benchmark/` | Phase 10 残 |
| 維持 | `launch_gui.py`, `check_import_cycles.py`, `*.cmd` | thin launcher | ✅ |

---

## 3. 中型ファイル（Phase 12 対象 / 一部 Phase 11）

| パス | 行数 | 分割方針 | Phase |
|------|------|---------|-------|
| `media/tracking_audit.py` | 26 | `media/audit/` 抽出済（facade） | 11 ✅ |
| `vast/safety.py` | 29 | `safety_settings` / `safety_lease` / `safety_guardrails` 抽出済（facade） | 12 ✅ |
| `inference/runner.py` | 355 | callback 組立を `callbacks` へ | 12 |
| `tracking/track_store.py` | 354 | hold/ghost 状態機械を分離 | 12 |
| `gui/panels/io_task.py` | 348 | `io_task_sections.py` 移譲 | 12 |
| `gui/panels/compute.py` | 346 | view と `VastController` 境界 | 12 |
| `gui/panels/job_runner.py` | 327 | 進捗/ログを `RunController` へ | 12 |
| `gui/controllers/vast_controller.py` | 326 | offer 探索 / progress 分離 | 12 |
| `tuning/analyze_clip.py` | 319 | コアのみ残す | 12 |
| `media/frame_audit.py` | 26 | `media/audit/frame.py` 抽出済（facade） | 11 ✅ |
| `inference/temporal_filter.py` | 301 | 設定 dataclass を `types` へ | 12 |

### 2026-07-29 animation 追加実測

| パス | 行数 | 状態 / 次の分割 |
|------|------|-----------------|
| `animation/puppet_render.py` | 23 | 旧 757 行から分割済み。リポジトリ内実利用ゼロ、外部互換専用 facade |
| `animation/puppet_assets.py` | 191 | asset/manifest、RGBA layer、grid、pivot 分離完了。strict mypy clean |
| `animation/puppet_mesh.py` | 149 | mesh weight、rotation、rigid prop 分離完了。strict mypy clean |
| `animation/puppet_layered.py` | 156 | layered sprite transform/composition 分離完了。strict mypy clean |
| `animation/puppet_continuous.py` | 318 | continuous mesh/expression composition 分離完了。strict mypy clean |
| `animation/puppet_renderer.py` | 55 | render mode に応じて compositor を選ぶ互換 router。strict mypy clean |
| `animation/puppet_timeline.py` | 63 | 分離完了。strict mypy clean |
| `animation/puppet_video.py` | 96 | 分離完了。strict mypy clean |
| `animation/puppet_cli.py` | 52 | 分離完了。strict mypy clean |

残存負債:

- renderer 内部の strict mypy 既存エラーは asset/mesh/composition 分離で 15→0 件。新境界 9 ファイルを strict gate の対象として維持する。
- `puppet_render.py` facade のローカル consumer はゼロ。外部互換のため、明示的な deprecation 告知と versioned release 境界が決まるまでは削除しない。
- `puppet_continuous.py` は 318 行だが continuous mesh/expression compositor として凝集している。行数だけを理由に再分割せず、表情 policy を独立変更する要求が出た時点で抽出する。

---

## 4. 監査2系統（Phase 11 — 完了）

| 系統 | モジュール | 状態 |
|------|-----------|------|
| 共通 | `media/audit/common.py` | JSONL / `_relpath` / 画像 I/O / 共通スキーマ |
| フレーム | `media/audit/frame.py` + `frame_audit.py` facade | ✅ |
| トラッキング | `media/audit/tracking_*.py` + `tracking_audit.py` facade | ✅ |

共通 JSONL ベース: `timestamp`, `classification`, `audit_kind`, `run_id`, `source_relpath`, `frame_index`（+ kind 固有フィールド）。

---

## 5. リポジトリ衛生（Phase 7 対象）

| 項目 | 状態 |
|------|------|
| CRLF チャーン | `.gitattributes` 変更で 91 ファイル modified 表示・実差分 8 ファイル/293行。`git add --renormalize .` で隔離 |
| 未追跡 | `confidential/`/`artifacts/`/`sample/`/`data/` は `.gitignore` 済み。`tests/rfdetr_demo/test_expected_person_count.py` は要否判断 |

---

## 6. 機能負債（リファクタと並行）

| 項目 | 状態 | 担当 Phase |
|------|------|-----------|
| `max_missed` 設定化（env/CLI） | ✅ `RFDETR_MAX_MISSED` + CLI | Phase 15c |
| 中央 sticky トラック | ✅ `RFDETR_STICKY_CENTER_TRACK` + CLI | Phase 15c（全編回帰は手動） |
| auto_tune KPI 試走 | 未達 | 実験トラック |

---

## 7. 回帰アンカー

手動実行（CI 外）。結果は PR 説明に記載。

```bash
.venv\Scripts\python.exe -m pytest tests/rfdetr_demo
uv run rfdetr-demo probe-count --mode stabilize --frames 20
uv run rfdetr-demo audit-tracking --max-frames 60
.venv\Scripts\python.exe scripts\check_import_cycles.py
```

### Golden 監査（713 フレーム、機密動画必要）

- run_id: `20260620T101049Z-ed8dabe5`
- 中央欠落: 68/713
- ID 切替: 31
- Phase 9 目標: ID 切替 ≤ 15（sticky 有効時）

---

## 8. import 循環

`scripts/check_import_cycles.py` で検出。Phase 13 以降: **0 件**（2026-08-02 確認）。

---

## 関連

- [refactor-master-plan.md](refactor-master-plan.md)
- [refactor-baseline-metrics.md](refactor-baseline-metrics.md)
