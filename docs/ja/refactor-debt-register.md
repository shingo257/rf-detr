# リファクタ負債登録簿

[refactor-master-plan.md](refactor-master-plan.md)（Phase 7–13 体系）の実行トラッキング用。**2026-06-28 実測で全面更新**（旧 Phase 13–20 体系の記録は破棄）。

**最終更新**: 2026-06-28

---

## 1. Deprecation facade（Phase 9 対象）

"deprecated facade" を名乗りながら実コードが依存している不完全移行。**利用を正規パスへ移し facade を削除する。**

| facade | 行数 | 実利用 | 移行先 | 状態 |
|--------|------|--------|--------|------|
| `tracking/detection_stabilizer.py` | 97 | runner, callbacks, tracking/`__init__`, overlays/keypoint, tracking_audit, probe_count（**6**） | `tracking/pipeline.py` (`PersonTrackPipeline`) | 未着手 |
| `vast/runner.py` | 55 | re-export | `vast/cli.py`, `offers.py`, `instance.py` | 未着手 |
| `vast/compat.py` | 58 | private alias | `vast/cli.py`, `instance.py`, `remote_io.py` | 未着手 |
| `inference/pipeline.py` | 11 | re-export | `cli/run_video.py`, `inference/runner.py` | 未着手 |
| `inference/uncertainty_viz.py` | 57 | re-export | `inference/uncertainty/` | 未着手 |
| `gui/controller.py` | 10 | re-export | `gui/state/job_state.py` | 未着手 |

---

## 2. `scripts/` shim（Phase 8 対象）

3行 `from ... import *` shim と 18 行 wrapper。entry point に一本化し削除。

| 種別 | ファイル | 移行先 |
|------|---------|--------|
| 3行 shim（~12） | `vast_*.py`, `media_guard.py`, `keypoint_temporal_filter.py`, `auto_tune_parameters.py`, `tune_live_preview.py`, `keypoint_uncertainty_viz.py`, `video_*.py` | 対応する `rfdetr_demo/*` |
| 18行 wrapper | `run_video_demo.py`, `video_demo_gui.py`, `vast_cleanup_orphans.py` | `[project.scripts]` entry point |
| 残すロジック | `run_mzoo_benchmark.py`（518） | Phase 10 で `rfdetr_demo/benchmark/` へ |
| 維持 | `launch_gui.py`(53), `check_import_cycles.py`(105), `*.cmd` | thin launcher のみ |

---

## 3. 中型ファイル（Phase 12 対象 / 一部 Phase 11）

| パス | 行数 | 分割方針 | Phase |
|------|------|---------|-------|
| `media/tracking_audit.py` | 475 | `audit/common` 抽出 | 11 |
| `vast/safety.py` | 417 | `consent`/`transfer_log`/`guardrails` | 12 |
| `inference/runner.py` | 355 | callback 組立を `callbacks` へ | 12 |
| `tracking/track_store.py` | 354 | hold/ghost 状態機械を分離 | 12 |
| `gui/panels/io_task.py` | 348 | `io_task_sections.py` 移譲 | 12 |
| `gui/panels/compute.py` | 346 | view と `VastController` 境界 | 12 |
| `gui/panels/job_runner.py` | 327 | 進捗/ログを `RunController` へ | 12 |
| `gui/controllers/vast_controller.py` | 326 | offer 探索 / progress 分離 | 12 |
| `tuning/analyze_clip.py` | 319 | コアのみ残す | 12 |
| `media/frame_audit.py` | 306 | `common` 抽出で縮む | 11 |
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

## 4. 監査2系統（Phase 11 対象）

| 系統 | モジュール | 重複 |
|------|-----------|------|
| フレーム監査 | `media/frame_audit.py` | JSONL 追記, `_relpath`, 画像 I/O |
| トラッキング監査 | `media/tracking_audit.py` | 同上 + 評価ロジック |

統合先: `media/audit/common.py`（JSONL・画像 I/O・共通スキーマ）。

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
| `max_missed` 設定化（env/CLI/GUI） | 設計済み | 9（pipeline 設定として） |
| 中央 sticky トラック | 監査で ID 切替 31 回確認 | 9 |
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

`scripts/check_import_cycles.py` で検出。Phase 7 時点: **0 件**（要再確認）。

---

## 関連

- [refactor-master-plan.md](refactor-master-plan.md)
- [refactor-baseline-metrics.md](refactor-baseline-metrics.md)
