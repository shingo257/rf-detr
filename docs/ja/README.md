# ER-FlowScan 向け日本語ドキュメント

RF-DETR フォーク（`shingo257/rf-detr`）および ER-FlowScan monorepo 向けのローカル技術資料です。  
upstream の [rfdetr.roboflow.com](https://rfdetr.roboflow.com) とは別に、フォーク固有の運用・連携をまとめています。

---

## ドキュメント一覧

| ドキュメント | 内容 |
|-------------|------|
| [models-and-coco-classes.md](models-and-coco-classes.md) | モデル構成、COCO 80 クラス、タスク別の使い分け |
| [vast-ai-integration-guide.md](vast-ai-integration-guide.md) | Vast.ai 外部 GPU の組み込み（FlashFind / rf-detr 共通） |
| [confidential-media-policy.md](confidential-media-policy.md) | 機密動画区画・Vast 転送ガバナンス |
| [refactor-boundaries.md](refactor-boundaries.md) | upstream / fork / confidential の責務境界 |
| [refactor-baseline-metrics.md](refactor-baseline-metrics.md) | リファクタ baseline メトリクス |
| [refactor-master-plan.md](refactor-master-plan.md) | **Phase 7–13 リファクタ計画**（公開 API 凍結まで） |
| [refactor-debt-register.md](refactor-debt-register.md) | 負債登録簿・実行トラッキング |
| [frame-independent-detection.md](frame-independent-detection.md) | フレーム独立推論と検出人数点滅の原因・対策 |
| [スイート横断対策](../../../docs/rfdetr-person-count-flicker-cross-app.md) | FlashFind / One-Shot 知見を踏まえた改善ロードマップ（モノレポ `docs/`） |

---

## 公開 API（`rfdetr_demo` 0.2.0）

安定シンボルは `rfdetr_demo` / `rfdetr_demo.public` から import してください。

| シンボル | 正規モジュール |
|----------|----------------|
| `run_demo` | `inference.runner` |
| `PersonTrackPipeline` | `tracking.pipeline` |
| `ConfidentialFrameAuditLogger` | `media.audit.frame` |
| `run_center_tracking_audit` | `media.audit.tracking` |
| `run_vast_cli` | `vast.cli` |
| `DEFAULT_PARAMETERS` | `tuning.auto_tune` |

詳細はソースの `src/rfdetr_demo/public.py`（`PUBLIC_API` 許可リスト）を参照。

---

## クイックリンク

| 操作 | コマンド / パス |
|------|----------------|
| 動画デモ（CLI） | `uv run rfdetr-demo` または `uv run rfdetr-demo video` |
| 人数プローブ | `uv run rfdetr-demo probe-count --mode stabilize --frames 20` |
| 中央トラック監査 | `uv run rfdetr-demo audit-tracking --max-frames 60` |
| クリップ品質解析 | `uv run rfdetr-demo analyze-clip --seconds 1` |
| MZoo ベンチマーク | `uv run rfdetr-mzoo-benchmark` |
| 動画デモ（GUI） | `uv run rfdetr-demo-gui` または `scripts\run_demo_gui.cmd` |
| import / 境界チェック | `uv run python scripts/check_import_cycles.py` |
| 機密入力動画 | `confidential/media/input/` |
| Vast orphan 回収 | `uv run rfdetr-vast-cleanup` または `scripts\vast_cleanup_orphans.cmd` |
| 機密メディア規程 | [confidential/README.md](../../confidential/README.md) |
| FlashFind Vast デプロイ | `../FlashFind/docs/deploy_vastai.md` |

旧 `scripts/*.py` 診断スクリプトは削除済み。互換が必要なものは DeprecationWarning 付き thin launcher のみ（`run_mzoo_benchmark.py`, `launch_gui.py`）。

---

## 関連 README セクション

- [Local Video Demo](../../README.md#local-video-demo--ローカル動画デモ)
- [Video Demo GUI](../../README.md#video-demo-gui--動画デモ-gui)
