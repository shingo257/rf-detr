# Vast.ai 組み込みガイド（ER-FlowScan 共通）

他プロジェクト（FlashFind / rf-detr / 新規 Python アプリなど）に **Vast.ai 外部 GPU** を安全に組み込むための実装指針です。  
本リポジトリ（rf-detr）と兄弟プロジェクト FlashFind の実装を統合した「コピー可能なパターン集」として整理しています。

**関連ドキュメント:** [日本語ドキュメント索引](README.md) · [モデル・COCO クラス解説](models-and-coco-classes.md) · [README — Video Demo GUI](../../README.md#video-demo-gui--動画デモ-gui) · [sample/README](../../sample/README.md)

**rf-detr で試す:** `uv run rfdetr-demo-gui`（GUI） / `uv run rfdetr-vast-cleanup`（orphan 回収）

**機密メディア（Phase 1）:** Vast への入力動画転送は `confidential/media/input/` 配下のみ許可。詳細は [confidential-media-policy.md](confidential-media-policy.md)。

---

## 1. 概要

| 項目 | 内容 |
|------|------|
| 公式 | [Vast.ai](https://vast.ai/) — GPU クラウドマーケットプレイス |
| CLI / SDK | `pip install vastai`（[CLI Hello World](https://docs.vast.ai/cli/hello-world)） |
| API キー | [Manage Keys](https://cloud.vast.ai/manage-keys/) で発行 |
| 課金 | **running 中の GPU 時間** + **ディスク保管料**（destroy まで） |

### 2 つの代表的パターン

| パターン | 用途 | 参考実装 |
|---------|------|----------|
| **A. 永続 Pod** | 推論 API を常時（またはセッション中）起動。SSH トンネルで HTTP 呼び出し | FlashFind `backend/app/vast_client.py`, `gpu_pod_lease.py` |
| **B. エフェメラルジョブ** | 1 ジョブごとにインスタンス作成 → 処理 → destroy | rf-detr `scripts/vast_ai_runner.py` |

どちらも **API キー解決・安全停止・進捗表示** の考え方は共通です。

---

## 2. 前提セットアップ（全プロジェクト共通）

### 2.1 初回のみ

```bat
uv pip install vastai
vastai set api-key YOUR_API_KEY
vastai create ssh-key
```

- SSH 公開鍵を [Vast.ai Keys](https://cloud.vast.ai/manage-keys/) に登録
- クレジットは必要最小限。**auto top-up OFF** 推奨（FlashFind `docs/deploy_vastai.md` 参照）

### 2.2 推奨 Docker イメージ（PyTorch / CUDA）

```
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
```

FlashFind では RTX 4090 + Vast PyTorch テンプレート（`/venv/main`）を検証済み。

---

## 3. API キー解決（ER-FlowScan 標準）

**1 つの API キーを複数プロジェクトで共有**できるよう、次の優先順位を推奨します。

| 優先度 | ソース | 例 |
|--------|--------|-----|
| 1 | アプリ明示入力（GUI / CLI `--api-key`） | ユーザー入力 |
| 2 | プロジェクトローカル設定（gitignore） | `artifacts/vast/vast-config.local.json` |
| 3 | プロジェクト専用 env | `MYAPP_VAST_API_KEY` |
| 4 | 共通 env | `VAST_API_KEY` |
| 5 | FlashFind 共有 env | `FLASHFIND_VAST_API_KEY` |
| 6 | FlashFind `.env` | `FlashFind/backend/.env` |
| 7 | vastai CLI 設定 | `~/.config/vastai/vast_api_key` |

rf-detr 実装: `scripts/vast_api_config.py`

```python
from vast_api_config import resolve_vast_api_key_info

info = resolve_vast_api_key_info(explicit=None)
print(info.source, info.masked)  # 読み込み元を UI に表示
```

### 新プロジェクト用 `.env` テンプレート

```env
# プロジェクト専用（推奨プレフィックス: MYAPP_）
MYAPP_VAST_API_KEY=

# または FlashFind と共有
# FLASHFIND_VAST_API_KEY=   ← FlashFind/backend/.env に既にある場合は省略可
```

### ローカル上書き（gitignore）

`artifacts/vast/vast-config.local.json`:

```json
{
  "api_key": "your-key-here"
}
```

---

## 4. モジュール構成（rf-detr — コピー元）

新規 Python プロジェクトに持ち込む場合、以下を **セットで** コピーし、プレフィックスとラベルを変更します。

| ファイル | 役割 |
|----------|------|
| `scripts/vast_api_config.py` | API キー解決・`.env` パース・ローカル保存 |
| `scripts/vast_safety.py` | リース永続化、orphan 回収、destroy リトライ、`atexit`/シグナル |
| `scripts/vast_start_phases.py` | 起動シーケンスフェーズ定義（FlashFind 互換） |
| `scripts/vast_start_progress.py` | Tkinter ステップ進捗 UI |
| `scripts/vast_preflight.py` | 実行前チェックリスト |
| `scripts/vast_ai_runner.py` | エフェメラルジョブ本体（要カスタム） |
| `scripts/vast_cleanup_orphans.py` | 手動 orphan 回収 CLI |

GUI 例: `scripts/video_demo_gui.py`  
起動: `scripts/run_demo_gui.cmd`

---

## 5. パターン A — 永続 Pod（FlashFind 型）

### 5.1 アーキテクチャ

```text
[ローカル PC]  アプリ API + UI
    └─ SSH トンネル localhost:8081 ─→ [Vast Pod] 推論サービス :18080
```

### 5.2 主要コンポーネント（FlashFind）

| パス | 内容 |
|------|------|
| `FlashFind/backend/app/vast_client.py` | REST: start / stop / destroy |
| `FlashFind/backend/app/gpu_pod_lease.py` | アイドル・最大時間・grace 自動停止 |
| `FlashFind/backend/app/routers/gpu_pod.py` | `/gpu/pod/*` API |
| `FlashFind/frontend/src/utils/gpuPodStartPhases.ts` | 起動 4 ステップ定義 |
| `FlashFind/frontend/src/components/GpuPodStartProgress.tsx` | 進捗 UI |
| `FlashFind/docs/deploy_vastai.md` | デプロイ手順 |

### 5.3 環境変数（FlashFind）

```env
FLASHFIND_VAST_API_KEY=
FLASHFIND_VAST_INSTANCE_ID=41472413
FLASHFIND_VAST_AUTO_STOP_IDLE_MINUTES=15
FLASHFIND_VAST_AUTO_STOP_RELEASE_GRACE_SEC=120
FLASHFIND_VAST_AUTO_STOP_MAX_HOURS=8
FLASHFIND_VAST_AUTO_STOP_ON_SHUTDOWN=true
FLASHFIND_VAST_AUTO_STOP_ONLY_IF_STARTED_BY_APP=true
```

### 5.4 Pod 起動シーケンス（UI 4 ステップ）

FlashFind `POD_START_STEPS` と同一:

1. Vast API に起動リクエスト
2. GPU インスタンスのブート
3. SSH ポートの割当
4. Pod 稼働開始

Web UI では `actual_status`（`loading` → `running`）を 3 秒間隔でポーリングし、ステップ表示を更新します。

### 5.5 安全装置

| 機構 | 動作 |
|------|------|
| アイドル自動停止 | 15 分無操作で `stop` |
| 最大セッション | 8 時間で `stop` |
| タブ閉鎖 grace | 120 秒後に `stop` |
| バックエンド終了時 | `stop` |
| 起動時 orphan | 期限切れ lease があれば即 `stop` |
| 手動起動 Pod | `ONLY_IF_STARTED_BY_APP=true` なら触らない |

---

## 6. パターン B — エフェメラルジョブ（rf-detr 型）

### 6.1 ライフサイクル

```text
search offers → create instance (--label) → wait running
→ vastai copy (upload) → vastai execute (remote job)
→ vastai copy (download) → destroy (finally)
```

### 6.2 起動シーケンス（UI 7 ステップ）

rf-detr `VAST_JOB_STEPS`（FlashFind 4 ステップ + ジョブ 3 ステップ）:

| # | ステップ | フェーズ |
|---|----------|----------|
| 1 | Vast API に起動リクエスト | `requesting` |
| 2 | GPU インスタンスのブート | `booting` |
| 3 | SSH ポートの割当 | `ssh_ready` |
| 4 | 入力データのアップロード | `uploading` |
| 5 | リモート GPU で処理 | `running` |
| 6 | 結果のダウンロード | `downloading` |
| 7 | インスタンスの破棄 | `cleanup` |

進捗イベント型: `VastProgressUpdate`（`scripts/vast_start_phases.py`）

### 6.3 環境変数（rf-detr）

| 変数 | デフォルト | 意味 |
|------|-----------|------|
| `RFDETR_VAST_API_KEY` | — | プロジェクト専用 API キー |
| `RFDETR_VAST_MAX_SESSION_HOURS` | `2` | インスタンス最大稼働時間 |
| `RFDETR_VAST_MAX_EXECUTE_HOURS` | `2` | リモート execute 最大時間 |
| `RFDETR_VAST_BOOT_TIMEOUT_SEC` | `900` | 起動待ちタイムアウト |
| `RFDETR_VAST_DESTROY_RETRIES` | `3` | destroy リトライ回数 |
| `RFDETR_VAST_INSTANCE_LABEL_PREFIX` | `rfdetr-demo` | orphan 検索用ラベル |
| `RFDETR_VAST_AUTO_CLEANUP_ORPHANS` | `true` | 起動時 orphan 破棄 |

長時間動画（713 フレーム等）は `MAX_EXECUTE_HOURS=3` 以上を検討。

### 6.4 安全装置

| 機構 | 実装 |
|------|------|
| `finally` で destroy | `VastJobGuard.destroy_if_needed()` |
| キャンセル | `threading.Event` → プロセス kill → destroy |
| 最大実行時間 | 超過で remote kill + destroy |
| destroy リトライ | 3 回、失敗時 CRITICAL ログ + 手動コマンド案内 |
| 緊急クリーンアップ | `atexit` + SIGINT/SIGTERM/SIGBREAK |
| リース永続化 | `artifacts/vast/vast-job-lease.local.json` |
| orphan 回収 | ラベル prefix + lease から `cleanup_orphan_instances()` |
| インスタンスラベル | `rfdetr-demo-{timestamp}` |

手動 orphan 回収:

```bat
scripts\vast_cleanup_orphans.cmd
```

---

## 7. 新プロジェクトへの組み込み手順

### Step 1 — 依存関係

```toml
# pyproject.toml [project.optional-dependencies]
vast = ["vastai>=0.3.0"]
```

```bat
uv pip install vastai
```

### Step 2 — モジュールをコピー

1. `vast_api_config.py` — `_FLASHFIND_ENV` パスを自プロジェクトの monorepo 構成に合わせて調整
2. `vast_safety.py` — env プレフィックスを `MYAPP_VAST_*` にリネーム
3. `vast_start_phases.py` — そのまま利用可能
4. `vast_preflight.py` — チェック項目を追加
5. `vast_ai_runner.py` — **リモート execute コマンドだけ** プロジェクト用に書き換え

### Step 3 — env プレフィックスの統一

```python
# vast_safety.py の from_env() を例:
max_session_sec=_env_float("MYAPP_VAST_MAX_SESSION_HOURS", 2.0) * 3600.0
instance_label_prefix=os.environ.get("MYAPP_VAST_INSTANCE_LABEL_PREFIX", "myapp-job")
```

`vast_api_config.py` に `MYAPP_VAST_API_KEY` を解決順に追加。

### Step 4 — リモートジョブの実装

エフェメラル型の最小テンプレート:

```bash
set -euo pipefail
python -m pip install -q --upgrade pip
python -m pip install -q YOUR_PACKAGE supervision opencv-python-headless
cd /workspace/job
python your_script.py --input /workspace/job/input.dat --output /workspace/job/output.dat
```

転送:

```bat
vastai copy local:.\input.dat INSTANCE_ID:/workspace/job/input.dat
vastai execute INSTANCE_ID "bash -lc '...'"
vastai copy INSTANCE_ID:/workspace/job/output.dat local:.\output.dat
vastai destroy instance INSTANCE_ID
```

**rf-detr（Phase 12）:** `src/rfdetr_demo/` パッケージと `vast/remote_runner.py` を `/workspace/rfdetr_job/package/` に SCP し、`PYTHONPATH` 経由で `remote_runner.py`（内部で `rfdetr_demo.cli`）を実行します。`scripts/run_video_demo.py` への依存はありません。実装: `src/rfdetr_demo/vast/video_job.py`。

### Step 5 — 進捗 UI

**Web（React）**: FlashFind の `gpuPodStartPhases.ts` + `GpuPodStartProgress.tsx` をそのまま移植。

**Desktop（Tkinter）**: rf-detr の `VastStartProgressPanel` を利用。

**CLI のみ**: `VastProgressUpdate` をログ出力:

```python
def on_progress(update: VastProgressUpdate) -> None:
    print(f"[{update.phase.value}] {update.message} ({update.percent:.0f}%)")
```

### Step 6 — Preflight

実行前に必ず:

- [ ] `vastai` CLI が PATH にある
- [ ] API キーが解決できる
- [ ] GPU オファー / Instance ID が選ばれている
- [ ] 安全上限（max hours）がジョブ長に見合う

rf-detr: `run_vast_preflight()` / GUI「事前チェック」

### Step 7 — gitignore

```
artifacts/vast/
*.local.json
.env
```

---

## 8. GPU オファー検索（CLI）

```bat
vastai search offers "rentable=true verified=true num_gpus=1 reliability>0.95 dph_total<0.80" --order dph --limit 10 --raw
```

rf-detr GUI の「GPU 検索」は同一条件を `scripts/vast_ai_runner.py` の `search_gpu_offers()` で実行。

インスタンス作成:

```bat
vastai create instance OFFER_ID ^
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime ^
  --disk 50 --label myapp-job-1234567890 ^
  --ssh --direct --raw
```

---

## 9. トラブルシューティング

| 症状 | 対処 |
|------|------|
| API キー未設定 | FlashFind `.env` または `vastai set api-key` |
| 起動が 5 分超 | `BOOT_TIMEOUT_SEC` 確認、別オファーを選択 |
| 課金が止まらない | `vastai show instances --raw` → `destroy instance ID` |
| 前回クラッシュで orphan | `scripts/vast_cleanup_orphans.cmd` |
| destroy 失敗 | ログの CRITICAL を確認、手動 destroy |
| scp 失敗 | 大文字 `-P`、Direct SSH、`touch ~/.no_auto_tmux`（[Vast SSH ドキュメント](https://docs.vast.ai/guides/instances/connect/ssh)） |
| リモート pip が遅い | Docker イメージに依存を事前 bake |

---

## 10. 参考リンク

| リソース | URL |
|----------|-----|
| Vast.ai トップ | https://vast.ai/ |
| CLI Hello World | https://docs.vast.ai/cli/hello-world |
| Python SDK | https://vast.ai/developers/sdk |
| API Keys | https://cloud.vast.ai/manage-keys/ |
| SSH 接続 | https://docs.vast.ai/guides/instances/connect/ssh |
| データ転送 | https://docs.vast.ai/guides/instances/storage/data-movement |

---

## 11. ER-FlowScan 内の実装マップ

| プロジェクト | パターン | 主要パス |
|-------------|---------|----------|
| **FlashFind** | 永続 Pod + SSH トンネル | `FlashFind/backend/app/vast_client.py`, `docs/deploy_vastai.md` |
| **rf-detr** | エフェメラル動画ジョブ + GUI | `rf-detr/scripts/vast_*.py`, `scripts/video_demo_gui.py` |
| **One-Shot-Learning** | （Vast 未使用） | ローカル YOLO パイプラインのみ |

新規プロジェクトでは **用途に応じて A か B を選び**、本ドキュメント §3（API キー）と §5/§6（安全装置）を必ず取り込んでください。

---

## 12. チェックリスト（リリース前）

```
[ ] API キーを git に含めていない
[ ] artifacts/vast/ を gitignore 済み
[ ] destroy / stop が finally / shutdown で必ず呼ばれる
[ ] 最大稼働時間・最大 execute 時間を設定済み
[ ] インスタンスに label prefix を付与済み
[ ] 起動時 orphan クリーンアップを有効化
[ ] Preflight で fail 時はジョブ開始をブロック
[ ] 進捗 UI で boot 中ステータスが見える
[ ] 長時間ジョブの MAX_EXECUTE_HOURS を見積もりより長めに設定
```
