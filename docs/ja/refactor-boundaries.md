# リファクタリング境界定義

## ゾーン一覧

| ゾーン | パス | 責務 | upstream マージ |
|--------|------|------|-----------------|
| **Upstream コア** | `src/rfdetr/` | 推論・学習・エクスポート | Roboflow 同期対象 |
| **フォーク デモ** | `src/rfdetr_demo/` | 動画デモ・GUI・Vast・チューニング | フォーク独自 |
| **レガシー scripts** | `scripts/` | thin wrapper（非推奨） | 削除予定 |
| **機密メディア** | `confidential/` | 入力/出力動画・監査ログ | **git 非追跡** |
| **非機密サンプル** | `sample/` | 公開可能デモ動画 | gitignore |
| **一般成果物** | `artifacts/` | 非機密デモ出力 | gitignore |

## 変更ルール

1. `src/rfdetr/` の変更は shim 削除・公開 API 追加に限定（Phase 2）
2. 新機能は `src/rfdetr_demo/` にのみ追加
3. 機密動画は `confidential/media/input/` のみ Vast 転送可（`media_guard`）
4. `scripts/` への新規ロジック追加禁止 — `rfdetr_demo` へ
5. 外部向け安定 API は `rfdetr_demo.public`（およびパッケージ root の同名 re-export）のみ
6. GUI は `rfdetr_demo.vast.safety` facade 経由。`safety_guardrails` / `safety_lease` / `safety_settings` 直 import 禁止（`scripts/check_import_cycles.py` が検査）

## Definition of Done（各 PR）

- [ ] 既存 pytest green
- [ ] 新規/変更モジュールに型ヒント
- [ ] 機密パスを扱う場合 `media_guard` 経由
- [ ] `docs/ja/` 索引更新（ユーザー向け変更時）
- [ ] 公開シンボル追加時は `PUBLIC_API` と `tests/rfdetr_demo/test_public_api.py` を更新