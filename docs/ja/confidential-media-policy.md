# 機密メディア取扱ポリシー

## 目的

gitignore 以上のガバナンスで、機密動画の誤コミット・IDE/AI への自動アップロード・意図しない Vast.ai 転送を防止します。

## 配置ルール

| 種別 | パス | git | Vast 転送 |
|------|------|-----|-----------|
| 機密入力 | `confidential/media/input/` | 非追跡 | **可**（同意後） |
| 機密出力 | `confidential/media/output/` | 非追跡 | 不可 |
| 監査ログ | `confidential/audit/` | 非追跡 | 不可 |
| 非機密デモ | `sample/` | 非追跡 | **不可**（既定） |
| 一般成果物 | `artifacts/demo/` | 非追跡 | 不可 |

## 初回セットアップ

```powershell
New-Item -ItemType Directory -Force confidential\media\input, confidential\media\output, confidential\audit
Copy-Item sample\mn1-2.mov confidential\media\input\mn1-2.mov   # 既存動画がある場合
```

## Vast.ai 転送

1. 入力は `confidential/media/input/` 配下のみ（allowlist）
2. GUI: 初回ダイアログ + `confidential/.vast-consent` 保存
3. CLI: `--vast-ack-transfer`（非対話環境）
4. 監査: `confidential/audit/vast-transfers.jsonl`
5. 例外: `RFDETR_VAST_ALLOW_ANY_SOURCE=1`（要二重確認・非推奨）

## IDE / Cursor

`.cursorignore` で `confidential/**`, `sample/**`, 動画拡張子を除外し、AI コンテキストへの混入を抑止します。

## 運用チェックリスト

- [ ] push 前: `git status` に動画・`confidential/` が含まれていない
- [ ] Vast 実行後: `confidential/audit/vast-transfers.jsonl` を確認
- [ ] Vast 実行後: インスタンス destroy 完了を GUI ログで確認
- [ ] 出力 MP4 は機密なら `confidential/media/output/` へ

関連: [refactor-boundaries.md](refactor-boundaries.md), [vast-ai-integration-guide.md](vast-ai-integration-guide.md)
