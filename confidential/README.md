# 機密メディア区画

このディレクトリは **git に追跡されません**。機密動画・解析結果・Vast 転送監査ログを置きます。

## 配置

| パス | 用途 |
|------|------|
| `media/input/` | ソース動画（例: `mn1-2.mov`） |
| `media/output/` | 解析結果 MP4 |
| `audit/` | Vast 転送監査 (`vast-transfers.jsonl`) |

## 初回セットアップ

```powershell
# 既存 sample から移行（任意）
Copy-Item sample\mn1-2.mov confidential\media\input\mn1-2.mov
```

詳細: [docs/ja/confidential-media-policy.md](../docs/ja/confidential-media-policy.md)
