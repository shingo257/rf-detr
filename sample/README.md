# Sample media / サンプルメディア

## ローカル専用（Git に含めない）

**`sample/` 配下の動画ファイル（`.mov` / `.mp4` など）はローカル専用です。**  
GitHub やその他の公開リポジトリには **コミット・プッシュしないでください**。

| 項目 | 内容 |
|------|------|
| 想定ファイル名 | `sample/mzoo.mov`（各自の PC にのみ配置） |
| リポジトリ | `sample/*` は `.gitignore` で除外（この README のみ追跡） |
| 出力 | `artifacts/demo/` も同様にローカルのみ（`.gitignore` 済み） |

**English:** Place demo videos locally only. They are gitignored and must not be uploaded to public remotes.

### 初回セットアップ

1. デモ用動画を **`sample/mzoo.mov`** として保存（ファイル名はスクリプト既定に合わせる）
2. 以下のコマンドでデモ実行
3. `git status` で動画が Untracked / ignored であることを確認してから push

---

## mzoo.mov（ダンス動画・ローカル配置）

| 項目 | 値 |
|------|-----|
| 用途 | RF-DETR **人物検出** / **キーポイント** デモ（COCO `person` 向け） |
| 解像度 | 556×1294（縦動画） |
| フレーム | 約 1270 @ 40 fps |

ジャガイモコンベア動画（COCO 外クラス）より、事前学習モデルのデモに適しています。

### 人物検出デモ

```bat
cd rf-detr
scripts\run_demo_video.cmd --task detect --person-only --frame-stride 2
```

出力: `artifacts/demo/mzoo_detected.mp4`（ローカルのみ）

### 骨格（キーポイント）デモ

```bat
scripts\run_demo_video.cmd --task keypoint --frame-stride 2 --max-frames 120
```

出力: `artifacts/demo/mzoo_keypoints.mp4`（ローカルのみ）  
（キーポイント Preview モデルは CPU では重いため、初回は `--max-frames` 推奨）

### 別の動画を使う場合

```bat
scripts\run_demo_video.cmd --source "C:\path\to\your\video.mp4" --task detect --person-only
```

`--source` で任意のローカルパスを指定できます（こちらも公開リポジトリには含めないでください）。

---

## GUI デモ（ローカル / Vast.ai GPU）

コマンドラインの代わりに GUI から実行できます。Vast.ai を選ぶと、リモート GPU で推論して結果 MP4 をダウンロードします。

```bat
cd rf-detr
scripts\run_demo_gui.cmd
```

| 項目 | 内容 |
|------|------|
| ローカル | CPU/GPU 環境で `run_video_demo.py` を実行 |
| 外部 GPU | Vast.ai インスタンスを起動 → アップロード → 推論 → ダウンロード → destroy |
| 不確実性 | キーポイント＋関節別ヒートマップ出力に対応 |

Vast.ai の API キー設定・安全装置・他プロジェクトへの移植: [docs/ja/vast-ai-integration-guide.md](../docs/ja/vast-ai-integration-guide.md)
