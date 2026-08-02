# ふっかちゃんニュートラルリグ生成記録

- 生成方式: Codex組み込み画像編集
- 用途分類: `precise-object-edit`
- 編集対象: `ふっかちゃん浴衣.jpg`
- 背景: クロマキー用単色 `#ff00ff`
- 採用透明素材: `artifacts/fukkachan_rig_v2/fukkachan_neutral_v2.png`

## 最終プロンプト

```text
Use case: precise-object-edit
Asset type: 2D layered character rig source sheet for animation
Input image: edit target and strict visual identity reference
Primary request: Repose the exact same Fukkachan character into a clean front-facing neutral rig pose. Keep the head centered and upright. Place both arms slightly away from the torso at about 35 degrees downward so the shoulders, sleeve roots, wrists, and hands are fully visible and do not overlap the body. Place both short legs slightly apart so each leg and foot is independently visible and does not overlap the other leg or the obi.
Scene/backdrop: perfectly flat solid #ff00ff chroma-key background, one uniform color, no shadows, gradients, texture, floor, or lighting variation.
Style/medium: preserve the original flat Japanese mascot illustration exactly: same thick dark-brown outlines, same simple flat fills, same proportions, same facial expression, same antler shapes, same forehead symbol, same ears, same pink side pouch, same blue yukata, same green obi, no added shading.
Composition: entire character visible with generous padding on all sides, centered, square canvas.
Constraints: preserve character identity and every visible design detail aggressively. Reconstruct only the normally hidden connection areas under the head, neck, shoulders, sleeve roots, obi, hips, leg roots, wrists, and ankles so the image can be separated into animation layers. Keep left-right symmetry where appropriate. No text, no labels, no watermark. Do not crop antlers, ears, hands, feet, or pouch.
Avoid: redesign, 3D rendering, gradients, painterly texture, extra limbs, missing limbs, changed face, changed colors, changed costume, pose overlap, white background, shadows. Do not use #ff00ff anywhere in the character.
```

## 透明化

ピンク部分を保護するためデスピルは使用せず、次の設定を採用した。

```powershell
python remove_chroma_key.py `
  --input fukkachan_neutral_chroma.png `
  --out fukkachan_neutral_v2.png `
  --auto-key border `
  --soft-matte `
  --transparent-threshold 8 `
  --opaque-threshold 110 `
  --edge-contract 1
```
