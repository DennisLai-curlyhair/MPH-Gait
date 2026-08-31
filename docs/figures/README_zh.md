# MPH-Gait 論文模型架構圖

本資料夾保存論文使用的可編輯向量圖。

## 檔案

```text
pointnet_tmax_mph_gait_architecture.svg
pointnet_tmax_mph_gait_architecture.png
pointnet_tmax_mph_gait_paper_block_diagram.svg
pointnet_tmax_mph_gait_paper_block_diagram.png
model_architecture_block_explanation_zh.md
tokenizer_real_frame/
```

圖中上半部為正式控制基線 `PointNet-TMax`，下半部為 proposed `MPH-Gait`。
架構內容已依下列正式程式核對：

```text
PointNet-TMax:
  pointnet-tmax/code/pc_v1/model.py

MPH-Gait:
  mph-gait/code/mph_gait/model.py
  mph-gait/code/mph_gait/tokenizer.py

Main config:
  mph-gait/code/mph_gait/configs/mph_gait_len15.yaml
```

## 圖中正式設定

```text
Input: [B, T=15, N=1024, 3]
Point stem: 3 -> 64 -> 128 -> 256 -> 512
Global aggregation: point max -> temporal max
Fine regular grid: 2 x 2 x 4 = 16 windows
Coarse regular grid: 1 x 2 x 2 = 4 windows
Hierarchy tokens: 20 x 64-D per frame
Token mixer: 1 Transformer layer, 4 heads, MLP ratio 2
Local aggregation: masked mean+max -> temporal max
Fusion: z = z_g + 0.20*tanh(alpha)*delta_h
Descriptor: 256-D BNNeck + L2 normalization
```

特別注意：

```text
1. 正式 MPH-Gait 沒有 shifted windows。
2. Transformer 只處理同一 frame 的 20 個 hierarchy tokens。
3. 不同 frame 之間仍使用 temporal max，沒有跨幀 Transformer。
4. Min-max coordinates 只用於 window routing；feature path 保留 metric scale。
```

## 建議 Caption

```text
Overview of PointNet-TMax and MPH-Gait. PointNet-TMax aggregates shared per-point
features using point-wise and temporal max pooling. MPH-Gait preserves this global
path and adds a lightweight fine/coarse point hierarchy. Twenty regular hierarchy
tokens are mixed within each frame and temporally max-pooled to form a bounded,
norm-matched correction to the global embedding. The Transformer does not operate
across frames.
```

## 論文使用

SVG 可直接用 Inkscape、Illustrator 或瀏覽器開啟。IEEE LaTeX 通常建議轉成 PDF：

```bash
inkscape pointnet_tmax_mph_gait_architecture.svg \
  --export-type=pdf \
  --export-filename=pointnet_tmax_mph_gait_architecture.pdf
```

若只能使用 PNG，建議輸出至少 2400 px 寬，避免雙欄放大後文字模糊。
