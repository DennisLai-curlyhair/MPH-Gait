# MPH-Gait Tokenizer 真實點雲分區可視化

本資料夾使用正式資料中的單一人體點雲 frame，重現 MPH-Gait Tokenizer 在
learned pooling 前的真實 window routing。

## 1. 圖檔

```text
tokenizer_fine16_real_frame.png
tokenizer_fine16_real_frame.svg
tokenizer_coarse4_real_frame.png
tokenizer_coarse4_real_frame.svg
tokenizer_partition_metadata.json
```

對應來源：

```text
dataset/PersonRecognitionWalking_29/person_003/P3_C4_V001/clear_data_060.npy
subject    = P003
clothing   = C4
video      = V001
frame      = 060
valid raw points = 5686
```

這是資料集正式的 `clear_data` 人體前景點雲，不是人工產生的示意點。

## 2. 與正式模型一致的處理流程

```text
1. 讀取 Azure Kinect camera [X, Y, Z]，單位 mm。
2. 套用正式 coordinate adapter：
     [forward, lateral, height_up] = [Z, X, -Y] * 0.001
3. 使用 evaluation loader 的 linspace/round 規則固定抽取 1024 點。
4. 每個 frame 減去 1024 點的 XYZ 平均值，只做 center、不做 scale。
5. 直接呼叫正式 LocalWindowTokenizer._routing_coordinates()。
6. 直接呼叫正式 LocalWindowTokenizer._flat_window_indices()。
7. 分別套用 fine [2,2,4] 與 coarse [1,2,2] regular partitions。
```

正式函式來源：

```text
mph-gait/code/mph_gait/tokenizer.py
shared/gait_core/pointcloud_dataset.py
```

可視化腳本：

```text
mph-gait/tools/visualize_tokenizer_partitions.py
```

## 3. Window ID

圖中 window label 格式為：

```text
F{forward_bin}-L{lateral_bin}-H{height_bin}
```

Fine partition：

```text
2 forward bins x 2 lateral bins x 4 height bins = 16 windows
```

Coarse partition：

```text
1 forward bin x 2 lateral bins x 2 height bins = 4 windows
```

圖中的 3D view 使用：

```text
x-axis = lateral
y-axis = forward
z-axis = height_up
```

## 4. 實際點數結果

Fine 16 window counts：

```text
F0-L0-H0 = 65
F0-L0-H1 = 106
F0-L0-H2 = 137
F0-L0-H3 = 86
F0-L1-H0 = 0
F0-L1-H1 = 28
F0-L1-H2 = 181
F0-L1-H3 = 132
F1-L0-H0 = 37
F1-L0-H1 = 7
F1-L0-H2 = 61
F1-L0-H3 = 0
F1-L1-H0 = 65
F1-L1-H1 = 90
F1-L1-H2 = 29
F1-L1-H3 = 0
sum = 1024
valid windows with at least 2 points = 13 / 16
```

Coarse 4 window counts：

```text
F0-L0-H0 = 215
F0-L0-H1 = 284
F0-L1-H0 = 183
F0-L1-H1 = 342
sum = 1024
valid windows with at least 2 points = 4 / 4
```

## 5. 圖片顯示與未顯示的內容

圖片已顯示：

```text
真實人體點雲
正式座標轉換
正式 deterministic point sampling
per-frame center normalization
正式 min-max routing
每個點實際分配到的 fine / coarse window
每個 window 的實際 point count
```

圖片未顯示：

```text
PointNet 512-D point features
512->64 point projection
learned content score
learned position-strength distance penalty
window 內的 softmax weighted feature pooling
geometry metadata projection
Transformer token mixing
```

因此這兩張圖應稱為：

```text
Fine/coarse regular-window routing visualization
```

不應稱為 attention map 或 learned token activation visualization。Tokenizer
在 routing 後，才會將同一 window 的 64-D point features 依 learned weights
聚合成一個 token。

## 6. 重新產生

在 repository root 執行：

```bash
conda run -n pytorch python \
  mph-gait/tools/visualize_tokenizer_partitions.py
```

指定其他 frame：

```bash
conda run -n pytorch python \
  mph-gait/tools/visualize_tokenizer_partitions.py \
  --input dataset/PersonRecognitionWalking_29/person_003/P3_C4_V001/clear_data_061.npy
```

腳本會同時輸出 PNG、SVG 與完整 JSON metadata，方便論文排版與結果稽核。
