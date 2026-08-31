# PointNet-TMax 與 MPH-Gait 模型架構 Block 說明

本文件對應：

- `pointnet_tmax_mph_gait_paper_block_diagram.svg`
- `pointnet_tmax_mph_gait_paper_block_diagram.png`

圖中 (a) 為控制基線 **PointNet-TMax**，(b) 為提出方法 **MPH-Gait**。
內容依以下正式程式核對：

```text
PointNet-TMax:
  pointnet-tmax/code/pc_v1/model.py

MPH-Gait:
  mph-gait/code/mph_gait/model.py
  mph-gait/code/mph_gait/tokenizer.py

Main config:
  mph-gait/code/mph_gait/configs/mph_gait_len15.yaml
```

## 1. 圖例

| 表示 | 含意 |
|---|---|
| 青色 | 兩個模型共用的 PointNet feature extractor |
| 藍色 | Global identity path |
| 橘色 | MPH-Gait local point-hierarchy path |
| 黃色 | Transformer token mixer |
| 綠色 | Residual fusion 與 retrieval descriptor |
| 紫色虛線 | 只在訓練時使用的 CE classifier |
| 堆疊平面 | Frames、feature maps 或 hierarchy tokens |
| 梯形 block | 特徵投影或編碼模組 |
| 橘色虛線 | 只負責 window assignment 的 coordinate routing |

## 2. 共用輸入與前處理

### Point sequence

```text
Input = [B,T,N,3]
T = clip frame 數，主設定 15
N = 每個 frame 點數，主設定 1024
3 = metric XYZ
```

模型直接處理人體點雲，不必先轉換成影像。

### Center XYZ

```text
p_centered[t,n] = p[t,n] - mean_n(p[t,n])
```

每個 frame 獨立減去點雲中心，用來移除人物在感測器座標中的絕對平移位置。
正式設定不做 scale normalization，因此仍保留身高、肢體比例與 metric scale。

## 3. PointNet-TMax Blocks

| Block | 運算與輸出 | 用途 |
|---|---|---|
| Shared PointNet | `1x1 Conv + BN + ReLU`，`3→64→128→256→512`，輸出 `[N,512]` | 將 XYZ 轉成共用 per-point features |
| Point Max | `[N,512]→[512]` | 取得 point-order invariant frame descriptor |
| Frame sequence | T 個 descriptors 組成 `[T,512]` | 保留每個 frame 的全身特徵 |
| Temporal Max | 在 T 個 frame 上逐 channel 取 max | 聚合 clip 最強證據；快速但不建模順序 |
| Embedding Head | `Linear 512→256 + BN + ReLU` | 建立 256-D identity embedding |
| BNNeck | 對 embedding 做 BatchNorm | 穩定分類與 retrieval feature distribution |
| CE Head | `Linear 256→train identities` | 只在訓練時區分 train IDs |
| Cosine Retrieval | L2 normalize 後比較 Probe／Gallery | 部署時辨識已註冊人物 |

整體公式：

```text
z_g = Head(max_t(max_n(PointNet(p_tn))))
```

PointNet-TMax 的優點是簡單、快速、參數少；限制是兩次 max pooling 會捨棄
局部區域配置與 frame order。

## 4. MPH-Gait Global Identity Path

MPH-Gait 使用相同的輸入、Center XYZ 與 Shared PointNet。Global path 也
保留 PointNet-TMax 的主要流程：

```text
Per-point feature
  -> Point Max
  -> Temporal Max
  -> Linear 512→256 + BN + ReLU
  -> global embedding z_g
```

此路徑保留已驗證有效的全身身形與幾何資訊。Local branch 是受限制的
correction，不是完全取代 global representation。

## 5. Local Point-Hierarchy Blocks

### Routing Coordinates

```text
routing = (XYZ - min(XYZ)) / (max(XYZ) - min(XYZ))
```

Routing coordinates 只用於 window assignment。Identity feature path 仍
使用保留 metric scale 的 PointNet features。

### Fine 與 Coarse Windows

| 層級 | Grid | 數量 | 目的 |
|---|---:|---:|---|
| Fine | `2×2×4` | 16 | 捕捉細部、不同高度與左右區域的局部幾何 |
| Coarse | `1×2×2` | 4 | 補充較大區域與跨局部上下文 |

正式 MPH-Gait 只有 regular windows，沒有 shifted windows。

### Window Tokenizer

PointNet 的 512-D feature 先投影成 64-D：

```text
Conv1d(512→64) + BN + ReLU
```

每個點產生 learned content score，並依其與 window center 的相對距離加入
position penalty：

```text
score = content_score - position_strength * relative_distance^2
token = weighted_sum(projected_point_features)
```

因此模型可在每個 window 中選擇較有辨識力且位置合理的點。

### Geometry Metadata

每個 token 加上 8-D metadata：

```text
window center 3-D
cell size 3-D
shifted flag 1-D
hierarchy level 1-D
```

Metadata 提供 token 的空間位置、window 尺度與 fine/coarse 層級。正式版本
沒有 shifted windows，所以 shifted flag 為 0。

### Validity Mask 與 Tokens

Window 至少包含 2 個點才有效，否則在 Transformer 與 pooling 中被 mask。

```text
16 fine + 4 coarse = 20 tokens/frame
token dimension = 64
token tensor = [20,64]
```

## 6. Within-Frame Transformer

正式設定：

```text
depth = 1
heads = 4
token dimension = 64
FFN dimension = 128
activation = GELU
```

Multi-head self-attention 只在**同一 frame 的 20 個 hierarchy tokens**
之間運算，用來學習 fine、coarse 與跨尺度區域的關係。Add & Norm 保留原
token 並穩定訓練；FFN 執行 `64→128→64` 非線性轉換。

這不是跨 frame Transformer，不直接計算 t 與 t+1 的 motion association。

## 7. Token Pool 與 Local Temporal Aggregation

對有效 mixed tokens 同時計算 masked mean 與 masked max：

```text
mean token = 64-D
max token = 64-D
concat = 128-D
Linear(128→64) + LayerNorm + ReLU
```

Mean 提供整體統計，max 保留顯著局部反應，輸出每個 frame 的 64-D
hierarchy feature h_t。

```text
h_seq = max_t(h_t)
Linear(64→256) + LayerNorm
```

Local temporal aggregation 仍使用 max，所以 MPH-Gait 仍未顯式建模
frame order；Transformer 強化的是單 frame 內局部與跨尺度關係。

## 8. Scale-Matched Residual

```text
local_direction = z_h / ||z_h||_2
delta_h = local_direction * stopgrad(||z_g||_2)
```

這使 local branch 主要學習修正方向，讓 local/global 具有相近數值尺度，
並避免 local branch 藉由 global norm 反向操控 base path。

## 9. Bounded Residual Fusion

```text
w = 0.20 * tanh(alpha)
z = z_g + w * delta_h
```

w 是可學習 signed scalar，範圍為 (-0.20,+0.20)，初始值為 0.05。正值加入
local correction，負值沿反方向修正。Cap 0.20 避免小資料下 local branch
完全覆蓋 global representation。

## 10. Final Descriptor

```text
embedding = BNNeck(z)
descriptor = L2_normalize(embedding)
descriptor dimension = 256
```

訓練時 embedding 接 CE classifier；Gallery 註冊與 Probe 辨識時，只使用
normalized descriptor 計算 cosine similarity。

## 11. 模型差異

| 項目 | PointNet-TMax | MPH-Gait |
|---|---|---|
| Input | Raw metric XYZ | Raw metric XYZ |
| Shared stem | `3→64→128→256→512` | 相同 |
| Global point pooling | Max | Max |
| Global temporal pooling | Max | Max |
| Local hierarchy | 無 | 16 fine + 4 coarse windows |
| Token mixer | 無 | Within-frame Transformer |
| Cross-frame attention | 無 | 無 |
| Fusion | 無 | Bounded scale-matched residual |
| Descriptor | 256-D | 256-D |
| Parameters | 313,664 | 405,954 |

MPH-Gait 增加 92,290 parameters，約增加 29.4%；兩個模型都低於 0.5M
parameters。

## 12. 論文 Method 描述

```text
MPH-Gait preserves the lightweight global PointNet representation while adding a
multi-scale local point-hierarchy branch. Each frame is partitioned into 16 fine
and four coarse regular windows. Content- and position-aware pooling converts the
irregular points in each window into 64-dimensional tokens, which are mixed by a
single within-frame Transformer encoder. The resulting hierarchy representation is
temporally max-pooled and injected into the global embedding through a bounded,
scale-matched residual.
```

## 13. 必須避免的錯誤解讀

```text
1. MPH-Gait 使用 shifted windows。       -> 錯，正式版本只有 regular windows。
2. Transformer 建模跨 frame motion。    -> 錯，只混合同 frame tokens。
3. Local branch 取代 global PointNet。   -> 錯，它是 bounded correction。
4. Min-max normalization 用於主特徵。   -> 錯，只用於 window routing。
5. MPH-Gait 顯式建模 frame order。       -> 錯，temporal pooling 仍是 max。
```
