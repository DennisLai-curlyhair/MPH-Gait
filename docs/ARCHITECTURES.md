# Architectures

## PointNet-TMax

```text
[B,T,1024,3] metric point clouds
  -> per-frame center normalization (scale retained)
  -> shared point MLP: 3 -> 64 -> 128 -> 256 -> 512
  -> point max pooling
  -> temporal max pooling
  -> 512 -> 256 embedding head
  -> BNNeck -> CE classifier / cosine retrieval
```

PointNet-TMax has 313,664 parameters and produces a 256-D descriptor.

## MPH-Gait

```text
Axis-aligned metric point sequence [B,T,1024,3]
                |
                +---------------- Global path ----------------+
                | shared point MLP -> point max -> temporal max|
                |                      -> base embedding (256)  |
                |                                               |
                +--------------- Point-hierarchy path ----------+
                  shared per-point features
                    -> fine regular grid 2 x 2 x 4: 16 tokens
                    -> coarse regular grid 1 x 2 x 2: 4 tokens
                    -> metadata-aware point-token pooling
                    -> one-layer, four-head within-frame Transformer
                    -> valid-token mean + max
                    -> per-frame local feature
                    -> temporal max
                    -> normalized local residual (256)

fused = base + w * local_residual
w = 0.20 * tanh(alpha)
  -> BNNeck -> CE classifier / cosine retrieval
```

The formal model uses 20 local tokens per frame, no shifted windows, and no
cross-frame Transformer. Temporal evidence is retained through the shared
15-frame input and temporal max pooling. MPH-Gait has 405,954 parameters and a
256-D descriptor.

## Projection

RGB-depth, gray-depth, and silhouette use the same projection model:

```text
projected frame sequence
  -> ResNet9-style frame encoder
  -> temporal max pooling
  -> horizontal pyramid pooling (16 bins)
  -> separate FC and BNNeck heads
  -> 4096-D flattened descriptor
```

Only the input representation/channel count changes.

## LidarGait++

LidarGait++ is fetched from the official OpenGait repository at the pinned
commit documented in `docs/THIRD_PARTY.md`. The adapter changes the dataset,
coordinate conversion, split, checkpoint selection, and common retrieval
reporting; it does not replace the official model implementation.
