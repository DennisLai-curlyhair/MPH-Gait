# 後續實驗執行說明

完整且具版本控制的實驗矩陣請以
[`EXPERIMENTS.md`](EXPERIMENTS.md) 為準。所有指令均從 repository 根目錄執行。

## 主實驗

```bash
WANDB=off bash pointnet-tmax/scripts/run_len15_5split_3seed.sh
WANDB=off bash mph-gait/scripts/run_len15_5split_3seed.sh
bash lidargaitpp/scripts/fetch_opengait.sh
WANDB=off bash lidargaitpp/scripts/run_len15_5split_3seed.sh
WANDB=off bash projection-baselines/scripts/run_len15_5split_3seed.sh
```

## Clip Length

```bash
SUITES=clip_length LENGTHS="1 4 8 30" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

## Training Clothing

```bash
SUITES=train_clothing TRAIN_VARIANTS="c2 c3" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

## Training Video Count

```bash
SUITES=train_video_count MAX_VIDEO_IDS="1 2 4 8" SEEDS="0 1 2" \
WANDB=off bash mph-gait/scripts/run_followup_5split_3seed.sh
```

以上環境變數也適用於 PointNet-TMax、adapted LidarGait++ 與 projection
follow-up launcher。以 `DRY_RUN=1 SEEDS=0 SPLITS=0` 可先檢查設定而不進行訓練。
