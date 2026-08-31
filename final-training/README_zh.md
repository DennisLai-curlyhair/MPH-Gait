# Final-24 三模型訓練與 Fixed-5 評估

本資料夾提供 PointNet-TMax、MPH-Gait、LidarGait++ 的 Final-24 訓練流程。
它和論文用的 5-split x 3-seed subject-disjoint 比較分開保存，目的主要是：

~~~text
使用最多且仍不接觸 Fixed-5 的 development training identities，
訓練可供系統部署與公開釋出的 canonical checkpoint。
~~~

## 1. 身分與資料分配

Final-24 training IDs：

~~~text
P001 P002 P004 P005 P006 P008 P009 P010 P011 P012 P013 P014
P015 P016 P017 P020 P021 P022 P023 P024 P025 P027 P028 P029
~~~

訓練資料：

~~~text
C2/C3 normal，V001-V016
clip_len = 15
num_points = 1024
drop_first_frames = 30
~~~

固定評估 IDs：

~~~text
P003 P007 P018 P019 P026
~~~

這五人不會進入訓練，也不會用來選 checkpoint。評估 gallery 使用 Fixed-5
自己的 C2/C3 controlled samples；probe 分開統計：

~~~text
final24_fixed5_c1:
  C1 personal clothing

final24_fixed5_c457:
  C4 + C5 + C7 special clothing

final24_fixed5_c4 / c5 / c7:
  各服裝條件的獨立 breakdown
~~~

## 2. Checkpoint 規則

Final-24 已經把全部 24 位 development subjects 用於訓練，因此沒有另一組
development validation IDs。三個方法都採事先固定的 training budget，並使用
最後 checkpoint：

| Method | Fixed budget | Selected checkpoint |
|---|---:|---|
| PointNet-TMax | 50 epochs | checkpoints/final.pt |
| MPH-Gait | 50 epochs | checkpoints/final.pt |
| LidarGait++ | 10000 iterations | official *-10000.pt |

這不是根據 Fixed-5 挑最好 epoch、iteration 或 seed。正式報告應呈現三個 seeds
的 mean +/- std；公開單一 canonical 權重時可預先固定 seed 0，不能看 Fixed-5
結果後挑最高 seed。PointNet-TMax 與 MPH-Gait 的相同 seed 會使用相同的資料
順序與點採樣亂數；seed 0/1/2 之間則會同時改變初始化與訓練亂數。

## 3. 執行指令

先檢查三份 config 是否使用完全相同的 ID 與 retrieval pairs：

~~~bash
conda run -n pytorch python final-training/tools/validate_final24_protocol.py
~~~

只檢查 PointNet-TMax / MPH-Gait 的實際資料數量，不訓練：

~~~bash
SEEDS=0 DRY_RUN=1 WANDB=off bash final-training/scripts/run_pointnet_tmax_final24_3seed.sh
SEEDS=0 DRY_RUN=1 WANDB=off bash final-training/scripts/run_mph_gait_final24_3seed.sh
~~~

LidarGait++ 只檢查實際資料與 pair 數量，不轉檔、不訓練：

~~~bash
SEEDS=0 INSPECT_ONLY=1 WANDB=off bash final-training/scripts/run_lidargaitpp_final24_3seed.sh
~~~

分別執行正式 3 seeds：

~~~bash
WANDB=off bash final-training/scripts/run_pointnet_tmax_final24_3seed.sh
WANDB=off bash final-training/scripts/run_mph_gait_final24_3seed.sh
WANDB=off bash final-training/scripts/run_lidargaitpp_final24_3seed.sh
~~~

依序跑完三個方法並產生跨方法比較：

~~~bash
WANDB=off bash final-training/scripts/run_all_final24_3seed.sh
~~~

若訓練中斷，使用相同的 RUN_STAMP 或直接指定原本 RUN_ROOT 重跑；預設
SKIP_COMPLETED=1 會跳過已有最終 retrieval 結果的 seed。

## 4. 輸出隔離

所有新結果只會寫入：

~~~text
outputs/final-training/
  pointnet_tmax/<run>/seed_0..2/
  mph_gait/<run>/seed_0..2/
  lidargaitpp/<run>/seed_0..2/
  comparisons/<run>/
~~~

每個方法的 summary 會分開產生：

~~~text
summary/c1_seed_metrics.csv
summary/c1_mean_std.json
summary/c457_seed_metrics.csv
summary/c457_mean_std.json
summary/c4_c5_c7_seed_metrics.csv
summary/summary.md
~~~

一鍵三模型流程另外輸出：

~~~text
comparisons/<run>/comparison_c1.csv
comparisons/<run>/comparison_c457.csv
comparisons/<run>/comparison.md
~~~

## 5. 解讀限制

Final-24 的 Fixed-5 gallery 只有五種 identity，因此它是 5-way deployment
checkpoint sanity evaluation。Rank-5 仍會受到每人多筆 gallery sample 的排序
影響，但區辨力低於大型 gallery；主要仍看 Rank-1、query-micro mAP 與
subject-macro mAP。

這組數字也不能直接和舊 protocol 的 mixed gallery 主表相減，因為候選人與
distractor 數量不同。論文的主要泛化證據仍應使用既有 5-split x 3-seed
subject-disjoint 結果；Final-24 的角色是公開 checkpoint、系統部署，以及
固定五人 C1/C457 的描述性驗證。
