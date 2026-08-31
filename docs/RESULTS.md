# Formal Results

All table values are means over 15 matched cells: 5 splits x seeds 0, 1, 2.
The machine-readable table is `docs/results/main_results.csv`; every cell's
config, protocol, provenance, retrieval metrics, and summary is stored under its
method directory.

## Main Finding

MPH-Gait changes PointNet-TMax as follows on fixed-five special clothing:

| Metric | PointNet-TMax | MPH-Gait | Delta |
|---|---:|---:|---:|
| Rank-1 | 0.6845 | 0.7294 | +0.0449 |
| Rank-5 | 0.8731 | 0.8888 | +0.0157 |
| Query mAP | 0.6698 | 0.6811 | +0.0113 |
| Subject mAP | 0.6467 | 0.6596 | +0.0129 |

MPH-Gait adds 92,290 parameters over PointNet-TMax and remains below 0.5M parameters.
The strongest evidence is improved special-clothing top-rank retrieval with a
small descriptor and model. The mAP gains are smaller and should not be
described as universal or statistically decisive without reporting paired
cell/seed analyses.

LidarGait++ is strongest on normal/personal clothing, while MPH-Gait has higher
special Rank-1, Rank-5, and mean special mAP in this dataset/protocol. This is a
scenario-specific comparison, not a claim that the original LidarGait++ method
is generally inferior.
