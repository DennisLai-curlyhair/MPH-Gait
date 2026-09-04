# Checkpoint Distribution

Formal weights are kept outside ordinary Git history. The four CSV manifests
define every selected checkpoint by method, matched seed, split, filename,
byte size, SHA-256, and download URL:

    projection-baselines/checkpoints/checkpoint_manifest.csv
    pointnet-tmax/checkpoints/checkpoint_manifest.csv
    mph-gait/checkpoints/checkpoint_manifest.csv
    lidargaitpp/checkpoints/checkpoint_manifest.csv

The complete bundle contains 90 selected checkpoints:

    Projection:  3 representations x 3 seeds x 5 splits = 45
    PointNet-TMax:       3 seeds x 5 splits = 15
    MPH-Gait:    3 seeds x 5 splits = 15
    LidarGait++: 3 seeds x 5 splits = 15

Checkpoint distribution location: **TBD**.

Each downloaded file can be verified against the byte size and SHA-256 stored
in its manifest.

Checkpoint hashes identify the exact weights used for the compact metrics under
each method's results/formal_len15_5split_3seed directory.

