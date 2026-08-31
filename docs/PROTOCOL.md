# Fixed-Special5 Protocol

Protocol ID: `fixed_special5_subject_holdout_5split_v1`.

## Identity Contract

Fixed clothing-stress IDs:

```text
P003, P007, P018, P019, P026
```

The other 24 identities are partitioned into five development groups. At each
split, three groups train the model, the next group validates it, and one group
is the general test cohort. The fixed five never enter training or validation.
Exact IDs are stored in `dataset/metadata/fixed_special5_protocol.json`.

## Samples

```text
train:   train IDs, C2/C3 normal, V001-V016
gallery: held-out IDs, C2/C3 normal, V017-V020
C1:      personal-clothing probes
special: C4/C5/C7 probes
```

Point-cloud models use 15 frames, 1024 sampled points per frame, skip the first
30 frames, and map Kinect camera coordinates to `[forward,lateral,height_up]`
in meters.

## Selection and Test

For each seed/split, checkpoints are selected only by validation IDs:

```text
validation gallery = C2/C3 normal
validation probe   = C1
selection metric   = query-micro mAP
```

Fixed-five C1 and special results are evaluated only after checkpoint selection.
The same fixed five are reused across development splits; split variation thus
measures training/development sensitivity, not five independent special cohorts.

## Metrics

- Rank-1/Rank-5: correct identity appears within the first 1/5 gallery ranks.
- Query-micro mAP: every probe sequence receives equal weight.
- Subject-macro mAP: first average probes within each subject, then average
  subjects equally. This is the primary fixed-five fairness metric because the
  five subjects have unequal probe counts.
