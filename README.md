# cbi_partitions

`cbi_partitions` is a Python library for Conformalized Bayesian Analysis (CBI) of posterior distributions over data partitions, based on partition-valued MCMC output.

Given MCMC samples and a notion of distance between data partitions, the library provides:
- a point estimate of the data clustering;
- credible sets of partitions with guaranteed posterior coverage constructed using conformal prediction principles, together with a normalized measure of posterior typicality for any given partition (interpretable as a p-value and suitable for hypothesis testing);
- tools to explore and summarize multimodal posterior structure over the space of partitions using density-based ideas.

---

## Installation

You can install the library directly from GitHub into any Python environment:

```bash
pip install https://github.com/nbariletto/cbi_partitions/archive/main.zip
```

---

<br>

## Overview

The library consists of three main components:

1. **Internal partition distance computations**, implemented in Numba for efficiency;
2. **`PartitionKDE`**, the standard density-based pipeline for CBI;
3. **`PartitionBall`**, a distance-based CBI method yielding metric credible balls.

Partitions are represented as integer-valued arrays of length `n`, where the `i`-th entry denotes the cluster label assigned to observation `i`.

---

## Partition distances (internal)

The library internally supports two distances between partitions:

- **Variation of Information (VI)**;
- **Binder distance** (one minus the Rand index).

Distance computation and optional remapping of cluster labels to a compact range are handled internally using Numba-accelerated routines. These operations are implementation details and are **not exposed as part of the public API**. Users interact with distances exclusively through the high-level conformal models described below.

---

## PartitionKDE

### Description

`PartitionKDE` implements a density-based conformal score on the space of partitions.

Given a collection of training partitions, each partition is scored by averaging an exponential kernel applied to its distances from the training set. The resulting score acts as:
- a proxy for posterior density;
- a conformity score for conformalized Bayesian inference;
- the basis for point estimation and posterior multimodality analysis.

---

### Constructor

```python
PartitionKDE(
    train_partitions,
    metric='vi',
    gamma=0.5,
    subsample_size=None,
    remap_labels=True
)
```

#### Parameters

- **`train_partitions`**: array-like of shape `(T, n)`  
  Training partitions, typically obtained from MCMC.
- **`metric`**: `'vi'` or `'binder'` (default `'vi'`)  
  Distance used internally by the kernel.
- **`gamma`**: float (default `0.5`)  
  Kernel decay parameter.
- **`subsample_size`**: int or `None` (default `None`)  
  Optional random subsample size for the training set.
- **`remap_labels`**: bool (default `True`)  
  Whether cluster labels are remapped internally.

---

### Methods

#### `score`

```python
score(partitions)
```

Computes the KDE score for one or more partitions.

- **Input**: a single partition or an array of partitions  
- **Output**: an array of scores, one per partition

Higher scores indicate greater posterior typicality.

---

#### `calibrate`

```python
calibrate(calib_partitions)
```

Scores all calibration partitions and computes additional quantities required for conformal inference and posterior exploration.

This method must be called before computing p-values.

---

#### `compute_p_value`

```python
compute_p_value(partition)
```

Computes a normalized measure of posterior typicality for a given partition.

The returned value is the fraction of calibration partitions whose score is less than or equal to the score of the given partition, with a standard finite-sample correction. It can be interpreted as a p-value under the assumption that the calibration samples and the tested partition are jointly drawn from the posterior distribution.

Higher values indicate greater posterior support.

---

#### `get_point_estimate`

```python
get_point_estimate(source='calibration')
```

Returns the partition with the highest KDE score.

- **`source`**: `'train'` or `'calibration'`

This estimator can be interpreted as a pseudo-MAP estimate.

---

## Posterior multimodality and density peaks

During calibration, the library also computes quantities used to explore posterior multimodality using density-based ideas.

For each calibration partition, the following are computed internally:
- its KDE score;
- its distance to the closest calibration partition with a higher score.

These quantities can be visualized and thresholded to identify well-separated high-density regions of the posterior.

---

### `plot_dpc_decision_graph`

```python
plot_dpc_decision_graph(save_path=None)
```

Plots the decision graph displaying KDE score versus distance to higher-density partitions.

---

### `get_dpc_modes`

```python
get_dpc_modes(s_thresh, delta_thresh)
```

Returns the indices of calibration partitions whose KDE score exceeds `s_thresh` and whose distance to higher-density samples exceeds `delta_thresh`. Returned indices are ordered by the product of score and separation.

---

## PartitionBall

### Description

`PartitionBall` implements a distance-based conformal procedure centered at a fixed point estimate. It yields credible sets that coincide with metric balls around the chosen center partition.

---

### Constructor

```python
PartitionBall(
    point_estimate_partition,
    metric='vi',
    remap_labels=True
)
```

---

### Methods

#### `score`

```python
score(partitions)
```

Computes distances between each partition and the center partition.

---

#### `calibrate`

```python
calibrate(calib_partitions)
```

Computes calibration distances to the center partition.

---

#### `compute_p_value`

```python
compute_p_value(partition)
```

Computes a non-conformity p-value based on distance to the center.

The returned value is the fraction of calibration partitions whose distance to the center is greater than or equal to that of the given partition, with a standard finite-sample correction. Smaller distances correspond to greater posterior typicality.

---

## Notes

- All distance computations are Numba-jitted for efficiency.
- Pairwise distance calculations scale quadratically in the number of calibration samples.
- The library is agnostic to the source of partition samples and can be used with any partition-valued MCMC output.

---

## References

[1] Bariletto, N., Ho, N., & Rinaldo, A. (2025). *Conformalized Bayesian Inference, with Applications to Random Partition Models*. arXiv preprint.
