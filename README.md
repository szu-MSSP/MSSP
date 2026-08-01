## Datasets

We evaluate the performance of our proposed algorithms and competitors on five real-world graph datasets: Gowalla, DBLP, UKDomain07, UKDomain06, and Papers100M.

The datasets can be downloaded from: [https://zenodo.org/records/21663651](https://zenodo.org/records/21663651).

## Configurations of the MSSP variants

| Variant | Sampling Pool | Sampling Distribution | Aggregation Strategy |
|---|---|---|---|
| CUR-SL2-AA<br>$T^2$-SL2-AA | Shared | $\ell_2^2$-based | Averaging |
| CUR-PL2-AA<br>$T^2$-PL2-AA | Partition-based | $\ell_2^2$-based | Averaging |
| CUR-PL2-VA<br>$T^2$-PL2-VA | Partition-based | $\ell_2^2$-based | Vertex-level |
| CUR-PD-VA<br>$T^2$-PD-VA | Partition-based | Degree-aware | Vertex-level |

## Environment

The proposed algorithms are implemented in Python (3.11.13) using NumPy (2.3.4) and SciPy (1.17.1), with parallel execution provided by Python's built-in threading module. Ground-truth PageRank values are computed using NetworKit (11.2.1).

All experiments are conducted on a server equipped with four Intel Xeon E7-4830 CPUs (56 cores at 2.0 GHz) and 2 TB of main memory.

## Algorithms

### 1. CUR-SL2-AA and $T^2$-SL2-AA
```bash
# args[1]: graph path
# args[2]: algorithm name, CUR or T2
# args[3]: output directory
# args[4]: number of nodes in the dataset
# args[5]: transition matrix and node-ID cache directory
# args[6]: sampling probability cache directory
# args[7]: number of threads
# args[8]: sampling ratios, e.g., 0.0001 ... 0.001

python MSSP.py \
  <graph_path> \
  <algorithm> \
  <output_dir> \
  <node_num> \
  <cache_dir> \
  <probability_dir> \
  <num_threads> \
  <sampling_ratio_1> <sampling_ratio_2> ...
```

### 2. CUR-PL2-AA and $T^2$-PL2-AA
```bash
python PL2-AA.py \
  <graph_path> \
  <algorithm> \
  <output_dir> \
  <node_num> \
  <cache_dir> \
  <probability_dir> \
  <num_threads> \
  <sampling_ratio_1> <sampling_ratio_2> ...
```

### 3. CUR-PL2-VA and $T^2$-PL2-VA
```bash
# args[1]: graph path
# args[2]: algorithm name, CUR or T2
# args[3]: output directory
# args[4]: number of nodes in the dataset
# args[5]: transition matrix and node-ID cache directory
# args[6]: sampling probability cache directory
# args[7]: number of threads
# args[8]: lambda, e.g., 0.9
# args[9:]: sampling ratios, e.g., 0.0001 ... 0.001

python PL2-VA.py \
  <graph_path> \
  <algorithm> \
  <output_dir> \
  <node_num> \
  <cache_dir> \
  <probability_dir> \
  <num_threads> \
  <lambda> \
  <sampling_ratio_1> <sampling_ratio_2> ...
```

### 4. CUR-PD-VA and $T^2$-PD-VA
```bash
python PD-VA.py \
  <graph_path> \
  <algorithm> \
  <output_dir> \
  <node_num> \
  <cache_dir> \
  <probability_dir> \
  <num_threads> \
  <lambda> \
  <sampling_ratio_1> <sampling_ratio_2> ...
```

### 5. Ground-truth

```bash
# args[1]: graph path
# args[2]: output path

python Groundtruth.py \
  <graph_path> \
  <output_path>
```
