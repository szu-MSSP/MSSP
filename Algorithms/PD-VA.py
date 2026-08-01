import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import re
import sys
import gc
import time
import math
import queue
import threading
import traceback
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from tqdm import tqdm

def load_or_build_A(graph_path, node_num, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    dataset_name = os.path.basename(graph_path).split('.')[0]
    A_path = os.path.join(cache_dir, f"{dataset_name}_norm.npz")
    csc_path = os.path.join(cache_dir, f"{dataset_name}_norm_csc.npz")
    nodeids_path = os.path.join(cache_dir, f"{dataset_name}_node_ids.npz")

    if os.path.exists(A_path) and os.path.exists(csc_path) and os.path.exists(nodeids_path):
        print(f"[NormCache] Loading", flush=True)
        A = sps.load_npz(A_path)
        A_csc = sps.load_npz(csc_path)
        node_ids = np.load(nodeids_path)["node_ids"]
        if A.shape != (node_num, node_num):
            raise ValueError(f"A cache shape mismatch: A.shape={A.shape}, node_num={node_num}")
        if A_csc.shape != (node_num, node_num):
            raise ValueError(f"A_csc cache shape mismatch: A_csc.shape={A_csc.shape}, node_num={node_num}")
        if len(node_ids) != node_num:
            raise ValueError(f"node_ids length mismatch: len={len(node_ids)}, node_num={node_num}")
        return A, A_csc, node_ids

    print("[NormCache] Not found, building AT from edge file ", flush=True)

    node_set = set()
    with open(graph_path, "r") as file:
        for line in tqdm(file, desc="Reading edges for node_set"):
            if line.startswith("%") or line.startswith("#"):
                continue
            from_node, to_node = map(int, line.strip().split()[:2])
            node_set.add(from_node)
            node_set.add(to_node)

    if len(node_set) != node_num:
        raise ValueError(
            f"node_num mismatch: argument={node_num}, edge_file_nodes={len(node_set)}"
        )

    node_dict = {element: index for index, element in enumerate(node_set)}
    node_ids = np.empty(node_num, dtype=np.int64)
    for orig_id, new_idx in node_dict.items():
        node_ids[new_idx] = orig_id

    row, col = [], []
    with open(graph_path, "r") as file:
        for line in tqdm(file, desc="Reading edges for AT (to->from)"):
            if line.startswith("%") or line.startswith("#"):
                continue
            from_node, to_node = map(int, line.strip().split()[:2])
            row.append(node_dict[to_node])
            col.append(node_dict[from_node])

    data = np.ones(len(row), dtype=float)
    A = sp.sparse.coo_array((data, (row, col)), shape=(node_num, node_num), dtype=float)

    print("[NormCache] AT ...", flush=True)
    S = A.sum(axis=0)
    S[S != 0] = 1.0 / S[S != 0]
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A_norm = A @ Q
    A_csc = A_norm.tocsc()
    sps.save_npz(A_path, A_norm)
    sps.save_npz(csc_path, A_csc)
    print(f"[MapCache] Saving", flush=True)
    np.savez_compressed(nodeids_path, node_ids=node_ids)

    del A, Q, S, node_set, node_dict, row, col, data
    gc.collect()
    return A_norm, A_csc, node_ids

def load_node_ids(graph_path, cache_dir):
    dataset_name = os.path.basename(graph_path).split('.')[0]
    nodeids_path = os.path.join(cache_dir, f"{dataset_name}_node_ids.npz")
    if not os.path.exists(nodeids_path):
        raise FileNotFoundError(f"[ERR] node_ids cache not found: {nodeids_path}")
    print(f"[MapCache] Loading node_ids from {nodeids_path}", flush=True)
    return np.load(nodeids_path)["node_ids"]

def ensure_sampling_dist(graph_path, node_num, cache_dir, proc_dir, algo):
    os.makedirs(proc_dir, exist_ok=True)
    dataset_name = os.path.basename(graph_path).split('.')[0]
    rowp_path = os.path.join(proc_dir, f"{dataset_name}_row_p.npy")
    colp_path = os.path.join(proc_dir, f"{dataset_name}_col_p.npy")
    t2p_path = os.path.join(proc_dir, f"{dataset_name}_t2_p.npy")
    if algo == "CUR":
        if os.path.exists(rowp_path) and os.path.exists(colp_path):
            print("Found row_p and col_p in proc_dir for CUR, skip computing.", flush=True)
            return rowp_path, colp_path
    elif algo == "T2":
        if os.path.exists(t2p_path):
            print(" Found t2_p in proc_dir for T2, skip computing.", flush=True)
            return t2p_path
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print("[DistCache] Not found. Computing cprob sampling distribution ...", flush=True)
    A, A_csc, _ = load_or_build_A(graph_path, node_num, cache_dir)

    A_sq = A.power(2)
    col_sums = np.asarray(A_sq.sum(axis=0)).ravel()  
    row_sums = np.asarray(A_sq.sum(axis=1)).ravel()  
    col_w = np.zeros_like(col_sums, dtype=np.float64)
    valid_col = col_sums > 0
    col_w[valid_col] = 1.0 / col_sums[valid_col]
    row_w = row_sums

    if algo == "CUR":
        col_sum = float(col_w.sum())
        row_sum = float(row_w.sum())
        if col_sum <= 0.0 or not np.isfinite(col_sum):
            raise ValueError(f"Invalid cprob CUR col_w sum: {col_sum}")
        if row_sum <= 0.0 or not np.isfinite(row_sum):
            raise ValueError(f"Invalid cprob CUR row_w sum: {row_sum}")
        col_p = col_w / col_sum
        row_p = row_w / row_sum

        print(f"[DistCache] Saving", flush=True)
        np.save(rowp_path, row_p.astype(np.float64))
        np.save(colp_path, col_p.astype(np.float64))
        del A_sq, col_sums, row_sums, col_w, row_w, valid_col, col_p, row_p
        gc.collect()
        return rowp_path, colp_path

    elif algo == "T2":
        p_t2 = np.multiply(col_w, row_w)
        p_sum = float(p_t2.sum())
        if p_sum <= 0.0 or not np.isfinite(p_sum):
            raise ValueError(f"Invalid cprob T2 probability sum: {p_sum}")
        t2_p = p_t2 / p_sum
        print(f"[DistCache] Saving", flush=True)
        np.save(t2p_path, t2_p.astype(np.float64))
        del A_sq, col_sums, row_sums, col_w, row_w, valid_col, p_t2, t2_p
        gc.collect()
        return t2p_path

def build_cr(proc_id, cpu_id, algo, A_csr, A_csc, I, J):
    t_slice = time.perf_counter()
    C = A_csc[:, J]
    R = A_csr[I, :]
    W = None
    if algo == "CUR":
        t_w = time.perf_counter()
        W = R[:, J]
        W = W.tocsr()
        w_time = time.perf_counter() - t_w

    A_nnz = A_csr.nnz
    node_num_local = A_csr.shape[0]
    print(f"[proc{cpu_id}] the number of C edges ratio: {C.nnz / A_nnz:.18f}", flush=True)
    print(f"[proc{cpu_id}] the number of R edges ratio: {R.nnz / A_nnz:.18f}", flush=True)

    slice_time = time.perf_counter() - t_slice
    return C, R, W, slice_time

def stage2_cpu_cur(proc_id, cpu_id, node_num, sampling_ratio, W, C, R):
    t_svds_start = time.perf_counter()
    nnz_count = W.nnz
    r_dim, c_dim = W.shape
    k = round(math.sqrt(min(r_dim, c_dim)))
    k_try = k
    need_eps = False
    while k_try >= 2:
        try:
            X, Z, YT = spsl.svds(W, k=k_try, which="LM", solver='propack')
            break
        except np.linalg.LinAlgError as e:
            msg = str(e)
            m = re.search(r"dimension (\d+)", msg)

            if m:
                new_k = int(m.group(1))
                if new_k >= k_try:
                    raise RuntimeError(f"[SVD][CPU] fallback did not decrease: {k_try} -> {new_k}")
                k_try = new_k
                need_eps = True
                continue
            if "did not converge" in msg:
                new_k = max(2, int(k_try * 0.8))  
                if new_k >= k_try:
                    raise RuntimeError(
                        f"[SVD][CPU] non-converged fallback did not decrease: {k_try} -> {new_k}"
                    )

                k_try = new_k
                need_eps = True
                continue
            raise
    sigma_max = float(np.max(Z))
    eps = 1e-3 * sigma_max
    mask = Z > eps
    Z = Z[mask]
    X = X[:, mask]
    YT = YT[mask, :]
    Z = 1.0 / Z
    X_T = X.T
    Y = YT.T

    def apply_U_cpu(v_r):
        u_k = X_T @ v_r
        u_k = Z * u_k
        return Y @ u_k
    t_svds_end = time.perf_counter()
    svds_time = t_svds_end - t_svds_start
    t_iter_start = time.perf_counter()
    col_num = C.shape[1]
    alpha = 0.85
    tol = 1 / node_num / 100
    R_cpu = np.full(node_num, 1.0 / node_num)
    tmp_r_cpu = R @ R_cpu
    R_cpu = apply_U_cpu(tmp_r_cpu)
    R_c = R_cpu
    P = np.full(node_num, 1.0 / node_num)

    iter_count = 0
    prev_err = None
    final_err = None
    for step in range(20):
        R_last_cpu = R_cpu
        R_cpu = R @ (C @ R_cpu)
        R_cpu = apply_U_cpu(R_cpu)
        R_cpu = alpha * R_cpu + (1 - alpha) * R_c

        err = float(np.abs(R_cpu - R_last_cpu).sum())
        if prev_err is not None and err > prev_err:
            print(f"rollback to iteration {iter_count}",flush=True,)
            R_cpu = R_last_cpu
            final_err = prev_err
            break
        iter_count = step + 1
        final_err = err
        prev_err = err
        if err < col_num * tol and (step + 1) >= 3:
            break
    R_cpu = (1 - alpha) * P + alpha * (C @ R_cpu)
    R_cpu = np.abs(R_cpu)
    norm1 = float(np.linalg.norm(R_cpu, ord=1))
    R_cpu /= norm1

    t_iter_end = time.perf_counter()
    iter_time =  t_iter_end - t_iter_start
    final_avg_err = final_err / col_num
    print(f"[proc{cpu_id}]iterations: {iter_count}, final avg err  : {final_avg_err:.18f}", flush=True)
    return R_cpu, svds_time, iter_time, final_avg_err, t_svds_start, t_svds_end, t_iter_start, t_iter_end

def stage2_cpu_t2(proc_id, cpu_id, node_num, sampling_ratio, J, C, R):

    t_iter_start = time.perf_counter()
    alpha = 0.85
    tol = 1 / node_num / 100
    c_dim = C.shape[1]
    P = np.full(node_num, 1.0 / node_num)
    r_cpu = R @ P
    R_c_cpu = r_cpu

    iter_count = 0
    for step in range(20):
        iter_count += 1
        r_last_cpu = r_cpu
        temp_r_cpu = C @ r_cpu
        r_cpu = R @ temp_r_cpu
        r_cpu = (alpha ** 2) * r_cpu + (1 - alpha ** 2) * R_c_cpu
        err = float(np.abs(r_cpu - r_last_cpu).sum())
        if err < c_dim * tol and step + 1 >= 3:
            break
    r_cpu = C @ r_cpu
    r_cpu = (alpha / (1 + alpha)) * r_cpu + (1 - alpha) * P

    r_cpu /= float(np.linalg.norm(r_cpu, ord=1))
    t_iter_end = time.perf_counter()
    iter_time = t_iter_end - t_iter_start
    final_avg_err = err / c_dim
    print(f"[proc{cpu_id}]iterations: {iter_count}, final avg err  : {final_avg_err:.18f}", flush=True)
    return r_cpu, 0.0, iter_time, final_avg_err, -1.0, -1.0, t_iter_start, t_iter_end


def merge_nodewise_epsilon(res, row_owner, col_owner, epsilon, chunk_size=1_000_000):
    D, node_num = res.shape
    eps = float(epsilon)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"epsilon must be in (0, 1), got {eps}")
    R_merge = np.empty(node_num, dtype=np.float64)
    case_none_total = 0
    case_same_total = 0
    case_diff_total = 0
    case_one_total = 0
    t_merge_start = time.perf_counter()

    for l in range(0, node_num, chunk_size):
        r_end = min(node_num, l + chunk_size)
        m = r_end - l
        rr = row_owner[l:r_end]
        cc = col_owner[l:r_end]
        weights = np.full((D, m), 1.0 / D, dtype=np.float64)

        mask_same = (rr >= 0) & (rr == cc)
        case_same_total += int(np.count_nonzero(mask_same))
        if D > 1 and np.any(mask_same):
            other_w = (1.0 - eps) / (D - 1)
            weights[:, mask_same] = other_w

            for d in range(D):
                md = mask_same & (rr == d)
                if np.any(md):
                    weights[d, md] = eps

        mask_diff = (rr >= 0) & (cc >= 0) & (rr != cc)
        case_diff_total += int(np.count_nonzero(mask_diff))
        if np.any(mask_diff):
            if D == 2:
                weights[:, mask_diff] = 0.5
            else:
                other_w = (1.0 - eps) / (D - 2)
                weights[:, mask_diff] = other_w

                for d in range(D):
                    md_r = mask_diff & (rr == d)
                    if np.any(md_r):
                        weights[d, md_r] = eps / 2.0

                    md_c = mask_diff & (cc == d)
                    if np.any(md_c):
                        weights[d, md_c] = eps / 2.0

        mask_one = ((rr >= 0) & (cc < 0)) | ((rr < 0) & (cc >= 0))
        case_one_total += int(np.count_nonzero(mask_one))

        if D > 1 and np.any(mask_one):
            other_w = (1.0 - eps / 2.0) / (D - 1)
            weights[:, mask_one] = other_w

            hit_owner = np.where(rr >= 0, rr, cc)
            for d in range(D):
                md = mask_one & (hit_owner == d)
                if np.any(md):
                    weights[d, md] = eps / 2.0
        mask_none = (rr < 0) & (cc < 0)
        case_none_total += int(np.count_nonzero(mask_none))

        R_merge[l:r_end] = np.sum(res[:, l:r_end] * weights, axis=0)

    norm1 = float(np.linalg.norm(R_merge, ord=1))
    if norm1 == 0.0:
        raise RuntimeError("[ERR] node-wise epsilon merge got zero L1 norm.")
    R_merge /= norm1
    merge_time = time.perf_counter() - t_merge_start
    stats = {
        "case_same": case_same_total, 
        "case_diff": case_diff_total,  
        "case_one": case_one_total,    
        "case_none": case_none_total,  
        "merge_time": merge_time,
    }

    return R_merge, stats

def merge_nodewise_t2_epsilon(res, t2_owner, epsilon, chunk_size=1_000_000):
    D, node_num = res.shape
    eps = float(epsilon)

    if not (0.0 < eps < 1.0):
        raise ValueError(f"epsilon must be in (0, 1), got {eps}")
    if D < 2:
        raise ValueError("node-wise T2 epsilon merge needs num_threads >= 2")
    R_merge = np.empty(node_num, dtype=np.float64)
    case_hit_total = 0
    case_none_total = 0
    t0_all = time.perf_counter()
    other_w = (1.0 - eps) / (D - 1)

    for l in range(0, node_num, chunk_size):
        r_end = min(node_num, l + chunk_size)
        owner = t2_owner[l:r_end]
        sum_all = np.sum(res[:, l:r_end], axis=0)
        out = sum_all / D
        mask_hit = owner >= 0
        case_hit_total += int(np.count_nonzero(mask_hit))
        case_none_total += int(np.count_nonzero(owner < 0))

        if np.any(mask_hit):
            out_hit_base = other_w * sum_all
            for d in range(D):
                md = mask_hit & (owner == d)
                if np.any(md):
                    out[md] = out_hit_base[md] + (eps - other_w) * res[d, l:r_end][md]
        R_merge[l:r_end] = out

    norm1 = float(np.linalg.norm(R_merge, ord=1))
    if norm1 == 0.0 or not np.isfinite(norm1):
        raise RuntimeError(f"[ERR] node-wise T2 epsilon merge got invalid L1 norm: {norm1}")
    R_merge /= norm1

    merge_time = time.perf_counter() - t0_all
    stats = {
        "case_hit": case_hit_total,
        "case_none": case_none_total,
        "merge_time": merge_time,
        "epsilon": eps,
        "chunk_size": chunk_size,
    }
    return R_merge, stats

def worker_thread(proc_id,algo,node_num,A_csr,A_csc,row_p,col_p,t2_p,
    marks,res,ee,tt,row_owner,col_owner,t2_owner,task_queue,error_list,num_threads):
    rng = np.random.default_rng()
    t_enter = time.perf_counter()
    marks[proc_id, 0] = t_enter
    marks[proc_id, 1] = t_enter

    try:
        while True:
            task = task_queue.get()
            if task is None:
                break

            job_id, sampling_ratio, a, b = task
            t_sample_start = time.perf_counter()
            c_total = int(node_num * sampling_ratio)
            c = (c_total * (proc_id + 1)) // num_threads - (c_total * proc_id) // num_threads
            block_start = (node_num * int(proc_id)) // num_threads
            block_end = (node_num * (int(proc_id) + 1)) // num_threads
            base = np.arange(block_start, block_end, dtype=np.int64)
            local_indices = (a * base + b) % node_num

            def normalize_local_p(global_p, local_indices, name):
                local_p = np.asarray(global_p[local_indices], dtype=np.float64).copy()
                local_sum = float(local_p.sum())
                if local_sum <= 0.0 or not np.isfinite(local_sum):
                    print(
                        f"[thread{proc_id}][WARN] local {name} probability sum invalid "
                        f"({local_sum}); fallback to uniform in [{block_start}, {block_end})",
                        flush=True,
                    )
                    local_p = None
                else:
                    local_p /= local_sum
                return local_p

            if algo == "T2":
                local_t2_p = normalize_local_p(t2_p, local_indices,"t2_p")
                sampled_index = rng.choice(local_indices, size=c, replace=False, p=local_t2_p)
                sampled_index.sort()
                I = sampled_index
                J = sampled_index
                if t2_owner is not None:
                    t2_owner[sampled_index] = proc_id

            elif algo == "CUR":
                local_col_p = normalize_local_p(col_p, local_indices,"col_p")
                local_row_p = normalize_local_p(row_p, local_indices,"row_p")
                J = rng.choice(local_indices, size=c, replace=False, p=local_col_p)
                I = rng.choice(local_indices, size=c, replace=False, p=local_row_p)
                I.sort()
                J.sort()
                if row_owner is not None and col_owner is not None:
                    row_owner[I] = proc_id
                    col_owner[J] = proc_id
            else:
                raise ValueError(f"Unknown algo for sampling: {algo}")

            idx_time = time.perf_counter() - t_sample_start
            t_sample_end = time.perf_counter()

            marks[proc_id, 2] = t_sample_start
            marks[proc_id, 3] = t_sample_end
            t_cr_start = time.perf_counter()
            C, R, W, cr_slice_time = build_cr(
                proc_id=proc_id,
                cpu_id=proc_id,
                algo=algo,
                A_csr=A_csr,
                A_csc=A_csc,
                I=I,
                J=J,
            )
            t_cr_end = time.perf_counter()
            marks[proc_id, 4] = t_cr_start
            marks[proc_id, 5] = t_cr_end

            if algo == "CUR":
                R_full, svd_time, iter_time, final_avg_err, t_svd_start, t_svd_end, t_iter_start, t_iter_end = (
                    stage2_cpu_cur(proc_id, proc_id, node_num, sampling_ratio, W, C, R)
                )
            elif algo == "T2":
                R_full, svd_time, iter_time, final_avg_err, t_svd_start, t_svd_end, t_iter_start, t_iter_end = (
                    stage2_cpu_t2(proc_id, proc_id, node_num, sampling_ratio, J, C, R)
                )
            else:
                raise ValueError(f"Unknown algo: {algo}")

            marks[proc_id, 6] = t_svd_start
            marks[proc_id, 7] = t_svd_end
            marks[proc_id, 8] = t_iter_start
            marks[proc_id, 9] = t_iter_end

            t_cleanup_start = time.perf_counter()
            del C, R, W
            gc.collect()
            t_cleanup_end = time.perf_counter()
            marks[proc_id, 10] = t_cleanup_start
            marks[proc_id, 11] = t_cleanup_end
            t_shm_write_start = time.perf_counter()
            res[proc_id, :] = R_full
            ee[proc_id] = final_avg_err
            tt[proc_id, 0] = idx_time
            tt[proc_id, 1] = cr_slice_time
            tt[proc_id, 2] = svd_time
            tt[proc_id, 3] = iter_time
            t_shm_write_end = time.perf_counter()
            marks[proc_id, 12] = t_shm_write_start
            marks[proc_id, 13] = t_shm_write_end
            marks[proc_id, 14] = time.perf_counter()

    except Exception:
        error_list[proc_id] = traceback.format_exc()


def run_mthread(algo, graph_path, output_dir, node_num, ratio_list, cache_dir, proc_dir, num_threads, epsilon_merge):
    os.makedirs(output_dir, exist_ok=True)
    epsilon_merge = float(epsilon_merge)
    ensure_sampling_dist(graph_path, node_num, cache_dir, proc_dir, algo)
    A, A_csc, node_ids = load_or_build_A(graph_path, node_num, cache_dir)

    dataset_name = os.path.basename(graph_path).split(".")[0]
    if algo == "T2":
        t2_p = np.load(os.path.join(proc_dir, f"{dataset_name}_t2_p.npy"))
        row_p, col_p = None, None
    elif algo == "CUR":
        row_p = np.load(os.path.join(proc_dir, f"{dataset_name}_row_p.npy"))
        col_p = np.load(os.path.join(proc_dir, f"{dataset_name}_col_p.npy"))
        t2_p = None
    else:
        raise ValueError(f"Unknown algo: {algo}")

    marks = np.full((num_threads, 15), -1.0, dtype=np.float64)
    res = np.zeros((num_threads, node_num), dtype=np.float64)
    tt = np.zeros((num_threads, 4), dtype=np.float64)
    ee = np.zeros(num_threads, dtype=np.float64)
    owner_dtype = np.int16
    if algo == "CUR":
        row_owner = np.full(node_num, -1, dtype=owner_dtype)
        col_owner = np.full(node_num, -1, dtype=owner_dtype)
        t2_owner = None
    elif algo == "T2":
        row_owner = None
        col_owner = None
        t2_owner = np.full(node_num, -1, dtype=owner_dtype)
    else:
        row_owner = None
        col_owner = None
        t2_owner = None

    task_queue = queue.Queue()
    error_list = [None] * num_threads

    threads = []
    for proc_id in range(num_threads):
        th = threading.Thread(
            target=worker_thread,
            args=(
                proc_id, algo, node_num, A, A_csc, row_p, col_p, t2_p,
                marks, res, ee, tt, row_owner, col_owner, t2_owner, task_queue, error_list,num_threads
            ),
            daemon=True,
        )
        threads.append(th)

    t_end2end_start = time.perf_counter()
    for th in threads:
        th.start()

    startup_wall = time.perf_counter() - t_end2end_start

    rng = np.random.default_rng()

    for job_id, sampling_ratio in enumerate(ratio_list):
        t_online_start = time.perf_counter()
        file_name = f"{sampling_ratio:.10g}.txt"
        output_path = os.path.join(output_dir, file_name)

        print("\n" + "=" * 70, flush=True)
        print(f"[RUN] algo={algo}", flush=True)
        print(f"[RUN] ratio={sampling_ratio}", flush=True)
        print(f"[RUN] file_name={file_name}", flush=True)
        print(f"[RUN] output_path={output_path}", flush=True)
        print("=" * 70, flush=True)

        marks[:] = -1.0
        tt[:] = 0.0
        res[:] = 0.0
        ee[:] = 0.0
        if algo == "CUR":
            row_owner.fill(-1)
            col_owner.fill(-1)
        elif algo == "T2":
            t2_owner.fill(-1)

        for i in range(num_threads):
            error_list[i] = None

        t_ab_start = time.perf_counter()
        a_try = 0
        while True:
            a_try += 1
            a = int(rng.integers(1, node_num))
            if math.gcd(a, node_num) == 1:
                break

        b = int(rng.integers(0, node_num))

        print(
            f"[AffineMap] ratio={sampling_ratio}, a={a}, b={b} ",
            flush=True
        )
        ab_time = time.perf_counter() - t_ab_start

        for _ in range(num_threads):
            task_queue.put((job_id, sampling_ratio, a, b))

        while True:
            errs = [err for err in error_list if err is not None]
            if errs:
                raise RuntimeError("thread failed:\n" + "\n".join(errs))
            if np.all(marks[:num_threads, 13] >= 0):
                break
            time.sleep(0.001)

        if algo == "CUR":
            R_merge, owner_stats = merge_nodewise_epsilon(
                res=res,
                row_owner=row_owner,
                col_owner=col_owner,
                epsilon=epsilon_merge,
                chunk_size=1_000_000,
            )
            merge_time = owner_stats["merge_time"]
            print(f"[NodeWiseMerge] epsilon={epsilon_merge:.2f}", flush=True)

        elif algo == "T2":
            R_merge, owner_stats = merge_nodewise_t2_epsilon(
                res=res,
                t2_owner=t2_owner,
                epsilon=epsilon_merge,
                chunk_size=1_000_000,
            )
            merge_time = owner_stats["merge_time"]
            print(f"[NodeWiseMerge][T2] epsilon={epsilon_merge:.2f}", flush=True)

        else:
            raise ValueError(f"Unknown algo: {algo}")


        online_wall = time.perf_counter() - t_online_start

        sampling_phase_wall = float(np.max(marks[:, 3]) - np.min(marks[:, 2]))
        cr_phase_wall = float(np.max(marks[:, 5]) - np.min(marks[:, 4]))
        stage1_phase_wall = float(np.max(marks[:, 5]) - np.min(marks[:, 2]))

        svd_phase_wall = 0.0 if algo == "T2" else float(np.max(marks[:, 7]) - np.min(marks[:, 6]))
        iter_phase_wall = float(np.max(marks[:, 9]) - np.min(marks[:, 8]))
        stage2_phase_wall = iter_phase_wall if algo == "T2" else float(np.max(marks[:, 9]) - np.min(marks[:, 6]))

        cleanup_phase_wall = float(np.max(marks[:, 11]) - np.min(marks[:, 10]))
        write_phase_wall = float(np.max(marks[:, 13]) - np.min(marks[:, 12]))
        comparable_wall = startup_wall + online_wall

        t_save = time.perf_counter()
        with open(output_path, "w+") as f:
            for i in range(len(node_ids)):
                f.write(f"{node_ids[i]}\t{R_merge[i]:.17f}\n")
        save_time = time.perf_counter() - t_save

        print(f"proc_dir                              : {proc_dir}", flush=True)
        print(f"[Time]  : {online_wall:.4f} s", flush=True)

    t_join_start = time.perf_counter()
    for _ in threads:
        task_queue.put(None)
    for th in threads:
        th.join()
    join_time = time.perf_counter() - t_join_start
    end2end_wall = time.perf_counter() - t_end2end_start


if __name__ == "__main__":
    args = sys.argv
    graph_path = args[1]
    algo = args[2]
    output_dir = args[3]
    node_num = int(args[4])
    cache_dir = args[5]
    proc_dir = args[6]
    num_threads = int(args[7])
    epsilon_merge = float(args[8])
    ratio_list = [float(x) for x in args[9:]]

    run_mthread(
        algo=algo,
        graph_path=graph_path,
        output_dir=output_dir,
        node_num=node_num,
        ratio_list=ratio_list,
        cache_dir=cache_dir,
        proc_dir=proc_dir,
        num_threads=num_threads,
        epsilon_merge=epsilon_merge,
    )
