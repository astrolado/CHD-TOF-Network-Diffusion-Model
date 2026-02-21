import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from scipy import sparse

# ====================
# CONFIG
# ====================
GENE_POOL_CSV = "combined_gene_pool.csv"
GENE_COL = "Table S2  1124 CHD-related genes"

PROTEIN_INFO_TXT = "9606.protein.info.v12.0.txt"
LINKS_TXT = "9606.protein.links.detailed.v12.0.txt"

# (used for the non-sweep outputs)
SCORE_THRESHOLD = 700
RWR_RESTART = 0.4
RWR_ITERS = 50

TOP_NONSEED_PRINT = 30
TOP_NONSEED_SAVE = 200
DEDUP_BY_GENE_SYMBOL = True  # set False if you want to keep isoform duplicates
FILTER_LOC_GENES = True      # remove LOC... and empty symbols from non-seed display/save

# Cross-validation settings 
RUN_CV = True
CV_REPEATS = 50
CV_HOLDOUT_FRAC = 0.20
CV_PREC_K = [10, 50, 100]
CV_RANDOM_SEED = 42

# Parameter sweep settings
RUN_SWEEP = True
SWEEP_THRESHOLDS = [600, 700, 800]      # STRING combined_score thresholds
SWEEP_RESTARTS = [0.3, 0.4, 0.5, 0.7]   # restart probabilities
SWEEP_ITERS = 50                        # keep same iters during sweep
SWEEP_CV_REPEATS = 30                   # reduce for speed; set =CV_REPEATS if you want
SWEEP_OUTPUT_CSV = "rwr_param_sweep.csv"

# ====================
# Helpers
# ====================
def scores_to_ranks(scores_array: np.ndarray) -> np.ndarray:
    """Higher score => better rank; rank 1 is best."""
    order = np.argsort(-scores_array)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(scores_array) + 1)
    return ranks

def build_W_from_edges(n_nodes: int, u: np.ndarray, v: np.ndarray, score: np.ndarray, threshold: int):
    """Build sparse adjacency A and column-normalized transition matrix W for a given score threshold."""
    mask = score >= threshold
    uu = u[mask]
    vv = v[mask]

    # Build undirected adjacency by storing both directions
    rows = np.concatenate([uu, vv]).astype(np.int32, copy=False)
    cols = np.concatenate([vv, uu]).astype(np.int32, copy=False)
    data = np.ones(rows.shape[0], dtype=np.float32)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    A.sum_duplicates()

    # Column-normalize: W = A * D^{-1}
    colsum = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)
    colsum[colsum == 0] = 1.0
    Dinv = sparse.diags(1.0 / colsum)
    W = A @ Dinv
    return A, W

def run_rwr_from_seed_idxs(W, seed_idxs, restart: float, iters: int, n_nodes: int) -> np.ndarray:
    p0 = np.zeros(n_nodes, dtype=np.float64)
    p0[seed_idxs] = 1.0
    p0 /= p0.sum()
    p = p0.copy()
    for _ in range(iters):
        p = (1 - restart) * (W @ p) + restart * p0
    return p

def cv_repeated_holdout(W, A, seed_idxs_all, restart: float, iters: int, repeats: int, holdout_frac: float, prec_k: list[int], rng: random.Random):
    """
    Repeated holdout CV.
    Reports:
      - mean/median rank, mean percentile, MRR for held-out seeds
      - Precision@K and Recall@K for held-out seeds
      - Degree and Random baselines
    """
    n = W.shape[0]
    if len(seed_idxs_all) < 5:
        return None

    # Degree baseline ranks (once per threshold)
    deg = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)
    deg_ranks = scores_to_ranks(deg)

    rows_out = []
    for rep in range(repeats):
        n_hold = max(1, int(round(holdout_frac * len(seed_idxs_all))))
        held = rng.sample(seed_idxs_all, n_hold)
        held_set = set(held)
        train = [x for x in seed_idxs_all if x not in held_set]

        # RWR
        scores = run_rwr_from_seed_idxs(W, train, restart, iters, n)
        ranks = scores_to_ranks(scores)

        # Random baseline
        rand_scores = np.random.random(n)
        rand_ranks = scores_to_ranks(rand_scores)

        held_ranks = ranks[held]
        held_deg_ranks = deg_ranks[held]
        held_rand_ranks = rand_ranks[held]

        row = {
            "repeat": rep,
            "n_train": len(train),
            "n_hold": len(held),
            "mean_rank": float(np.mean(held_ranks)),
            "median_rank": float(np.median(held_ranks)),
            "mean_percentile": float(np.mean(held_ranks / n)),
            "mrr": float(np.mean(1.0 / held_ranks)),
            "mean_rank_degree": float(np.mean(held_deg_ranks)),
            "mrr_degree": float(np.mean(1.0 / held_deg_ranks)),
            "mean_rank_random": float(np.mean(held_rand_ranks)),
            "mrr_random": float(np.mean(1.0 / held_rand_ranks)),
        }

        # Precision@K and Recall@K
        for k in prec_k:
            topk = set(np.argsort(-scores)[:k])
            topk_deg = set(np.argsort(-deg)[:k])
            topk_rand = set(np.argsort(-rand_scores)[:k])

            hits = sum(1 for idx in held if idx in topk)
            hits_deg = sum(1 for idx in held if idx in topk_deg)
            hits_rand = sum(1 for idx in held if idx in topk_rand)

            row[f"recall@{k}"] = hits / len(held)
            row[f"precision@{k}"] = hits / k

            row[f"recall_degree@{k}"] = hits_deg / len(held)
            row[f"precision_degree@{k}"] = hits_deg / k

            row[f"recall_random@{k}"] = hits_rand / len(held)
            row[f"precision_random@{k}"] = hits_rand / k

        rows_out.append(row)

    df = pd.DataFrame(rows_out)
    mean = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
    return df, mean, std

# ====================
# 1) Load seed genes
# ====================
df = pd.read_csv(GENE_POOL_CSV)
seed_genes = (
    df.loc[df["is_seed_gene"] == True, GENE_COL]
      .dropna()
      .astype(str)
      .str.strip()
      .str.upper()
      .unique()
      .tolist()
)
print("Seed genes:", len(seed_genes))

# ====================
# 2) Map seed genes -> one ENSP protein each (streaming protein.info)
# ====================
print("Mapping seed genes -> ENSP proteins (streaming protein.info)...")
print("protein.info size (MB):", round(os.path.getsize(PROTEIN_INFO_TXT) / (1024 * 1024), 2))

seed_set = set(seed_genes)
gene_to_one_prot = {}  # gene -> ENSP...

with open(PROTEIN_INFO_TXT, "r", encoding="utf-8", errors="ignore") as f:
    _ = f.readline()  # header
    for line in f:
        parts = line.strip().split(None, 3)
        if len(parts) < 2:
            continue
        prot_id, pref = parts[0], parts[1]
        pref = pref.upper()

        if prot_id.startswith("9606."):
            prot_id = prot_id[5:]
        if not prot_id.startswith("ENSP"):
            continue

        if pref in seed_set and pref not in gene_to_one_prot:
            gene_to_one_prot[pref] = prot_id
            if len(gene_to_one_prot) == len(seed_set):
                break

seed_proteins = sorted(set(gene_to_one_prot.values()))
print("Seed proteins (ENSP) mapped:", len(seed_proteins))
if len(seed_proteins) == 0:
    raise RuntimeError("No seeds mapped to ENSP IDs from protein.info.")

# ====================
# 3) Stream-parse STRING links ONCE, store undirected edges + score
#    We store edges for the *minimum* threshold so we can reuse them in the sweep.
# ====================
min_thr = min([SCORE_THRESHOLD] + (SWEEP_THRESHOLDS if RUN_SWEEP else []))
print(f"Reading STRING links once (storing edges with score >= {min_thr}) ...")

file_size = os.path.getsize(LINKS_TXT)

node_index = {}
u_list = []
v_list = []
s_list = []

def get_idx(pid: str) -> int:
    if pid not in node_index:
        node_index[pid] = len(node_index)
    return node_index[pid]

bad_lines = 0
seen_header = False

with open(LINKS_TXT, "r", encoding="utf-8", errors="ignore") as f, tqdm(
    total=file_size, unit="B", unit_scale=True, desc="Reading links"
) as pbar:
    for line in f:
        pbar.update(len(line.encode("utf-8", errors="ignore")))

        line = line.strip()
        if not line:
            continue

        if not seen_header:
            if line.lower().startswith("protein1"):
                seen_header = True
                continue
            seen_header = True

        parts = line.split()
        if len(parts) < 3:
            bad_lines += 1
            continue

        p1 = parts[0]
        p2 = parts[1]
        score_str = parts[-1]

        if p1.startswith("9606."):
            p1 = p1[5:]
        if p2.startswith("9606."):
            p2 = p2[5:]

        if not (p1.startswith("ENSP") and p2.startswith("ENSP")):
            continue

        try:
            score = int(float(score_str))
        except Exception:
            bad_lines += 1
            continue

        if score < min_thr:
            continue

        i = get_idx(p1)
        j = get_idx(p2)

        # store undirected once (i,j)
        u_list.append(i)
        v_list.append(j)
        s_list.append(score)

print("Done parsing links.")
print("Bad/ignored lines:", bad_lines)
n = len(node_index)
print("Nodes (ENSP):", n)
print("Undirected edges stored:", len(u_list))

if n == 0 or len(u_list) == 0:
    raise RuntimeError("No ENSP edges stored. Check your links file/species.")

u = np.array(u_list, dtype=np.int32)
v = np.array(v_list, dtype=np.int32)
score = np.array(s_list, dtype=np.int32)

# ====================
# 4) Build matrices for main run threshold
# ====================
A, W = build_W_from_edges(n, u, v, score, SCORE_THRESHOLD)

# ====================
# 5) Seed indices present in the network
# ====================
present_seed_idxs = [node_index[p] for p in seed_proteins if p in node_index]
print("Seed proteins present in network:", len(present_seed_idxs))
if len(present_seed_idxs) == 0:
    raise RuntimeError("No seed proteins overlap with ENSP network from links.")

# ====================
# 6) Main run RWR
# ====================
print("Running sparse RWR (main run)...")
p = run_rwr_from_seed_idxs(W, present_seed_idxs, RWR_RESTART, RWR_ITERS, n)

# ====================
# 7) Map protein IDs back to gene symbols (streaming protein.info, only for nodes)
# ====================
print("Mapping ENSP -> gene symbol (streaming protein.info)...")
idx_to_node = [None] * n
for pid, i in node_index.items():
    idx_to_node[i] = pid

need = set(idx_to_node)
prot_to_gene = {}

with open(PROTEIN_INFO_TXT, "r", encoding="utf-8", errors="ignore") as f:
    _ = f.readline()
    for line in f:
        parts = line.strip().split(None, 3)
        if len(parts) < 2:
            continue
        pid, pref = parts[0], parts[1]
        if pid.startswith("9606."):
            pid = pid[5:]
        if pid in need and pid not in prot_to_gene:
            prot_to_gene[pid] = pref.upper()
            if len(prot_to_gene) == len(need):
                break

# ====================
# 8) Save results + print novel candidates
# ====================
results = pd.DataFrame({
    "protein_id": idx_to_node,
    "gene_symbol": [prot_to_gene.get(pid, "") for pid in idx_to_node],
    "RWR_score": p
})
results["is_seed_gene"] = results["gene_symbol"].isin(seed_genes)
results = results.sort_values("RWR_score", ascending=False)

novel = results[results["is_seed_gene"] == False].copy()
if DEDUP_BY_GENE_SYMBOL:
    novel = novel.drop_duplicates(subset=["gene_symbol"])
if FILTER_LOC_GENES:
    novel = novel[novel["gene_symbol"].notna()]
    novel = novel[novel["gene_symbol"] != ""]
    novel = novel[~novel["gene_symbol"].str.startswith("LOC", na=False)]

print(f"\nTop {TOP_NONSEED_PRINT} NON-SEED candidates (main run):")
print(novel.head(TOP_NONSEED_PRINT)[["gene_symbol", "protein_id", "RWR_score"]].to_string(index=False))

results.to_csv("rwr_results.csv", index=False)
novel.head(TOP_NONSEED_SAVE).to_csv("rwr_novel_candidates_top200.csv", index=False)

print("\nSaved: rwr_results.csv")
print("Saved: rwr_novel_candidates_top200.csv")
print("\nTop results (including seeds):")
print(results.head(20))

# ====================
# 9) Cross-validation (main run parameters)
# ====================
if RUN_CV:
    print("\n=== Running cross-validation (main params) ===")
    rng = random.Random(CV_RANDOM_SEED)
    seed_idxs_all = present_seed_idxs[:]  # already present in node_index

    out = cv_repeated_holdout(
        W=W, A=A, seed_idxs_all=seed_idxs_all,
        restart=RWR_RESTART, iters=RWR_ITERS,
        repeats=CV_REPEATS, holdout_frac=CV_HOLDOUT_FRAC,
        prec_k=CV_PREC_K, rng=rng
    )
    if out is None:
        print("Too few seeds present for CV; skipping.")
    else:
        cv_df, m, s = out
        cv_df.to_csv("rwr_cv_results.csv", index=False)
        print("Saved: rwr_cv_results.csv")

        print("\n=== CV summary (main params) mean ± std ===")
        print(f"Mean rank (RWR): {m['mean_rank']:.1f} ± {s['mean_rank']:.1f}")
        print(f"Mean percentile (RWR): {m['mean_percentile']:.4f} ± {s['mean_percentile']:.4f}")
        print(f"MRR (RWR): {m['mrr']:.4f} ± {s['mrr']:.4f}")
        print(f"Mean rank (Degree): {m['mean_rank_degree']:.1f} ± {s['mean_rank_degree']:.1f}")
        print(f"Mean rank (Random): {m['mean_rank_random']:.1f} ± {s['mean_rank_random']:.1f}")

        for k in CV_PREC_K:
            print(
                f"K={k}: "
                f"Recall RWR {m[f'recall@{k}']:.3f} | Degree {m[f'recall_degree@{k}']:.3f} | Random {m[f'recall_random@{k}']:.3f} || "
                f"Precision RWR {m[f'precision@{k}']:.3f} | Degree {m[f'precision_degree@{k}']:.3f} | Random {m[f'precision_random@{k}']:.3f}"
            )

# ====================
# 10) Parameter sweep (threshold x restart)
# ====================
if RUN_SWEEP:
    print("\n=== Running parameter sweep ===")
    rng = random.Random(CV_RANDOM_SEED)  # reproducible splits across sweep

    sweep_rows = []
    for thr in SWEEP_THRESHOLDS:
        # Build A/W for this threshold
        t0 = time.time()
        A_thr, W_thr = build_W_from_edges(n, u, v, score, thr)

        # Seeds present under this graph (node set is same, but degrees may drop)
        seed_idxs_all = [node_index[p] for p in seed_proteins if p in node_index]
        if len(seed_idxs_all) < 5:
            print(f"Threshold {thr}: too few seeds for CV; skipping.")
            continue

        for restart in SWEEP_RESTARTS:
            out = cv_repeated_holdout(
                W=W_thr, A=A_thr, seed_idxs_all=seed_idxs_all,
                restart=restart, iters=SWEEP_ITERS,
                repeats=SWEEP_CV_REPEATS, holdout_frac=CV_HOLDOUT_FRAC,
                prec_k=CV_PREC_K, rng=rng
            )
            if out is None:
                continue
            cv_df, m, s = out

            row = {
                "threshold": thr,
                "restart": restart,
                "iters": SWEEP_ITERS,
                "cv_repeats": SWEEP_CV_REPEATS,
                "n_nodes": n,
                "n_edges_undirected": int(np.sum(score >= thr)),
                "mean_rank": float(m["mean_rank"]),
                "std_rank": float(s["mean_rank"]),
                "mean_percentile": float(m["mean_percentile"]),
                "mrr": float(m["mrr"]),
                "mean_rank_degree": float(m["mean_rank_degree"]),
                "mrr_degree": float(m["mrr_degree"]),
                "mean_rank_random": float(m["mean_rank_random"]),
                "mrr_random": float(m["mrr_random"]),
            }
            for k in CV_PREC_K:
                row[f"recall@{k}"] = float(m[f"recall@{k}"])
                row[f"precision@{k}"] = float(m[f"precision@{k}"])
            sweep_rows.append(row)

        print(f"Finished threshold {thr} in {time.time() - t0:.1f}s")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_OUTPUT_CSV, index=False)
    print(f"\nSaved sweep results: {SWEEP_OUTPUT_CSV}")

    # Show "best" settings by mean_rank (lower is better) and MRR (higher is better)
    if not sweep_df.empty:
        best_rank = sweep_df.sort_values("mean_rank", ascending=True).head(5)
        best_mrr = sweep_df.sort_values("mrr", ascending=False).head(5)

        print("\nTop 5 settings by lowest mean held-out rank:")
        print(best_rank[["threshold", "restart", "mean_rank", "mean_percentile", "mrr"]].to_string(index=False))

        print("\nTop 5 settings by highest MRR:")
        print(best_mrr[["threshold", "restart", "mean_rank", "mean_percentile", "mrr"]].to_string(index=False))

input("\nPress Enter to exit...")
