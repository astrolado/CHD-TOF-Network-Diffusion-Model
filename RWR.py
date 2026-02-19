import os
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

SCORE_THRESHOLD = 700
RWR_RESTART = 0.4
RWR_ITERS = 50

TOP_NONSEED_PRINT = 30
TOP_NONSEED_SAVE = 200
DEDUP_BY_GENE_SYMBOL = True  # set False if you want to keep isoform duplicates

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

        # Normalize protein id
        if prot_id.startswith("9606."):
            prot_id = prot_id[5:]

        # Keep only canonical ENSP IDs
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
# 3) Stream-parse STRING links and build sparse adjacency (ENSP-only)
# ====================
print("Building sparse ENSP-only network from STRING links (streaming)...")
file_size = os.path.getsize(LINKS_TXT)

node_index = {}
rows = []
cols = []

def get_idx(pid: str) -> int:
    if pid not in node_index:
        node_index[pid] = len(node_index)
    return node_index[pid]

bad_lines = 0
kept_edges = 0
seen_header = False

with open(LINKS_TXT, "r", encoding="utf-8", errors="ignore") as f, tqdm(
    total=file_size, unit="B", unit_scale=True, desc="Reading links"
) as pbar:
    for line in f:
        pbar.update(len(line.encode("utf-8", errors="ignore")))

        line = line.strip()
        if not line:
            continue

        # Skip header line if present
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
        score_str = parts[-1]  # combined_score is last column

        # Normalize IDs
        if p1.startswith("9606."):
            p1 = p1[5:]
        if p2.startswith("9606."):
            p2 = p2[5:]

        # ENSP-only filter
        if not (p1.startswith("ENSP") and p2.startswith("ENSP")):
            continue

        # Parse score safely
        try:
            score = int(float(score_str))
        except Exception:
            bad_lines += 1
            continue

        if score <= SCORE_THRESHOLD:
            continue

        i = get_idx(p1)
        j = get_idx(p2)

        # undirected -> store both directions
        rows.append(i); cols.append(j)
        rows.append(j); cols.append(i)
        kept_edges += 2

print("Done building edge list.")
print("Bad/ignored lines:", bad_lines)
print("Nodes (ENSP):", len(node_index))
print("Edges stored (directed entries):", kept_edges)

if len(node_index) == 0:
    raise RuntimeError("0 ENSP nodes from links file; protein namespace mismatch.")

# ====================
# 4) Build sparse matrices
# ====================
n = len(node_index)
data = np.ones(len(rows), dtype=np.float32)
A = sparse.csr_matrix((data, (np.array(rows), np.array(cols))), shape=(n, n))
A.sum_duplicates()

# Column-normalize: W = A * D^{-1}
colsum = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)
colsum[colsum == 0] = 1.0
Dinv = sparse.diags(1.0 / colsum)
W = A @ Dinv  # sparse

# ====================
# 5) Seed vector p0
# ====================
p0 = np.zeros(n, dtype=np.float64)
present_seed_idxs = []

for pid in seed_proteins:
    if pid in node_index:
        present_seed_idxs.append(node_index[pid])

print("Seed proteins present in network:", len(present_seed_idxs))

if len(present_seed_idxs) == 0:
    raise RuntimeError("No seed proteins overlap with ENSP network from links.")

p0[present_seed_idxs] = 1.0
p0 /= p0.sum()

# ====================
# 6) Run sparse RWR
# ====================
print("Running sparse RWR...")
r = RWR_RESTART
p = p0.copy()

for _ in trange(RWR_ITERS, desc="RWR iterations"):
    p = (1 - r) * (W @ p) + r * p0

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

# --- Top non-seed candidates ---
novel = results[results["is_seed_gene"] == False].copy()

if DEDUP_BY_GENE_SYMBOL:
    novel = novel.drop_duplicates(subset=["gene_symbol"])

print(f"\nTop {TOP_NONSEED_PRINT} NON-SEED candidates:")
print(novel.head(TOP_NONSEED_PRINT)[["gene_symbol", "protein_id", "RWR_score"]].to_string(index=False))

# Save full + top novel files
results.to_csv("rwr_results.csv", index=False)
novel.head(TOP_NONSEED_SAVE).to_csv("rwr_novel_candidates_top200.csv", index=False)

print("\nSaved: rwr_results.csv")
print("Saved: rwr_novel_candidates_top200.csv")

print("\nTop results (including seeds):")
print(results.head(20))

input("Press Enter to exit...")
