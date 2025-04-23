#!/usr/bin/env python3
"""
gen_zipf_data_multi.py

Generate Zipf-distributed A,B pairs for lengths 10K and 1M,
with parameters s=2.5 and s=3.0. Outputs four files:
  rows{n}_skew{suffix}_card16.txt
"""

import numpy as np

def generate_zipf_pairs(filename, exponent, n, max_rank=2**16, seed=12345):
    """
    Generate `n` Zipf draws in [1..max_rank] for two vectors A and B,
    and write them as comma-separated pairs to `filename`.
    """
    rng = np.random.default_rng(seed)
    A, B = [], []
    # oversample and filter until we have exactly n valid entries
    while len(A) < n or len(B) < n:
        needA = n - len(A)
        needB = n - len(B)
        sampA = rng.zipf(exponent, size=needA * 2)
        sampB = rng.zipf(exponent, size=needB * 2)
        # keep only those <= max_rank
        A.extend(x for x in sampA if x <= max_rank)
        A = A[:n]
        B.extend(x for x in sampB if x <= max_rank)
        B = B[:n]

    # write out
    with open(filename, "w") as f:
        for a, b in zip(A, B):
            f.write(f"{a},{b}\n")
    print(f"Wrote {filename}")

if __name__ == "__main__":
    exponents = [2.5, 3.0]
    lengths   = [10_000, 1_000_000]

    for s in exponents:
        # build suffix: "25" for 2.5, "3" for 3.0
        suffix = str(int(s)) if s.is_integer() else str(s).replace('.', '')
        for n in lengths:
            fname = f"rows{n}_skew{suffix}_card16.txt"
            generate_zipf_pairs(fname, exponent=s, n=n)
