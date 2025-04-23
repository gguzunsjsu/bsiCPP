#!/usr/bin/env python3
"""
gen_zipf_s3_multi.py

Generate Zipf(s=3) A,B pairs for multiple vector lengths:
  100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000

Each line in rows{n}_skew3_card16.txt is:
    A,B
with A,B ~ Zipf(s=3) truncated to [1..65536].
"""

import numpy as np

def generate_zipf_pairs(filename, exponent, n, max_rank=2**16, seed=None):
    rng = np.random.default_rng(seed)
    # generate A
    A = []
    while len(A) < n:
        # oversample by 20%
        to_sample = int((n - len(A)) * 1.2) + 1
        samp = rng.zipf(exponent, size=to_sample)
        # keep only valid ranks
        valid = samp[samp <= max_rank]
        A.extend(valid.tolist())
    A = A[:n]

    # generate B
    B = []
    while len(B) < n:
        to_sample = int((n - len(B)) * 1.2) + 1
        samp = rng.zipf(exponent, size=to_sample)
        valid = samp[samp <= max_rank]
        B.extend(valid.tolist())
    B = B[:n]

    # write file
    with open(filename, "w") as f:
        for a, b in zip(A, B):
            f.write(f"{a},{b}\n")
    print(f"Wrote {filename} ({n} pairs)")

if __name__ == "__main__":
    EXPONENT = 3.0
    SIZES    = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    for n in SIZES:
        fname = f"rows{n}_skew3_card16.txt"
        generate_zipf_pairs(
            filename=fname,
            exponent=EXPONENT,
            n=n,
            max_rank=2**16,
            seed=12345  # for reproducibility
        )
