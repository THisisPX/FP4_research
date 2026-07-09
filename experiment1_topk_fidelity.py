"""
Experiment 1+: Precision-aware Top-K Filtering Fidelity

Revised experiment that directly measures what matters for "FP4 Explore / BF16 Train":
Given N completions per prompt across BF16 vs FP4, how many of the top-K candidates
(by reward) are shared between the two precisions?

Key metrics:
  - Precision@K: Among BF16's top-K, how many are also in FP4's top-K?
  - Recall@K: Among FP4's top-K, how many are also in BF16's top-K?
  - Best-vs-rest: Is FP4's best completion also BF16's best?

Usage:
    python3 experiment1_topk_fidelity.py \
        --data_path /path/to/gsm8k.parquet \
        --num_prompts 100 --num_completions 32
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from openai import OpenAI


PRECISIONS = {
    "bf16":  {"port": 8002, "label": "BF16"},
    "fp8":   {"port": 8003, "label": "FP8"},
    "nvfp4": {"port": 8004, "label": "FP4"},
}

MATH_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def extract_answer(text: str) -> str | None:
    m = re.findall(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m[-1].strip().replace(",", "").replace(" ", "")
    nums = re.findall(r'(?<!\d)-?\d+\.?\d*(?!\d)', text)
    return nums[-1] if nums else None


def check_correct(pred: str | None, ref: str) -> float:
    if pred is None:
        return 0.0
    ref_n = ref.strip().replace(",", "").replace(" ", "").replace("$", "")
    if pred.strip() == ref_n:
        return 1.0
    try:
        if abs(float(pred.strip()) - float(ref_n)) < 1e-5:
            return 1.0
    except (ValueError, TypeError):
        pass
    return 0.0


def generate_batch(client, model_name, prompts, n, temperature, max_tokens):
    """Concurrent generation for all (prompt × n) completions."""
    futures = {}
    with ThreadPoolExecutor(max_workers=64) as ex:
        for i, prompt in enumerate(prompts):
            for j in range(n):
                def _gen(p=prompt, mn=model_name):
                    return client.completions.create(
                        model=mn, prompt=p,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=0.95,
                    ).choices[0].text
                futures[ex.submit(_gen)] = (i, j)

        results = [[""] * n for _ in range(len(prompts))]
        done = 0
        total = len(futures)
        t0 = time.time()
        for fut in as_completed(futures):
            i, j = futures[fut]
            try:
                results[i][j] = fut.result()
            except Exception as e:
                results[i][j] = f"<ERROR>"
            done += 1
            if done % 200 == 0:
                e = time.time() - t0
                print(f"    [{done}/{total}]  {done/e:.0f} req/s  ETA {(total-done)/(done/e):.0f}s")
    return results


def topk_filtering_metrics(bf16_completions, fp4_completions, references, ks=[1, 2, 4, 8]):
    """
    For each prompt, extract completions, compute rewards, and measure
    how well FP4's top-K overlap with BF16's top-K.
    """
    results = {k: {"precision": [], "recall": [], "best_match": []} for k in ks}
    all_bf16_rewards = []
    all_fp4_rewards = []

    for i, (bf16_texts, fp4_texts) in enumerate(zip(bf16_completions, fp4_completions)):
        ref = references[i]

        # Extract answers and compute rewards
        bf16_answers = [extract_answer(t) for t in bf16_texts]
        fp4_answers = [extract_answer(t) for t in fp4_texts]
        bf16_r = np.array([check_correct(a, ref) for a in bf16_answers])
        fp4_r = np.array([check_correct(a, ref) for a in fp4_answers])

        all_bf16_rewards.extend(bf16_r)
        all_fp4_rewards.extend(fp4_r)

        N = len(bf16_r)

        # For each K, compute overlap
        for K in ks:
            K_eff = min(K, N)
            if K_eff == 0:
                continue

            bf16_topk = set(np.argsort(bf16_r)[-K_eff:])
            fp4_topk = set(np.argsort(fp4_r)[-K_eff:])

            # Precision@K: fraction of BF16 top-K that FP4 also puts in top-K
            precision = len(bf16_topk & fp4_topk) / K_eff
            # Recall@K: fraction of FP4 top-K that BF16 also puts in top-K
            recall = len(bf16_topk & fp4_topk) / K_eff

            results[K]["precision"].append(precision)
            results[K]["recall"].append(recall)

            # Best match (K=1)
            if K == 1:
                results[K]["best_match"].append(
                    1.0 if np.argmax(bf16_r) == np.argmax(fp4_r) else 0.0
                )

    return results, all_bf16_rewards, all_fp4_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_completions", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Experiment 1+: Top-K Filtering Fidelity")
    print("=" * 60)
    print(f"Prompts:     {args.num_prompts}")
    print(f"Completions: {args.num_completions} per prompt per precision")
    print(f"Temperature:  {args.temperature}")
    print(f"Total gens:   {args.num_prompts * args.num_completions * 2}")

    # ── Load data ─────────────────────────────────────────
    print(f"\n── Loading data ──")
    if args.data_path and args.data_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(args.data_path)
        ds = df.to_dict("records")[:args.num_prompts]
    else:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        ds = ds.select(range(min(len(ds), args.num_prompts)))

    prompts, refs = [], []
    for s in ds:
        q = s["question"]
        a = s["answer"]
        prompts.append(MATH_PROMPT.format(problem=q))
        m = re.search(r'####\s*(-?[\d,]+)', a)
        refs.append(m.group(1).replace(",", "") if m else a)
    print(f"  {len(prompts)} prompts")

    # ── Generate ──────────────────────────────────────────
    completions_store = {}

    for key in ["bf16", "nvfp4"]:
        cfg = PRECISIONS[key]
        print(f"\n── {cfg['label']} (port {cfg['port']}) ──")
        client = OpenAI(base_url=f"http://localhost:{cfg['port']}/v1", api_key="no")
        mn = client.models.list().data[0].id
        print(f"  Model: {mn}")

        t0 = time.time()
        comps = generate_batch(
            client, mn, prompts, args.num_completions,
            args.temperature, args.max_tokens,
        )
        elapsed = time.time() - t0
        ntokens_est = sum(len(c) for clist in comps for c in clist) / 4  # rough
        print(f"  {elapsed:.0f}s total (~{ntokens_est/elapsed:.0f} tok/s)")
        completions_store[key] = comps

    # ── Top-K filtering analysis ──────────────────────────
    print(f"\n{'=' * 60}")
    print("Top-K Filtering Fidelity: FP4 vs BF16")
    print(f"{'=' * 60}")

    K_VALUES = [1, 2, 4, 8, 16]
    metrics, bf16_all, fp4_all = topk_filtering_metrics(
        completions_store["bf16"],
        completions_store["nvfp4"],
        refs,
        ks=K_VALUES,
    )

    # Also FP8 for comparison
    metrics_fp8, fp8_all, _ = topk_filtering_metrics(
        completions_store["bf16"],
        completions_store.get("fp8", completions_store["bf16"]),
        refs,
        ks=K_VALUES,
    )

    print(f"\n{'K':<6} {'Prec@K':>10} {'Recall@K':>10} {'FP8 Prec@K':>12} {'FP8 Best':>10}")
    print("-" * 56)
    for K in K_VALUES:
        prec = np.mean(metrics[K]["precision"]) if metrics[K]["precision"] else 0
        rec = np.mean(metrics[K]["recall"]) if metrics[K]["recall"] else 0
        prec_fp8 = np.mean(metrics_fp8[K]["precision"]) if metrics_fp8[K]["precision"] else 0
        best_fp8 = np.mean(metrics_fp8[K].get("best_match", [0])) if K == 1 else 0
        best = np.mean(metrics[K].get("best_match", [0])) if K == 1 else 0
        print(f"{K:<6} {prec:>10.3f} {rec:>10.3f} {prec_fp8:>12.3f} "
              f"{'@1=' + str(best_fp8)[:5] if K == 1 else '':>10}")

    # ── Generation quality ────────────────────────────────
    print(f"\n── Generation Quality ──")
    print(f"  BF16:  mean={np.mean(bf16_all):.3f}  pass@1={np.mean([1 if r>0 else 0 for r in bf16_all]):.3f}")
    print(f"  FP4:   mean={np.mean(fp4_all):.3f}  pass@1={np.mean([1 if r>0 else 0 for r in fp4_all]):.3f}")

    # ── Key finding ──────────────────────────────────────
    k8_prec = np.mean(metrics[8]["precision"]) if metrics[8]["precision"] else 0
    k1_best = np.mean(metrics[1].get("best_match", [0])) if metrics[1].get("best_match", [0]) else 0

    print(f"\n── Key Findings ──")
    print(f"  Top-8 overlap:   {k8_prec:.3f}")
    print(f"  Best match rate: {k1_best:.3f}")
    if k8_prec > 0.70:
        print(f"  → FP4 is a RELIABLE filter for top candidates")
        print(f"  → FP4 Explore / BF16 Train is VIABLE")
    elif k8_prec > 0.50:
        print(f"  → FP4 is a MODERATE filter — usable for coarse filtering")
        print(f"  → Consider FP4 coarse + FP8 fine two-stage filtering")
    else:
        print(f"  → FP4 top-K overlap is LOW")
        print(f"  → Need to investigate autoregressive error propagation")

    # ── Save ──────────────────────────────────────────────
    output = {
        "config": {
            "num_prompts": len(prompts),
            "num_completions": args.num_completions,
            "temperature": args.temperature,
        },
        "topk_fidelity": {
            K: {
                "precision_mean": float(np.mean(metrics[K]["precision"])),
                "precision_std": float(np.std(metrics[K]["precision"])),
                "recall_mean": float(np.mean(metrics[K]["recall"])),
                "recall_std": float(np.std(metrics[K]["recall"])),
                "n_valid": len(metrics[K]["precision"]),
                "best_match_rate": float(np.mean(metrics[K].get("best_match", [0]))) if K == 1 else None,
            }
            for K in K_VALUES
        },
        "generation_quality": {
            "bf16": {"mean_reward": float(np.mean(bf16_all))},
            "fp4": {"mean_reward": float(np.mean(fp4_all))},
        },
    }
    fp = os.path.join(args.output_dir,
                      f"exp1_topk_{args.num_prompts}p_{args.num_completions}c_{ts}.json")
    with open(fp, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {fp}")


if __name__ == "__main__":
    main()
