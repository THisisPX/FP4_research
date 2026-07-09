"""
Experiment 1: FP4 Reward Ranking Fidelity for LLM Text Generation

Research question: Does FP4 quantization preserve reward ranking compared to BF16?

Three endpoints (B300):
    BF16  → http://localhost:8000
    FP8   → http://localhost:8001
    NVFP4 → http://localhost:8002

Key metric: Kendall's τ between BF16 ranking and FP4/FP8 ranking.

Usage:
    pip install openai datasets scipy
    python3 experiment1_ranking_fidelity.py --num_prompts 100 --num_completions 16
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
from scipy.stats import kendalltau, spearmanr
from datasets import load_dataset


# ── Config ────────────────────────────────────────────────

PRECISIONS = {
    "bf16":  {"port": 8000, "label": "BF16", "color": "blue"},
    "fp8":   {"port": 8001, "label": "FP8",  "color": "green"},
    "nvfp4": {"port": 8002, "label": "FP4",  "color": "red"},
}

MATH_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


# ── Answer extraction ─────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract boxed answer or last number from generated text."""
    # \\boxed{...}
    m = re.findall(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m[-1].strip().replace(",", "").replace(" ", "")

    # Fallback: last standalone number
    nums = re.findall(r'(?<!\d)-?\d+\.?\d*(?!\d)', text)
    if nums:
        return nums[-1]

    return None


def check_correct(pred: str | None, ref: str) -> float:
    """Binary reward: 1.0 if answer matches reference."""
    if pred is None:
        return 0.0

    ref_n = ref.strip().replace(",", "").replace(" ", "").replace("$", "")
    pred_n = pred.strip()

    # Exact match
    if pred_n == ref_n:
        return 1.0

    # Numeric match within tolerance
    try:
        p = float(pred_n)
        r = float(ref_n)
        if abs(p - r) < 1e-5:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


# ── Generation ────────────────────────────────────────────

def generate_batch(client: OpenAI, prompts: list[str], n: int,
                   temperature: float, max_tokens: int) -> list[list[str]]:
    """Generate n completions for each prompt.
    Returns: list of lists (outer=prompts, inner=n completions).
    """

    # Send all (prompt × n) requests concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=64) as ex:
        for i, prompt in enumerate(prompts):
            for j in range(n):
                fut = ex.submit(
                    lambda p=prompt: client.completions.create(
                        model="default",
                        prompt=p,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.95,
                        seed=None,
                    ).choices[0].text
                )
                futures[fut] = (i, j)

        # Collect results: init as empty
        results = [[""] * n for _ in range(len(prompts))]
        done = 0
        total = len(futures)
        t0 = time.time()
        for fut in as_completed(futures):
            i, j = futures[fut]
            try:
                results[i][j] = fut.result()
            except Exception as e:
                results[i][j] = f"<ERROR: {e}>"
            done += 1
            if done % 100 == 0:
                e = time.time() - t0
                rps = done / e
                eta = (total - done) / rps
                print(f"    [{done}/{total}]  {rps:.1f} req/s  ETA {eta:.0f}s")

    return results


# ── Reward computation ────────────────────────────────────

def compute_rewards(all_completions: list[list[str]],
                    references: list[str]) -> list[list[float]]:
    """Compute binary reward for each completion."""
    rewards = []
    for i, completions in enumerate(all_completions):
        ref = references[i]
        r = [check_correct(extract_answer(c), ref) for c in completions]
        rewards.append(r)
    return rewards


# ── Ranking fidelity ─────────────────────────────────────

def ranking_metrics(bf16_rewards: list[list[float]],
                    quant_rewards: list[list[float]],
                    label: str) -> dict:
    """Compute Kendall's τ, Spearman's ρ, Top-K overlap, Top-1 match."""
    n_ok = 0
    ktaus = []
    srhos = []
    top4_overlaps = []
    top1_matches = []

    all_bf16 = []
    all_quant = []

    for b, q in zip(bf16_rewards, quant_rewards):
        all_bf16.extend(b)
        all_quant.extend(q)

        # Need at least 3 values with some variance for ranking
        if len(b) < 3 or len(set(b)) < 2:
            continue

        tau, _ = kendalltau(b, q, variant='b')
        rho, _ = spearmanr(b, q)
        if not np.isnan(tau):
            ktaus.append(tau)
            srhos.append(rho)

        # Top-4 overlap
        b_top4 = set(np.argsort(b)[-4:])
        q_top4 = set(np.argsort(q)[-4:])
        top4_overlaps.append(len(b_top4 & q_top4) / 4.0)

        # Top-1 match
        top1_matches.append(1.0 if np.argmax(b) == np.argmax(q) else 0.0)
        n_ok += 1

    global_tau, _ = kendalltau(all_bf16, all_quant, variant='b')
    global_rho, _ = spearmanr(all_bf16, all_quant)

    return {
        "label": label,
        "n_valid_prompts": n_ok,
        "n_total_pairs": len(all_bf16),
        "global": {
            "kendall_tau": float(global_tau) if not np.isnan(global_tau) else 0.0,
            "spearman_rho": float(global_rho) if not np.isnan(global_rho) else 0.0,
        },
        "per_prompt": {
            "kendall_tau_mean": float(np.mean(ktaus)),
            "kendall_tau_std": float(np.std(ktaus)),
            "spearman_rho_mean": float(np.mean(srhos)),
            "spearman_rho_std": float(np.std(srhos)),
            "top4_overlap_mean": float(np.mean(top4_overlaps)),
            "top4_overlap_std": float(np.std(top4_overlaps)),
            "top1_match_rate": float(np.mean(top1_matches)),
        },
    }


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_completions", type=int, default=16, help="G in GRPO")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--skip_bf16", action="store_true")
    parser.add_argument("--skip_fp8", action="store_true")
    parser.add_argument("--skip_fp4", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Experiment 1: FP4 Reward Ranking Fidelity")
    print("=" * 60)
    print(f"Benchmark:   {args.benchmark}")
    print(f"Prompts:     {args.num_prompts}")
    print(f"Completions: {args.num_completions} per prompt per precision")
    print(f"Total gens:  {args.num_prompts * args.num_completions * 3}")

    # ── Load dataset ──────────────────────────────────────
    print(f"\n── Loading {args.benchmark} ──")
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(len(ds), args.num_prompts)))

    prompts = [MATH_PROMPT.format(problem=s["question"]) for s in ds]
    # GSM8K answers are in "answer" field like "#### 42"
    refs = []
    for s in ds:
        a = s["answer"]
        m = re.search(r'####\s*(-?[\d,]+)', a)
        refs.append(m.group(1).replace(",", "") if m else a)
    print(f"  {len(prompts)} prompts loaded")

    # ── Generate & score per precision ────────────────────
    all_rewards = {}

    for key in ["bf16", "fp8", "nvfp4"]:
        if getattr(args, f"skip_{key}"):
            continue

        cfg = PRECISIONS[key]
        print(f"\n── {cfg['label']} (port {cfg['port']}) ──")

        client = OpenAI(
            base_url=f"http://localhost:{cfg['port']}/v1",
            api_key="not-needed",
        )

        t0 = time.time()
        completions = generate_batch(
            client, prompts, args.num_completions,
            args.temperature, args.max_tokens,
        )
        gen_time = time.time() - t0
        print(f"  Generation: {gen_time:.0f}s")

        rewards = compute_rewards(completions, refs)
        # Stats
        all_r = [r for rlist in rewards for r in rlist]
        mean_r = np.mean(all_r)
        pass1 = np.mean([max(rlist) for rlist in rewards])
        print(f"  Mean reward: {mean_r:.3f}  pass@1: {pass1:.3f}")

        all_rewards[key] = rewards

    # ── Ranking fidelity ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Ranking Fidelity vs BF16")
    print(f"{'=' * 60}")

    results = {}

    for key in ["fp8", "nvfp4"]:
        if key not in all_rewards or "bf16" not in all_rewards:
            continue

        m = ranking_metrics(all_rewards["bf16"], all_rewards[key], key.upper())
        results[key] = m

        print(f"\n── {m['label']} vs BF16 ──")
        print(f"  Global Kendall τ:     {m['global']['kendall_tau']:.4f}")
        print(f"  Global Spearman ρ:    {m['global']['spearman_rho']:.4f}")
        print(f"  Per-prompt τ (mean):  {m['per_prompt']['kendall_tau_mean']:.4f} "
              f"± {m['per_prompt']['kendall_tau_std']:.4f}")
        print(f"  Top-4 overlap:        {m['per_prompt']['top4_overlap_mean']:.3f} "
              f"± {m['per_prompt']['top4_overlap_std']:.3f}")
        print(f"  Top-1 match rate:     {m['per_prompt']['top1_match_rate']:.3f}")
        print(f"  Valid prompts:        {m['n_valid_prompts']}")

        # Interpretation
        tau = m['global']['kendall_tau']
        if tau > 0.70:
            v = "STRONG — FP4 Explore / BF16 Train likely viable"
        elif tau > 0.45:
            v = "MODERATE — hybrid filtering (FP4 coarse + FP8 fine) needed"
        elif tau > 0.25:
            v = "WEAK — significant ranking distortion; investigate error propagation"
        else:
            v = "NEAR-RANDOM — FP4 not viable for ranking in autoregressive text"
        print(f"  → {v}")

    # ── Efficiency ────────────────────────────────────────
    # Quick benchmark: single-completion throughput
    print(f"\n── Throughput Estimate ──")
    for key in ["bf16", "fp8", "nvfp4"]:
        if key not in all_rewards:
            continue
        client = OpenAI(base_url=f"http://localhost:{PRECISIONS[key]['port']}/v1",
                        api_key="no")
        t0 = time.time()
        _ = client.completions.create(
            model="default",
            prompt=prompts[0],
            temperature=0.0,
            max_tokens=256,
        )
        one = time.time() - t0
        print(f"  {PRECISIONS[key]['label']}: single completion {one:.2f}s")

    # ── Save ──────────────────────────────────────────────
    out = {
        "config": {
            "benchmark": args.benchmark,
            "num_prompts": len(prompts),
            "num_completions": args.num_completions,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "generation_quality": {},
        "ranking_fidelity": {},
    }
    for key, rewards in all_rewards.items():
        flat = [r for rl in rewards for r in rl]
        out["generation_quality"][key] = {
            "mean_reward": float(np.mean(flat)),
            "pass_at_1": float(np.mean([max(rl) for rl in rewards])),
            "n_completions": len(flat),
        }
    for key, m in results.items():
        out["ranking_fidelity"][key] = m

    fp = os.path.join(args.output_dir,
                      f"exp1_fidelity_{args.benchmark}_{args.num_prompts}p_{args.num_completions}c_{ts}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nResults saved: {fp}")
    print("Done.")


if __name__ == "__main__":
    main()
