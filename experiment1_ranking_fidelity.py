"""
Experiment 1: FP4 Reward Ranking Fidelity for LLM Text Generation

Research question: Does FP4 quantization preserve reward ranking compared to BF16?

Usage:
    python3 experiment1_ranking_fidelity.py \
        --data_path /path/to/gsm8k.parquet \
        --num_prompts 10 --num_completions 8
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


# ── Config ────────────────────────────────────────────────

PRECISIONS = {
    "bf16":  {"port": 8002, "label": "BF16", "color": "blue"},
    "fp8":   {"port": 8003, "label": "FP8",  "color": "green"},
    "nvfp4": {"port": 8004, "label": "FP4",  "color": "red"},
}

MATH_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


# ── Answer extraction ─────────────────────────────────────

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


# ── Generation ────────────────────────────────────────────

def generate_batch(client, model_name, prompts, n,
                   temperature, max_tokens):
    """Generate n completions per prompt."""
    futures = {}
    with ThreadPoolExecutor(max_workers=64) as ex:
        for i, prompt in enumerate(prompts):
            for j in range(n):
                def _gen(p=prompt, mn=model_name):
                    return client.completions.create(
                        model=mn, prompt=p,
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=0.95, seed=None,
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
                results[i][j] = f"<ERROR: {e}>"
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rps = done / elapsed
                eta = (total - done) / rps
                print(f"    [{done}/{total}]  {rps:.1f} req/s  ETA {eta:.0f}s")
    return results


# ── Reward computation ────────────────────────────────────

def compute_rewards(all_completions, references):
    rewards = []
    for i, completions in enumerate(all_completions):
        ref = references[i]
        rewards.append([check_correct(extract_answer(c), ref) for c in completions])
    return rewards


# ── Ranking fidelity ─────────────────────────────────────

def ranking_metrics(bf16_rewards, quant_rewards, label):
    ktaus, srhos = [], []
    top4_overlaps, top1_matches = [], []
    all_bf16, all_quant = [], []

    for b, q in zip(bf16_rewards, quant_rewards):
        all_bf16.extend(b)
        all_quant.extend(q)
        if len(b) < 3 or len(set(b)) < 2:
            continue
        tau, _ = kendalltau(b, q, variant='b')
        rho, _ = spearmanr(b, q)
        if not np.isnan(tau):
            ktaus.append(tau)
            srhos.append(rho)
        b_top4 = set(np.argsort(b)[-4:])
        q_top4 = set(np.argsort(q)[-4:])
        top4_overlaps.append(len(b_top4 & q_top4) / 4.0)
        top1_matches.append(1.0 if np.argmax(b) == np.argmax(q) else 0.0)

    global_tau, _ = kendalltau(all_bf16, all_quant, variant='b')
    global_rho, _ = spearmanr(all_bf16, all_quant)

    return {
        "label": label,
        "n_valid_prompts": len(ktaus),
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
    parser.add_argument("--num_completions", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--data_path", type=str, default=None)
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
    print(f"Prompts:     {args.num_prompts}")
    print(f"Completions: {args.num_completions} per prompt per precision")
    print(f"Total gens:  {args.num_prompts * args.num_completions * 3}")

    # ── Load dataset ──────────────────────────────────────
    print(f"\n── Loading data ──")
    if args.data_path and args.data_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(args.data_path)
        ds = df.to_dict("records")[:args.num_prompts]
    elif args.data_path:
        import pandas as pd
        df = pd.read_json(args.data_path, lines=True)
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
    print(f"  {len(prompts)} prompts loaded")

    # ── Generate & score per precision ────────────────────
    all_rewards = {}

    skip_map = {"bf16": "skip_bf16", "fp8": "skip_fp8", "nvfp4": "skip_fp4"}
    for key in ["bf16", "fp8", "nvfp4"]:
        if getattr(args, skip_map[key]):
            continue

        cfg = PRECISIONS[key]
        print(f"\n── {cfg['label']} (port {cfg['port']}) ──")

        client = OpenAI(base_url=f"http://localhost:{cfg['port']}/v1", api_key="no")
        model_name = client.models.list().data[0].id
        print(f"  Model: {model_name}")

        t0 = time.time()
        completions = generate_batch(
            client, model_name, prompts, args.num_completions,
            args.temperature, args.max_tokens,
        )
        gen_time = time.time() - t0
        print(f"  Generation: {gen_time:.0f}s")

        rewards = compute_rewards(completions, refs)
        flat = [r for rl in rewards for r in rl]
        pass1 = np.mean([max(rl) for rl in rewards])
        print(f"  Mean reward: {np.mean(flat):.3f}  pass@1: {pass1:.3f}")

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
        print(f"  Per-prompt τ (mean):  {m['per_prompt']['kendall_tau_mean']:.4f} ± {m['per_prompt']['kendall_tau_std']:.4f}")
        print(f"  Top-4 overlap:        {m['per_prompt']['top4_overlap_mean']:.3f} ± {m['per_prompt']['top4_overlap_std']:.3f}")
        print(f"  Top-1 match rate:     {m['per_prompt']['top1_match_rate']:.3f}")
        print(f"  Valid prompts:        {m['n_valid_prompts']}")

        tau = m['global']['kendall_tau']
        if tau > 0.70:
            v = "STRONG — FP4 Explore / BF16 Train likely viable"
        elif tau > 0.45:
            v = "MODERATE — hybrid filtering needed"
        elif tau > 0.25:
            v = "WEAK — significant ranking distortion"
        else:
            v = "NEAR-RANDOM — FP4 not viable for autoregressive text ranking"
        print(f"  → {v}")

    # ── Throughput estimate ───────────────────────────────
    print(f"\n── Throughput Estimate ──")
    for key in ["bf16", "fp8", "nvfp4"]:
        if key not in all_rewards:
            continue
        client = OpenAI(base_url=f"http://localhost:{PRECISIONS[key]['port']}/v1", api_key="no")
        mn = client.models.list().data[0].id
        t0 = time.time()
        client.completions.create(model=mn, prompt=prompts[0],
                                   temperature=0.0, max_tokens=256)
        print(f"  {PRECISIONS[key]['label']}: {time.time() - t0:.2f}s / 256 tokens")

    # ── Save ──────────────────────────────────────────────
    out = {
        "config": {
            "num_prompts": len(prompts),
            "num_completions": args.num_completions,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "generation_quality": {},
        "ranking_fidelity": {},
    }
    for k, rewards in all_rewards.items():
        flat = [r for rl in rewards for r in rl]
        out["generation_quality"][k] = {
            "mean_reward": float(np.mean(flat)),
            "pass_at_1": float(np.mean([max(rl) for rl in rewards])),
            "n_completions": len(flat),
        }
    for k, m in results.items():
        out["ranking_fidelity"][k] = m

    fp = os.path.join(args.output_dir,
                      f"exp1_fidelity_{args.num_prompts}p_{args.num_completions}c_{ts}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {fp}")
    print("Done.")


if __name__ == "__main__":
    main()
