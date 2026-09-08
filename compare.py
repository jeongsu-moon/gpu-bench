#!/usr/bin/env python3
"""Diff two or more bench.py JSONs into a comparison table.

    python3 compare.py bench_3090x3.json bench_blackwell72.json

The last file listed is the reference (ratios are computed against it).
Reports three separate views -- per-GPU, whole-node, and VRAM -- because
collapsing them into one number is what makes these comparisons misleading.
"""
import json, sys

DTYPES = ("fp32", "tf32", "fp16", "bf16", "fp8_e4m3")


def load(path):
    with open(path) as f:
        return json.load(f)


def best(r, key, idx="0"):
    v = r.get("compute", {}).get(idx, {}).get(key, {})
    return v.get("best_tflops")


def node_sum(r, key):
    vals = [r["compute"][i].get(key, {}).get("best_tflops")
            for i in r.get("compute", {})]
    vals = [v for v in vals if v]
    return round(sum(vals), 1) if vals else None


def node_mem(r, key):
    vals = [r["memory"][i][key] for i in r.get("memory", {})]
    return round(sum(vals), 1) if vals else None


def cell(v, unit=""):
    return f"{v}{unit}" if v is not None else "n/a"


def ratio(v, ref):
    if v is None or ref in (None, 0):
        return "--"
    return f"{v / ref:.2f}x"


def row(label, vals, ref_i, unit=""):
    ref = vals[ref_i]
    cells = [f"{cell(v, unit):>14}" for v in vals]
    rats = [f"{ratio(v, ref):>12}" for i, v in enumerate(vals) if i != ref_i]
    return f"  {label:<16}" + "".join(cells) + "   " + "".join(rats)


def section(title):
    print(f"\n{title}\n" + "-" * (len(title)))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    paths = sys.argv[1:]
    rs = [load(p) for p in paths]
    tags = [r["tag"] for r in rs]
    ref_i = len(rs) - 1
    ref_tag = tags[ref_i]

    print("=" * 78)
    for r in rs:
        names = ", ".join(sorted({g["name"] for g in r["gpus"]}))
        cnt = len(r["gpus"])
        print(f"  {r['tag']:<14} {cnt}x {names}  |  torch {r['torch']} / cu{r['cuda']} "
              f"/ drv {r['driver']}")
    print(f"  ratios vs '{ref_tag}'")
    print("=" * 78)

    hdr = "  " + " " * 16 + "".join(f"{t:>14}" for t in tags) + "   " + \
          "".join(f"{t[:12]:>12}" for i, t in enumerate(tags) if i != ref_i)

    section("PER-GPU  (single device -- architecture comparison)")
    print(hdr)
    for d in DTYPES:
        print(row(d + " TFLOPS", [best(r, d) for r in rs], ref_i))
    for k, lab in (("copy_GBs", "mem copy"), ("read_GBs", "mem read"),
                   ("triad_GBs", "mem triad")):
        print(row(lab + " GB/s",
                  [r.get("memory", {}).get("0", {}).get(k) for r in rs], ref_i))

    section("WHOLE-NODE  (sum over GPUs -- data-parallel ceiling)")
    print(hdr)
    for d in ("bf16", "fp16"):
        print(row(d + " TFLOPS", [node_sum(r, d) for r in rs], ref_i))
    print(row("mem read GB/s", [node_mem(r, "read_GBs") for r in rs], ref_i))
    print(row("GPU count", [float(len(r["gpus"])) for r in rs], ref_i))

    section("VRAM  (largest single pool is what gates model size)")
    print(hdr)
    print(row("per GPU GiB", [r["gpus"][0]["vram_GiB"] for r in rs], ref_i))
    print(row("node total GiB",
              [round(sum(g["vram_GiB"] for g in r["gpus"]), 1) for r in rs], ref_i))

    if any("sustained" in r for r in rs):
        section("SUSTAINED LOAD  (throttling under concurrent all-GPU bf16)")
        for r in rs:
            s = r.get("sustained")
            if not s:
                print(f"  {r['tag']}: not measured")
                continue
            print(f"  {r['tag']}  ({s['seconds']}s, {s['concurrent_gpus']} GPU "
                  "concurrent)")
            for i, g in s["per_gpu"].items():
                t = s.get("telemetry", {}).get(i, {})
                extra = ""
                if t:
                    extra = (f"   {t['sm_mhz']['start']:.0f}->{t['sm_mhz']['end']:.0f} MHz"
                             f"  peak {t['watt_max']:.0f}W  {t['temp_c']['max']:.0f}C")
                print(f"    GPU{i}  {g['first']:>7} -> {g['last']:>7} TFLOPS  "
                      f"retention {g['retention_pct']:>5}%{extra}")

    if any("nccl_allreduce" in r for r in rs):
        section("INTERCONNECT  (all_reduce busbw -- only meaningful multi-GPU)")
        for r in rs:
            n = r.get("nccl_allreduce")
            if not n:
                print(f"  {r['tag']}: single GPU -- no interconnect cost at all")
                continue
            if "skipped" in n or "error" in n:
                print(f"  {r['tag']}: {n.get('skipped') or n.get('error')}")
                continue
            print(f"  {r['tag']}  " + "  ".join(
                f"{k}={v['busbw_GBs']}GB/s" for k, v in n.items()))
        for r in rs:
            if "p2p" in r:
                print(f"\n  {r['tag']} p2p copy GB/s (row=src, col=dst):")
                m = r["p2p"]["copy_GBs"]
                keys = sorted(m, key=int)
                print("      " + "".join(f"{('->' + k):>9}" for k in keys))
                for i in keys:
                    print(f"    {i} " + "".join(
                        f"{cell(m[i][j]):>9}" for j in keys))

    print("\n" + "=" * 78)
    print("  Read PER-GPU for architecture, WHOLE-NODE for total capacity, VRAM for")
    print("  what fits at all. A model that does not fit in one GPU's pool scores")
    print("  zero regardless of the TFLOPS above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
