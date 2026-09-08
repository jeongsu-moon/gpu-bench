#!/usr/bin/env python3
"""Portable single-file GPU benchmark: compute ceiling, memory bandwidth,
interconnect, and sustained-load throttling behaviour.

No dependencies beyond PyTorch. Copy to each host, run, diff the JSONs.

    python3 bench.py --tag blackwell72
    python3 bench.py --tag 3090x3 --sustain 300

Subset GPUs with CUDA_VISIBLE_DEVICES (the NCCL stage uses every visible one).
"""
import argparse, json, os, platform, subprocess, sys, tempfile, threading, time, warnings

import torch

warnings.filterwarnings("ignore")

MATMUL_SIZES = (4096, 8192, 16384)
NCCL_SIZES_MIB = (8, 64, 512)


# ---------------------------------------------------------------- helpers

def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def timed(fn, dev, warmup=5, iters=20):
    """Mean seconds per call, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(dev)
    return start.elapsed_time(end) / 1e3 / iters


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


# ------------------------------------------------------- layer 1: compute

def make_operands(dev, dtype, n):
    """Returns (a, b, closure) or raises."""
    if dtype == "fp8":
        f8 = torch.float8_e4m3fn
        a = torch.randn(n, n, device=dev).to(f8)
        b = torch.randn(n, n, device=dev).to(f8).t()  # _scaled_mm wants col-major
        s = torch.tensor(1.0, device=dev)
        return a, b, lambda: torch._scaled_mm(a, b, scale_a=s, scale_b=s,
                                              out_dtype=torch.bfloat16)
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    return a, b, lambda: a @ b


def matmul_tflops(dev, dtype, n, tf32=False):
    set_tf32(tf32)
    try:
        a, b, fn = make_operands(dev, dtype, n)
        fn()  # shake out unsupported-kernel errors before timing
    except Exception as e:
        torch.cuda.empty_cache()
        return {"error": type(e).__name__ + ": " + str(e).split("\n")[0][:110]}
    dt = timed(fn, dev)
    del a, b
    torch.cuda.empty_cache()
    return {"ms": round(dt * 1e3, 3), "tflops": round(2 * n ** 3 / dt / 1e12, 1)}


def compute_sweep(dev, sizes):
    dtypes = [("fp32", torch.float32, False), ("tf32", torch.float32, True),
              ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False),
              ("fp8_e4m3", "fp8", False)]
    out = {}
    for label, dt, tf32 in dtypes:
        per_size = {}
        for n in sizes:
            per_size[str(n)] = matmul_tflops(dev, dt, n, tf32)
        best = max((v["tflops"] for v in per_size.values() if "tflops" in v),
                   default=None)
        out[label] = {"best_tflops": best, "by_size": per_size}
    set_tf32(False)
    return out


# ---------------------------------------------------- layer 2: memory bw

def mem_bandwidth(dev, gib=4):
    n = int(gib * (1 << 30) // 4)
    a = torch.empty(n, device=dev, dtype=torch.float32).uniform_()
    b = torch.empty_like(a)
    c = torch.empty_like(a)
    nb = n * 4

    copy = timed(lambda: b.copy_(a), dev, 3, 10)          # 1 read + 1 write
    read = timed(lambda: torch.sum(a), dev, 3, 10)        # 1 read
    triad = timed(lambda: torch.add(b, c, alpha=2.0, out=a), dev, 3, 10)  # 2r + 1w

    res = {"copy_GBs": round(2 * nb / copy / 1e9, 1),
           "read_GBs": round(nb / read / 1e9, 1),
           "triad_GBs": round(3 * nb / triad / 1e9, 1)}
    del a, b, c
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------- layer 3: interconnect

def p2p_matrix(idxs, mib=256):
    """Device-to-device copy bandwidth, GB/s. Diagonal = intra-device."""
    n = mib * (1 << 20) // 4
    mat, access = {}, {}
    for i in idxs:
        row, arow = {}, {}
        src = torch.empty(n, device=f"cuda:{i}", dtype=torch.float32).uniform_()
        for j in idxs:
            try:
                arow[str(j)] = (i == j) or torch.cuda.can_device_access_peer(i, j)
                dst = torch.empty(n, device=f"cuda:{j}", dtype=torch.float32)
                dt = timed(lambda: dst.copy_(src), torch.device(f"cuda:{i}"), 3, 10)
                row[str(j)] = round(n * 4 / dt / 1e9, 1)
                del dst
            except Exception as e:
                row[str(j)] = None
                arow[str(j)] = str(e)[:60]
            torch.cuda.empty_cache()
        mat[str(i)], access[str(i)] = row, arow
        del src
        torch.cuda.empty_cache()
    return {"copy_GBs": mat, "peer_access": access}


def _nccl_worker(rank, world, sizes, outpath):
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device(f"cuda:{rank}")
    res = {}
    for mib in sizes:
        t = torch.ones(mib * (1 << 20) // 2, device=dev, dtype=torch.bfloat16)
        for _ in range(5):
            dist.all_reduce(t)
        torch.cuda.synchronize(dev)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(t)
        torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t0) / 20
        algbw = t.numel() * 2 / dt / 1e9
        res[f"{mib}MiB"] = {
            "ms": round(dt * 1e3, 3),
            "algbw_GBs": round(algbw, 1),
            "busbw_GBs": round(algbw * 2 * (world - 1) / world, 1),
        }
        del t
        torch.cuda.empty_cache()
    if rank == 0:
        with open(outpath, "w") as f:
            json.dump(res, f)
    dist.destroy_process_group()


def nccl_allreduce(world, sizes):
    if world < 2:
        return {"skipped": "single GPU"}
    import torch.multiprocessing as mp
    path = os.path.join(tempfile.gettempdir(), f"nccl_{os.getpid()}.json")
    try:
        mp.spawn(_nccl_worker, args=(world, sizes, path), nprocs=world, join=True)
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e).split("\n")[0][:200]}
    finally:
        if os.path.exists(path):
            os.remove(path)


# --------------------------------------------- layer 4: sustained / thermal

class SmiSampler(threading.Thread):
    """Polls clocks/power/temp so throttling is visible after the fact."""
    Q = ("index,clocks.sm,power.draw,temperature.gpu,utilization.gpu")

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.period, self.samples, self._done = period, [], threading.Event()

    def run(self):
        while not self._done.is_set():
            out = sh(f"nvidia-smi --query-gpu={self.Q} "
                     "--format=csv,noheader,nounits", timeout=10)
            ts = round(time.time(), 1)
            for line in out.splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 5:
                    try:
                        self.samples.append({"t": ts, "gpu": int(p[0]),
                                             "sm_mhz": float(p[1]), "watt": float(p[2]),
                                             "temp_c": float(p[3]), "util": float(p[4])})
                    except ValueError:
                        pass
            self._done.wait(self.period)

    def stop(self):
        self._done.set()
        self.join(timeout=10)

    def per_gpu(self):
        out = {}
        for s in self.samples:
            out.setdefault(str(s["gpu"]), []).append(s)
        summary = {}
        for g, ss in out.items():
            summary[g] = {
                "n": len(ss),
                "sm_mhz": {"start": ss[0]["sm_mhz"], "end": ss[-1]["sm_mhz"],
                           "min": min(x["sm_mhz"] for x in ss)},
                "watt_max": max(x["watt"] for x in ss),
                "temp_c": {"start": ss[0]["temp_c"], "max": max(x["temp_c"] for x in ss)},
            }
        return summary


def _sustain_one(idx, seconds, n, dtype, window, results):
    dev = torch.device(f"cuda:{idx}")
    torch.cuda.set_device(dev)
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    flop = 2 * n ** 3
    wins, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        iters, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < window:
            a @ b
            iters += 1
        torch.cuda.synchronize(dev)
        dt = time.perf_counter() - t0
        wins.append(round(flop * iters / dt / 1e12, 1))
    del a, b
    torch.cuda.empty_cache()
    results[idx] = wins


def sustained(idxs, seconds, n=8192, window=10.0):
    """All GPUs loaded concurrently -- that is the realistic thermal case."""
    results = {}
    smi = SmiSampler()
    smi.start()
    threads = [threading.Thread(target=_sustain_one,
                                args=(i, seconds, n, torch.bfloat16, window, results))
               for i in idxs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    smi.stop()
    per_gpu = {}
    for i, wins in sorted(results.items()):
        if not wins:
            continue
        per_gpu[str(i)] = {
            "windows_tflops": wins,
            "first": wins[0], "last": wins[-1], "min": min(wins),
            "retention_pct": round(100 * wins[-1] / wins[0], 1),
        }
    return {"seconds": seconds, "window_s": window, "dtype": "bf16", "n": n,
            "concurrent_gpus": len(idxs), "per_gpu": per_gpu,
            "telemetry": smi.per_gpu()}


# ---------------------------------------------------------------- driver

def gpu_meta(i):
    p = torch.cuda.get_device_properties(i)
    return {"index": i, "name": p.name, "sm": f"{p.major}.{p.minor}",
            "sms": p.multi_processor_count,
            "vram_GiB": round(p.total_memory / (1 << 30), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="host label, e.g. blackwell72")
    ap.add_argument("--sustain", type=int, default=300,
                    help="sustained-load seconds per GPU, 0 to skip (default 300)")
    ap.add_argument("--skip-nccl", action="store_true")
    ap.add_argument("--skip-p2p", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="single matmul size, no sustained load")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible")

    idxs = list(range(torch.cuda.device_count()))
    sizes = (8192,) if args.quick else MATMUL_SIZES
    sustain_s = 0 if args.quick else args.sustain

    res = {
        "tag": args.tag,
        "host": platform.node(),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader "
                     "| head -1"),
        "cpu": sh("lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs"),
        "gpus": [gpu_meta(i) for i in idxs],
        "power_limit_W": sh("nvidia-smi --query-gpu=power.limit "
                            "--format=csv,noheader,nounits").splitlines(),
        "topo": sh("nvidia-smi topo -m"),
        "compute": {}, "memory": {},
    }

    for i in idxs:
        dev = torch.device(f"cuda:{i}")
        torch.cuda.set_device(dev)
        print(f"[{args.tag}] GPU{i} {res['gpus'][i]['name']} -- compute sweep",
              flush=True)
        res["compute"][str(i)] = compute_sweep(dev, sizes)
        print(f"[{args.tag}] GPU{i} -- memory bandwidth", flush=True)
        res["memory"][str(i)] = mem_bandwidth(dev)

    if len(idxs) > 1 and not args.skip_p2p:
        print(f"[{args.tag}] p2p matrix", flush=True)
        res["p2p"] = p2p_matrix(idxs)
    if len(idxs) > 1 and not args.skip_nccl:
        print(f"[{args.tag}] nccl all_reduce (world={len(idxs)})", flush=True)
        res["nccl_allreduce"] = nccl_allreduce(len(idxs), NCCL_SIZES_MIB)

    if sustain_s > 0:
        print(f"[{args.tag}] sustained load {sustain_s}s on {len(idxs)} GPU(s) "
              "-- this is the slow part", flush=True)
        res["sustained"] = sustained(idxs, sustain_s)

    path = args.out or f"bench_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=2)

    # ---- console summary
    print(f"\n=== {args.tag} ({res['host']}) torch {res['torch']} / cu{res['cuda']} ===")
    for g in res["gpus"]:
        i = str(g["index"])
        print(f"\nGPU{i} {g['name']}  {g['vram_GiB']}GiB  SM{g['sm']}  {g['sms']} SMs")
        m = res["memory"][i]
        print(f"  mem    copy {m['copy_GBs']:>7} | read {m['read_GBs']:>7} | "
              f"triad {m['triad_GBs']:>7}  GB/s")
        for label, v in res["compute"][i].items():
            if v["best_tflops"] is None:
                err = next(iter(v["by_size"].values())).get("error", "unsupported")
                print(f"  {label:<9}     n/a  ({err[:60]})")
            else:
                print(f"  {label:<9} {v['best_tflops']:>7} TFLOPS")
    if "sustained" in res:
        print("\nsustained (all GPUs concurrent, bf16):")
        for i, s in res["sustained"]["per_gpu"].items():
            print(f"  GPU{i}  {s['first']} -> {s['last']} TFLOPS  "
                  f"(min {s['min']}, retention {s['retention_pct']}%)")
        for i, t in res["sustained"]["telemetry"].items():
            print(f"  GPU{i}  sm {t['sm_mhz']['start']:.0f}->{t['sm_mhz']['end']:.0f} MHz "
                  f"(min {t['sm_mhz']['min']:.0f})  peak {t['watt_max']:.0f}W  "
                  f"{t['temp_c']['max']:.0f}C")
    if "nccl_allreduce" in res and "error" not in res["nccl_allreduce"]:
        print("\nnccl all_reduce busbw:")
        for k, v in res["nccl_allreduce"].items():
            print(f"  {k:>8}  {v['busbw_GBs']:>7} GB/s")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
