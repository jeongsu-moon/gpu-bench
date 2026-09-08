# gpu-bench

Portable GPU comparison for **1x RTX PRO 5000 Blackwell 72GB** vs **3x RTX 3090
(NVLink)**. Nothing here is tied to a particular project or model.

Two files, one dependency (PyTorch):

- `bench.py` — measures compute ceiling, memory bandwidth, interconnect, and
  sustained-load throttling. Writes a JSON.
- `compare.py` — diffs two or more JSONs into a table.

## Run

On **each** host:

```bash
nvidia-smi -pm 1                  # persistence mode, reduces clock jitter
python3 bench.py --tag blackwell72          # this host
python3 bench.py --tag 3090x3               # the other host
```

Defaults do a 300s sustained load, which is the point — see "Throttling" below.
`--quick` skips it for a 2-minute sanity run. Then:

```bash
python3 compare.py bench_3090x3.json bench_blackwell72.json
```

The **last** file is the reference; ratios are computed against it.

To benchmark a subset of GPUs, use `CUDA_VISIBLE_DEVICES=0,1` — the NCCL stage
uses every visible device.

## What each layer tells you

| Layer | Metric | Reads as |
|---|---|---|
| matmul sweep | TFLOPS by dtype | compute ceiling; prefill / large-batch training track this |
| memory bandwidth | GB/s copy, read, triad | decode and small-batch training track this, **not** TFLOPS |
| p2p + NCCL all_reduce | GB/s busbw | cost of splitting a model across GPUs |
| sustained load | TFLOPS retention, clocks, watts | what you actually get after 5 minutes |

`compare.py` deliberately reports **per-GPU**, **whole-node**, and **VRAM**
separately. Collapsing them into one number is what makes these comparisons
misleading:

- **per-GPU** answers "which architecture is faster"
- **whole-node** answers "how much work fits through the box per hour" — for
  3x 3090 that means three independent processes (data parallel), which for any
  model that fits in 24GB beats tensor-parallel across NVLink
- **VRAM** answers "does it run at all". A model needing more than 24GB scores
  zero on the 3090 box no matter what the TFLOPS column says.

## Things that will bite you

**NVLink on the 3090 is a 2-way bridge only.** Three cards cannot all be
linked — the real topology is one `NV4` pair plus a third card on PCIe. Run
`nvidia-smi topo -m` on that host before trusting any "3-way NVLink" claim;
`bench.py` captures the topology into the JSON for you.

**The 3090 has no FP8 tensor cores.** Ampere tops out at bf16/fp16; Blackwell
does FP8 (and FP4). The `fp8_e4m3` row will read `n/a` on the 3090 — that gap is
real, not a measurement failure, and it widens further if you quantize.

**Throttling is the whole reason for the 300s default.** Three 3090s under
concurrent load in one chassis heat-soak badly; a 30-second benchmark will not
show it. `bench.py` loads every GPU *simultaneously* and logs SM clock, power,
and temperature throughout, then reports retention (last window ÷ first window).
Compare retention, not just peak.

**Driver and CUDA versions do not need to match across hosts.** Blackwell
requires CUDA 12.8+, which the 3090 host may not have. That is fine for layers
1–3; only pin versions if you go on to layer 4 below.

## Layer 4: real workload

The three layers above give you hardware ratios. They will *not* predict
end-to-end application speed, because real workloads mix compute-bound and
bandwidth-bound phases. If you need a number to plan against, add one
off-the-shelf workload benchmark and run it identically on both hosts:

- **LLM** — `llama.cpp`'s `llama-bench`. A single binary plus one GGUF file, so
  no Python environment matching required. Reports `pp512` (prefill,
  compute-bound) and `tg128` (decode, bandwidth-bound) separately, which map
  directly onto layers 1 and 2 above.
- **Vision / training** — `timm`'s `benchmark.py`, fixed model and batch size,
  report img/s. Pin the same torch version on both hosts for this one.

Run the workload benchmark at a batch size that fits in **24GB**, or the
comparison is not apples-to-apples — then separately note what the 72GB card can
do at a batch size the 3090 cannot reach at all.
