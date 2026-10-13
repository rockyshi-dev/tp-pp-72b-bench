# tp-pp-72b-bench

Measures how three ways of splitting Qwen2.5-72B-Instruct (bf16) over eight 24 GB GPUs without
peer-to-peer compare in vLLM v0.30.0: tensor parallelism over all eight cards (TP8), pipeline
parallelism over eight stages (PP8), and the hybrid of four-way tensor parallelism in each of two
pipeline stages (TP4xPP2). It reports time to first token, the gap between output tokens and output
throughput at 1, 8 and 32 concurrent requests, and decides two questions with rules fixed in the
code:

- **Table A.** TP8 against PP8: which gives the lower TTFT, the lower token gap and the higher
  throughput? vLLM's documentation advises pipeline parallelism on GPUs without NVLink "for higher
  throughput and lower communication overhead"; Table A says which metric pipeline parallelism
  actually improves on this hardware.
- **Table B.** Does the hybrid TP4xPP2 beat both pure splits on any metric and load?

## What it measures

**Hardware the arms run on.** Eight GPUs on two CPU sockets, GPUs 0-3 on NUMA node 0 and 4-7 on
node 1, no GPU peer-to-peer (stock driver), so NCCL stages every transfer through host memory (its
SHM transport) and transfers between the sockets also cross the processor interconnect. The
model needs all eight cards (135.4 GiB of weights).

| Arm | vLLM flags | Stages and cards |
|---|---|---|
| TP8 | `--tensor-parallel-size 8` | one stage, GPUs 0-7; every all-reduce crosses the sockets |
| TP4xPP2 | `-tp 4 -pp 2` | GPUs 0-3 (socket 0), 4-7 (socket 1); all-reduces stay inside a socket, one hop crosses |
| PP8 | `--pipeline-parallel-size 8` | GPU i is stage i; seven hops, one across the sockets |

**Only the parallelism flags differ between arms.** Everything else is the same in every arm:

- the model `Qwen/Qwen2.5-72B-Instruct` at revision `495f39366efef23836d0cfae4fbe635880d2be31`,
  served by its local path; vLLM `0.30.0+cu129` (commit `ced6857a`); all eight cards
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,...,7`);
- **a pinned KV pool**: `--kv-cache-memory-bytes 2415919104` (2.25 GiB per GPU, about 58,976
  tokens in every split; 32 x 1,664 = 53,248 are needed at c = 32). If any arm cannot start, all
  arms move down a ladder: step 2 = 1.25 GiB with `--max-model-len 8192` (top level c = 16), step 3
  = 0.75 GiB (levels c = 1 and 8 only). An arm that fails at every step is infeasible and dropped;
  the rest are re-tested from step 1;
- **placement by vLLM's own NUMA binding**: `--numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1
  --numa-bind-cpus` with CPUs `0-21,52-73` for GPUs 0-3 and `26-43,78-95` for GPUs 4-7; each worker
  runs under `numactl --physcpubind=<list> --membind=<its GPU's node>`. The API server holds CPUs
  `22-25,74-77`, the load generator `44-51`, the runner and its 1 Hz monitor `96-103`. A `numactl`
  binary is provided outside the Python environment and put first on the server's `PATH` only;
- every other setting a vLLM default (recorded from a DEBUG configuration start: `max_num_seqs`,
  `max_num_batched_tokens`, async scheduling, model runner, concurrent batches, CUDA graph mode,
  prefix caching, KV block size, the all-reduce backend).

Two startup deviations are allowed, each only when a default start fails with its signature and
then for all arms: `NCCL_CUMEM_HOST_ENABLE=0` (a segfault in `cuMemCreate`) and
`NCCL_P2P_DISABLE=1` (a hang after "vLLM is using nccl==" with no progress for 30 minutes).

**Workload.** Chat requests whose rendered prompt is exactly 1,536 tokens (unmodified excerpts of
the vLLM v0.30.0 source, an explicit empty system message, every prompt different from its first
text token on, so there are no prefix-cache hits), `max_tokens` = `min_tokens` = 128,
`temperature` 0, streaming with usage. Closed loop at c = 1, 8 and 32 with 30, 80 and 160 measured
requests after a warm-up of 2c, one client process per eight streams. Every request has exactly
128 output tokens, so at c = 8 and 32 the requests that start together finish together and the
load arrives in waves; that is the load shape, not an artefact.

**Metrics.** Per request: **TTFT**, from sending to the first non-empty text chunk; **token gap**,
the median interval between consecutive content chunks (only for a request that arrived as 128
chunks); mean TPOT, (last chunk - first chunk) / 127, reported only. Per level and round: the median
TTFT, the median token gap and **output throughput** = completion tokens / (last measured
completion - first measured send).

The token gap is a median, so it leaves out the intervals that contain another request's prefill.
Under load mean TPOT can order the arms differently; `report` prints both.

**Procedure.** Four phases, each a separate command with its own output directory and each
needing the receipts of the ones before it:

1. **start-test**: per arm, start, one burst of 64 requests at the step's top level, check that the
   engine is alive, no CUDA error or out-of-memory line was logged, the eight worker binding lines
   and each worker's `Cpus_allowed_list` are as declared, the KV pool was reserved as pinned and (TP
   arms) the all-reduce backend line is present. Free memory per GPU after start is recorded. This
   settles the ladder step and is each arm's compile warm-up.
2. **settings**: per arm, a start with `VLLM_LOGGING_LEVEL=DEBUG`, its startup fields, and gate E.
3. **transport**: per arm, a start with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,SHM`
   (one NCCL debug file per process) and 8 requests; gate P.
4. **timed**: five rounds; round k runs the arms in the order [TP8, TP4xPP2, PP8] rotated left by
   k - 1 (rounds 4 and 5 repeat the orders of rounds 1 and 2). Each arm: start, levels c = 1, 8, 32,
   stop, wait until every server process has exited and every card's memory is back.

**Touch-and-free before every server start.** Each worker's memory is bound to its node, so it can
only be had by reclaiming page cache, and reclaim inside a timed request would be measured. Before
every server start a helper process per NUMA node, bound to that node, writes 40 GB and frees it,
so the kernel reclaims then. **This evicts page cache on both nodes, including other users' cached
files.** The helper refuses, allocating nothing, unless the node's free memory plus file pages
exceed 44 GB, and sets its own `oom_score_adj` to 1000. Afterwards each node must have at least
20 GB usable free memory (Normal-zone free minus its low watermark; one repeat on a failing node),
or the phase stops before the server starts. Per level, each node's usable free memory and the
level's `pgscan_kswapd`, `pgscan_direct` and `allocstall_*` deltas are recorded.

## Gates

A failed gate is recorded with its reason; nothing is dropped silently.

| Gate | Pass condition | If it fails |
|---|---|---|
| S start | the arm passes its start test at the study's ladder step | infeasible after the ladder; fewer than two feasible arms ends the study |
| P transport | every NCCL `Channel ... via` line names SHM, none `via P2P` or `via NET`; at least one SHM line for every stage boundary (PP8: 0->1 ... 6->7; TP4xPP2: 0->4, 1->5, 2->6, 3->7) and from every TP rank; a `Check P2P Type ... isAllDirectP2p 0 ... isAllCudaP2p 0` line from every worker; in TP arms the enabled all-reduce backends of the TP group are exactly `['PYNCCL']`; no arm enables CUSTOM, SYMM_MEM or NCCL_SYMM_MEM | stop, no timing |
| E equal output | 30 prompts plus 5 non-English prompts, one at a time, non-streaming, temperature 0, 128 tokens, `logprobs` with top 5, `repetition_penalty: 1.0`, token ids returned. Per pair of arms and prompt, at the first differing position the difference is benign if each chosen token is in the other arm's top 5 and each arm's own margin is below 1.0 nat; otherwise corruption | an arm corrupted against every other arm is excluded (an unattributable pair excludes both); excluded arms are treated as infeasible |
| E premise | before gate E is read: at every position the emitted token's logprob equals the largest top-5 value among non-stop tokens (stop ids 151645 and 151643 are masked by `min_tokens`) | no exclusion is applied, the settings phase does not pass, nothing later runs |
| K capacity | `num_gpu_blocks x block_size` equal across arms and at least c_max x 1,664; no preemption during a level | the level is invalid in all arms |
| T tokens | every measured request has 1,536 prompt and 128 completion tokens | drop it; more than 2% dropped reruns the level once (second prompt bank, every arm), then invalid |
| C cache | prefix-cache hits below 1% of queried tokens | as T |
| G chunks | at least 95% of requests arrived as 128 content chunks | that level's gap cell is removed for all pairs |
| N placement | eight worker binding lines with the declared CPUs and node, no "numactl args ... rejected" warning, each worker's `Cpus_allowed_list` as declared, at least 95% of each worker's anonymous memory on its GPU's node | rerun the round (at most twice; one replacement round per study), then not measurable |
| Q quiet | no other compute process on the cards; CPU time of processes outside the run at most 2% of each socket over the level (the engine core, bound to both nodes, is attributed per thread to the socket it ran on); each card's clock below 90% of its level median in at most 10% of samples; an errored monitor sample fails the level | rerun the round (at most twice), then not measurable |
| M memory | no out-of-memory or CUDA error and no engine death during a timed level | rerun that arm's part of the round once; a second failure stops the study for review |

## Decision

**Pairwise verdict.** For arms X and Y, a metric and a level, each round gives r = X / Y of the
medians (TTFT, gap) or Y / X (throughput), so r < 1 means X is better. With five rounds: **X
better** if all five r < 0.90, **Y better** if all five > 1.10, **no material difference** if all
five are within 0.90-1.10, **unresolved** otherwise. Deciding cells: TTFT and gap at every valid
level, throughput at every valid level except c = 1. **Cell winner**: the arm that is better than
every other feasible arm in that cell.

**Table A** (TP8 against PP8; first row that applies): A0 not measurable (gate S, P, N or Q ended
the study, or fewer than two valid levels); A1 TP8 or PP8 infeasible or excluded; **A2 trade-off**
(at least one "TP8 better" and one "PP8 better"; the documentation's throughput clause is called
contradicted if any throughput verdict is "TP8 better"); A3 unstable (an unresolved verdict, no
trade-off); A4 no story (every verdict no material difference); A5 TP one-sided; A6 PP one-sided.

**Table B**: B0 not computable (fewer than three arms or two valid levels); B1 hybrid best
(TP4xPP2 wins some cell and no pure arm wins any); **B2 mixed** (TP4xPP2 and a pure arm each win
some cell); B3 pure only; B4 no winner.

**Expected outcome, stated before the first run.** The result matches it only if all of these
hold:

- Table A row A2 and Table B row B3;
- TP8 against PP8: "PP8 better" on TTFT at every level, "TP8 better" on token gap at every level,
  "PP8 better" on throughput at every level except c = 1 (so the throughput clause is supported);
- token gap ordered TP8 < TP4xPP2 < PP8 at every level: "TP8 better" against TP4xPP2 and
  "TP4xPP2 better" against PP8.

A missing cell counts as not matching. `report` prints "Matches the expected outcome stated in the
README: yes", or "no" followed by every condition that failed.

The timed phase writes these tables to `decision-for-review.json` as a calculation, with
`requires_review` set for rows A0, A1, A3, A4, B0, B4, an excluded arm, a terminal gate, or a
result that does not match the expected outcome.

## Requirements

- Linux x86_64, **two sockets with 104 logical CPUs laid out as above** (socket 0 = CPUs 0-25 and
  52-77, socket 1 = 26-51 and 78-103) and **eight 24 GB NVIDIA GPUs**, 0-3 on NUMA node 0 and 4-7 on
  node 1; the run checks this layout (`lscpu`, `nvidia-smi topo -m`) and refuses another one. More than
  44 GB of free memory plus page cache per NUMA node for the touch-and-free step. `nvidia-smi`, `taskset`, `ldd`, `curl`.
- Python environment with **vLLM `0.30.0+cu129`** (the CUDA 12.9 build of commit
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`), its torch and transformers, and `aiohttp`; the run
  refuses another vLLM version. `conda-package-handling` (any Python) for `numactl-install`.
- The two conda-forge packages `numactl-2.0.18-hb03c661_4.conda` and
  `libnuma-2.0.18-hb03c661_4.conda` (linux-64; their sha256 are checked).
- A clean git checkout of vLLM at `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (prompt text).
- The model files (145.4 GB) and its hub metadata (below).
- The whole machine for the phases: any other compute process on a card stops a start, and CPU
  load from other processes fails gate Q.

## Paths

Every machine path is a flag or an environment variable. Run `python -m tppbench` from the
repository root (the package is not installed), or put the root on `PYTHONPATH`; relative paths,
including the default data directory, are relative to the current directory.

| What | Flag | Environment | Default |
|---|---|---|---|
| data directory: setup receipts, `prompts.json`, compile caches, locks, numactl prefix | `--data-dir` | `TPPB_DATA_DIR` | `./tppb-data` |
| model snapshot directory (served by this path) | `--model` | `TPPB_MODEL` | `<HF cache>/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f3936...`, where `<HF cache>` is `$HF_HOME`, or `~/.cache/huggingface` when `HF_HOME` is unset |
| per-GPU lock files (`gpu<N>.lock`; share them with other GPU jobs) | `--lock-dir` | `TPPB_LOCK_DIR` | `<data-dir>/locks` |
| extra lock files held for the whole phase (repeatable; `{gpu}` expands to 0-7, e.g. `--lock /shared/locks/gpu-{gpu}.lock` to exclude jobs that use another naming) | `--lock` | - | none |

## How to run

```bash
export TPPB_DATA_DIR=/data/tppb HF_HOME=/data/hf
# 1. numactl outside the Python environment, then the CPU-only probe of the binding chain
python -m tppbench numactl-install --packages ~/pkgs     # -> $TPPB_DATA_DIR/setup/numactl-install.json
python -m tppbench numactl-probe                         # exit 1 unless mode is restricted or fallback

# 2. the model: hub metadata of the pinned revision, then download (8 byte ranges per file) and verify
curl -s 'https://huggingface.co/api/models/Qwen/Qwen2.5-72B-Instruct/revision/495f39366efef23836d0cfae4fbe635880d2be31?blobs=true' > meta.json
python -m tppbench fetch download --metadata meta.json --hf-home $HF_HOME [--endpoint https://<mirror>]
python -m tppbench fetch verify   --metadata meta.json --hf-home $HF_HOME   # -> setup/MODEL_READY.json

# 3. prompts (CPU only), then an independent replay
python -m tppbench prompts --vllm-source ~/vllm             # -> $TPPB_DATA_DIR/prompts.json
python -m tppbench verify-prompts --vllm-source ~/vllm

# 4. the four phases, one after the other, each in a new directory (the whole machine is used),
#    detached from the terminal in one session; each phase starts only if the previous one passed
D=$TPPB_DATA_DIR/runs
setsid sh -c "
  python -m tppbench start-test --output $D/start-01 &&
  python -m tppbench settings --output $D/settings-01 --start-receipt $D/start-01/receipt.json &&
  python -m tppbench transport --output $D/transport-01 --start-receipt $D/start-01/receipt.json \
      --settings-receipt $D/settings-01/receipt.json &&
  python -m tppbench timed --output $D/timed-01 --start-receipt $D/start-01/receipt.json \
      --settings-receipt $D/settings-01/receipt.json --transport-receipt $D/transport-01/receipt.json
" > phases.log 2>&1 < /dev/null &

# 5. the reading
python -m tppbench report $D/timed-01 --json timed-01-report.json
```

A phase exits 0 only when it passed. A phase refuses a prerequisite receipt that did
not pass, used other prompts or deviations, or was produced by different code (the sha256 of every
file of `tppbench/` is in each `run-manifest.json`). If the start test stops with a deviation
signature (`DeviationSignature`), write a deviation receipt `{"failures": [<the failing start's
start-failure.json>]}` and rerun every phase with `--deviation-receipt`. Other options: `--port`
(default 8150), `--prompts`, `--probe-receipt`, `--lock`; see `--help`.

A server that does not answer after a failure is never killed by the run: it is left running with
`MANUAL_ATTENTION.json` naming its pid, and nothing else starts.

Expected wall time on 8x RTX 4090: each start 5-8 minutes; start tests about 1 hour; settings and
transport about 1.2 hours; the timed phase about 5.5-6 hours (PP8's c = 1 level is the slowest).

## Files written

Every phase's `--output` directory holds `run-manifest.json` (code hashes, paths, topology,
numactl, prompts hash, study), `receipt.json` (pass, sha256 of every evidence file) and, on a
failure, `INCOMPLETE.json` (or `STOPPED-host-memory.json`). Each server start has its own
directory with `manifest.json` (argv, environment, versions, generation config), `server.log`,
`before-gpus.json`, `ready-identity.json`, `ready-cards.json` (worker pid per card, free memory
after start), `api-server-placement.json`, `metrics-startup.txt`, `start-record.json`,
`shutdown.json`, and beside it `<start>-host-memory/host-memory.json`. Each level directory
(`c1`, `c8`, `c32`) holds `warmup/` and `measured/` (`client-*.jsonl`: one record per request with
send time, every content-chunk timestamp, usage), `metrics-before.txt`, `metrics-after.txt`,
`numa/` (`numa_maps` per server process), `placement.json`, `telemetry.jsonl` and
`telemetry-with-boundaries.json`, `host-memory-level.json`, `gate-m.json` and
`gate-calculation.json`.

Per phase: `ladder.json` and `ladder-attempts.json` (start-test, one `pass<P>-step<S>-<arm>/`
per start); `<arm>/startup-fields.json`, `<arm>/gate-e/*.json`, `gate-e.json`, `study.json`
(settings); `<arm>/nccl/` (one NCCL debug file per process), `<arm>/nccl-info-lines.jsonl`,
`<arm>/transport.json` (transport); `round-<k>-attempt-<n>/<arm>/...`, `paired.json`,
`retry-action.json`, `accepted-rounds.json` and `decision-for-review.json` (timed).

The recorded environment in each `manifest.json` has the value of every variable whose name
contains KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL, or starts with SSH_ (any case), replaced by
`<redacted>`. The manifests still record absolute paths, user-level variables (`USER`, `HOME`,
`PWD`, ...) and GPU UUIDs; read them before publishing a run directory.

## How to read the result

`python -m tppbench report <timed dir>` reads the raw records again: for every accepted round,
arm and level it recomputes each request's TTFT, token gap and mean TPOT from its timestamps, gates
K, T, C and G and the level's values from the requests and the `/metrics` snapshots, combines them
as the run does (gate K equality across the arms of a round; for a level retried after a T or C
failure, the replacement cell's K is ANDed with the original's, its G is false when the original
passed T and C but failed G, and K equality covers original and retry cells), and computes Tables A
and B from those values (gates N, Q and M come from the stored cells). It prints:

- the reading table: per level, metric and arm the median over the five rounds (`--statistic
  mean` for the mean; the header names the one used), or the five-round range where the rounds
  span 2% or more, with the cell winner marked; TTFT in seconds with two decimals, token gap in ms
  and throughput in tok/s with one. Verdicts use per-round ratios and do not depend on the
  statistic;
- mean TPOT per level and arm;
- every pairwise verdict with the range of its five ratios;
- Table A (verdicts, row, throughput clause) and Table B (cell winners, row), and whether the
  result matches the expected outcome stated above (section "Decision");
- a gate summary over the accepted cells (passes per gate, KV capacity, preemptions, dropped
  requests, prefix hits, the largest foreign-CPU fraction and monitor interval);
- a warning for every recomputed value, gate or verdict that differs from what the run stored
  (exit code 1). An incomplete run (not five accepted rounds, or a terminal gate) prints no tables
  (exit code 2).

A verdict is about the definitions above: the token gap is a median per request, and a statement
holds for 1,536-token prompts and 128 fixed output tokens, bf16 weights and 24 GB cards without
peer-to-peer across two sockets.

## Tests

```bash
python -m unittest            # 179 CPU-only tests; no GPU, network or vLLM needed
```

They cover every gate (with log lines in the pinned vLLM and NCCL formats), the decision rows,
the ladder, retries and rotation, the phase drivers with mocked servers, the touch-and-free helper
(on recorded `/proc` readings), the setup steps with mocked downloads, and `report` on a synthetic
timed run.

## License

MIT
