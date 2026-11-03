# Out-of-distribution evaluation on real logs (Loghub BGL)

The synthetic corpus in the README is closed-world: the classifier's keywords and
the generator's vocabulary are the same thing, so perfect detection there proves
little. This page scores the pipeline's *structure* on logs it was never designed
around: the **BlueGene/L (BGL) supercomputer RAS log** from
[Loghub](https://github.com/logpai/loghub) — 4,747,963 lines over 215 days, with a
per-line alert label (`-` for normal, or a tag such as `KERNDTLB`, `KERNSTOR`,
`APPSEV`) assigned by the machine's operators.

Nothing about BGL was tuned for: the components (`KERNEL`, `APP`, `MMCS`,
`DISCOVERY`, `HARDWARE`, ...) are not in the silicon catalogue, the alert
taxonomy is not ours, and the message vocabulary is different.

## Method

```bash
svla import-bgl --download --window-min 30      # 4.75 M lines -> 2,906 windows + labels (13 s)
svla eval --logs data/bgl/silicon_logs --labels data/bgl/labels --out reports/metrics_bgl.json
svla eval --llm --logs data/bgl/silicon_logs --labels data/bgl/labels --out reports/metrics_bgl_claude.json   # paid; resumable
```

- **Windows as runs.** The stream is cut into 30-minute windows (windows with
  fewer than 20 lines are dropped). A window is *anomalous* if it contains at
  least one alert-labelled line; its *root cause* label is the most frequent
  alert tag. This mirrors the session-window setup used in the log-anomaly
  literature on BGL.
- **Unchanged pipeline.** Parsing, Drain3 templating, redaction, classification,
  clustering, retrieval and the hypothesis agent run exactly as for the
  synthetic corpus. Unknown components get a component-level failure type
  (`KERNEL_FAILURE`, `APP_FAILURE`, ...).
- **Hold-out.** 70% of windows seed the incident store with their alert tags;
  the other 30% are scored as new incidents. With no catalogue entries for BGL,
  the root cause a window is given comes **only from retrieval votes** over
  similar past windows.

## Results

2,906 windows, 633 anomalous; 1,249 failures classified; 968 windows held out
(442 failures, 296 with an alert label).

| Metric | Claude agent (transformers) | rule-based (transformers) | rule-based (hashing, offline) |
|---|:---:|:---:|:---:|
| Window anomaly detection precision / recall / F1 | 0.64 / 1.00 / 0.78 | 0.64 / 1.00 / 0.78 | 0.64 / 1.00 / 0.78 |
| Retrieval recall@5 (same component type) / MRR | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.99 |
| Retrieval recall@5 (same alert tag) | 0.85 | 0.86 | 0.84 |
| **Alert tag named from retrieved incidents, top-1** | **0.76** (224/296) | **0.77** (229/296) | **0.71** (210/296) |
| Answered (did not say OTHER) | 90% | 97% | 93% |
| Mean confidence when right / wrong | 0.84 / 0.62 | 0.77 / 0.60 | 0.76 / 0.59 |
| Cluster purity vs. alert tag | 0.82 | 0.82 | 0.82 |
| Parse coverage (lines matched by the BGL reader) | 100% | 100% | 100% |

Per alert family, top-1 (Claude / rule-based, transformer embeddings): `APP`
0.94 / 0.92 (63), `KERNEL` 0.80 / 0.83 (187), `MMCS` 0.67 / 0.67 (6), `MONITOR`
0.67 / 0.67 (3), `LINKCARD` 0.60 / 0.40 (5), `BGLMASTER` 0.56 / 0.56 (9),
`HARDWARE` 0.17 / 0.17 (6), `DISCOVERY` 0.06 / 0.06 (17).

## What this says

- **Recall transfers, precision does not.** Every alert window is flagged, but
  only 64% of flagged windows contain an operator alert: BGL logs many routine
  `FATAL`-level lines (for example `ciod` application exits) that operators did
  not consider alerts. On real data "severity ≥ ERROR" is a recall-oriented
  first filter, not a verdict. The next step is a learned window-level
  detector on top of the templates, which the synthetic corpus could never
  motivate because there it is already perfect.
- **Retrieval is the part that generalises.** With zero domain knowledge, the
  nearest past windows name the operators' alert tag 77% of the time overall
  (71% with the dependency-free hashing embeddings) and 92% for application alerts. Where it fails, the component has few
  examples (`DISCOVERY`, `HARDWARE`, `MMCS`: 6–17 held-out cases each) — a data
  problem, not a method problem.
- **Clusters are mostly tag-pure (0.82)** even though they were formed with no
  knowledge of the tags, which supports using families as triage buckets.
- **Claude and the rules tie again (0.76 vs 0.77).** Both can only choose among
  the tags attached to the retrieved windows, so the ceiling is set by retrieval
  (0.85 recall@5). Claude abstains more often (answers 90% vs 97%) and is better
  calibrated: 0.84 when right against 0.62 when wrong, versus 0.77 / 0.60 for
  the rules. Same pattern as the synthetic corpus: the LLM's edge is in knowing
  when it does not know, not in raw accuracy.
- **Confidence still separates right from wrong** (rules 0.77 vs 0.60), though less
  sharply than on the synthetic corpus; a catalogue-free setting leaves only
  retrieval agreement to go on.
- **Failure-type scores are not reported.** BGL's alert taxonomy does not
  overlap the silicon failure types, so the typing metric is undefined here
  (the evaluator says so explicitly in the metrics file).

## Cost of the Claude column, honestly

The published Claude column is 442 `claude-opus-5-5` calls at medium effort,
about 2.5 hours wall-clock on a laptop (sequential, ~15 s per call). Getting
there cost more: a first full pass returned an empty column because the tool
schema only allowed catalogue ids (fixed in `afbe4ab`), a second pass stalled
overnight on dropped connections, and 29 calls in the third failed on the
network and were redone from the per-failure cache. Roughly 1,300 calls in
total, in the order of $15–25. The request timeout and the resumable cache
(`--out` + `.hypcache.json`) exist because of this.

## Reproduce

Both metric files are written by the commands above (`metrics_bgl_offline.json`
with `SVLA_OFFLINE=1`, `metrics_bgl_st.json` with transformer embeddings). The
2,000-line Loghub sample under `examples/bgl/` drives the adapter tests.
