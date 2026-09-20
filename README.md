# Something Simpler Wins, But Not Always the Same One — extended version and artifacts

Companion repository for the ICASSP 2027 submission *Something Simpler Wins,
But Not Always the Same One: Auditing RLVR Control Groups Across Three Backbone
Scales*.

The four-page paper points here for everything a page limit excludes. This
repository holds the extended manuscript, the pre-registration and its deviation
log, the solver development logs, the per-item outcome vectors for every arm in
every group, and the scripts that turn those vectors into every statistic the
extended version reports.

---

## The short version of what is here

| | |
|---|---|
| **Extended manuscript** | [`paper/extended_version.pdf`](paper/extended_version.pdf) — 33 pages, with six appendices |
| **Headline result** | 17 frozen-test groups, $\Delta$ from $-1.06$ to $-16.42$ points, all negative |
| **The registered test** | GRPO fails to reach the pre-registered $+3$ Go threshold in **14 of 17** groups at 95%; median $P(\Delta \geq +3) = 0.0013$ |
| **Honest caveat 1** | The stricter claim that GRPO is *worse* holds in only 5 of 17, and under Holm correction **2 of 17** are significant |
| **Honest caveat 2** | **No GRPO run completed one epoch** over its prompt pool (0.10–0.79); at 7B only 50–82 prompts per run carried a non-zero advantage |
| **What survives the audit** | Cross-fitted $\Delta$ (select the control arm on half the test set, score on the other half) is negative in **17/17**; the $\max$ selection bias never exceeds 0.47 points |

Reproduce every number:

```bash
python analysis/gen_extended_stats.py
```

The machinery that produced them — the verifier, both solvers, the five arms and
the Δ decision rule — is in [`code/`](code/), annotated against the claims each
file supports.

---

## What the extended version adds over the four-page paper

Four independent reviews of the conference version converged on the same three
demands. The extended version answers all three with data, and two of the three
answers are unfavourable to us. Both are in the main text, not buried.

**1. The decision rule takes a `max` on the frozen test set — isn't that a
winner's curse?** Answered, favourably. A paired bootstrap that recomputes the
`max` on each resample differs from one with the arm frozen at the full-sample
argmax by at most 0.47 points (mean −0.09). A cross-fitted estimator that
selects the control arm on a random half of the test set and scores it on the
other half is negative in 17 of 17 groups, mean −4.18 against the naive −4.44.
The reason is that in 11 of 17 groups the winning arm is a *deterministic*
solver with zero sampling variance leading by at least 8.5 points, so no
selection takes place.

**2. Only two p-values were reported.** Answered, and the answer depends on
which hypothesis you test. The pre-registered decision rule is a Go threshold —
GRPO passes if Δ ≥ +3 — not a test of inferiority. Against the registered
question, a paired bootstrap excludes Δ ≥ +3 in **14 of 17** groups, with
median P(Δ ≥ +3) = 0.0013 and never above 0.0804. Against the stricter question
"is GRPO worse", the interval excludes zero in 5 of 17 and Holm-corrected
McNemar leaves 2 of 17. All 17 tests are now reported both ways, with bootstrap
intervals and per-group minimum detectable effects. The median corrected MDE is
6.93 points against a median measured effect of 2.84, so the design is
underpowered for its own +3 threshold at the group level — in both directions.
The quantitative claim rests on the Go-threshold result and on six independent
operating points agreeing in sign (p = 0.031), not on the count of 17.

One caveat against ourselves: the two Holm-significant groups (Config 2,
m = 128) are also the two in which the RL arm was least healthy —
positive-reward rates of 0.081 and 0.092, a third seed that collapsed to zero,
and 0.19–0.22 epochs of pool coverage. They should not be read as the study's
strongest evidence.

**3. Was GRPO given a fair budget?** Answered, unfavourably. Compute is now
reported on four axes, and prompt-pool coverage is reported for the first time:
no run reached one epoch, and at 7B coverage is about 10%. The conference
version argued that the 7B runs' high positive-reward rate showed they were
training well; **that inference is backwards and is retracted here**. Under
GRPO, an all-correct sample group has zero advantage, so a high positive-reward
rate means *fewer* informative prompts — 50 to 82 per run at 7B, against 719 to
821 at 0.5B.

Also added: a control-group aggregator sensitivity analysis (the same runs report
+15.07 against the warm start alone and −4.44 against the `max`); an
inference-time best-of-*n* arm (not deployable here, because the verifier needs
the gold answer — and even its oracle bound loses to the solver at three of four
operating points); a complementarity analysis (GRPO and the best control arm
disagree on 82 of 282 items in Config 1); an expanded related-work section
including the table-program RL literature that predates RLVR; and two
corrections to the conference version, listed below.

## Corrections to the conference version

1. **7B significance.** The conference version reports the largest 7B margin as
   significant under McNemar (p = 0.017) while declaring a Holm correction
   elsewhere. Under Holm over the family of 17, that value is 0.254 and is not
   significant. No 7B group is significant after correction.
2. **The "8.5 points" figure.** This is the *smallest* margin by which the
   solver leads the nearest learning arm across Config 1's six groups
   (8.51–36.88), not a typical value. The extended version says "at least 8.5".

---

## Layout

```
paper/
  extended_version.pdf     31-page extended manuscript
  tex/                     full LaTeX source (pdflatex + bibtex)
prereg/
  PREREG.md                initial protocol, frozen before Config 1
  PREREG_v2_DRAFT.md       revised protocol
  PREREG_v2_ADDENDUM_CodeTATQA.md
                           the addendum frozen before the confirmatory task
  DEVIATIONS.md            procedural departures, round 1
  DEVIATIONS_v2.md         procedural departures, round 2, with impact signs
  solver_log_*.md          solver development, round by round, with dev accuracy
code/                      the experimental machinery; see code/README.md
  common/                  verifier, both solvers, the Delta decision rule
  config1_codetatqa_0.5b/  Config 1
  config2_codefinqa_1b/    Config 2
  config3_codetatqa_7b/    Config 3
  followup/                the 7B follow-up runs of Appendix G
  CHECKSUMS.md             MD5 of every file, verified against the run machine
analysis/
  gen_extended_stats.py    per-group stats, aggregators, cross-fitting,
                           coverage, zero-advantage -> EXTENDED_STATS.md + tables
  gen_data_facts.py        raw results -> DATA_FACTS.md
  gen_supp_tables.py       ablation, PoT, audit, stability tables
  hacking_audit.py         verifier-hacking audit (H1-H4)
  verify2.py               cross-checks paper numbers against raw JSON
results_backup/            pruned run archive, original paths preserved
  test_final/              per-item outcome vectors, Config 1, 5 arms x 6 groups
  test_final_config2/      per-item outcome vectors, Config 2
  config3_7b/              per-item outcome vectors, Config 3 (7B)
  */runs_g2/grpo_*/meta.json
                           recorded metadata for all 18 GRPO runs, including
                           groups / groups_all_zero / groups_all_one
  */runs_v2/bon_*.json     best-of-n and majority-voting measurements
  */runs_v2/ablation_*.json  solver ablation (rules / retrieval / deployed)
  pot/                     Program-of-Thought / PAL prompting baseline
EXTENDED_STATS.md          generated statistical appendix (Chinese)
DATA_FACTS.md              generated fact table, the authority for every number
```

`analysis/gen_extended_stats.py` runs against this tree unmodified and
regenerates `EXTENDED_STATS.md` plus every generated LaTeX table under
`paper/tex/tab/`. It uses only the Python standard library and takes about a
minute. The other scripts (`gen_data_facts.py`, `gen_supp_tables.py`,
`hacking_audit.py`, `verify2.py`) read parts of the run archive that are not
included here — reward dumps and training logs run to 120 MB — and are shipped
so the procedure is inspectable, not so it can be rerun from this bundle.

### Notes on the artifacts

`prereg/` and the generated `.md` fact tables are in Chinese: they are the
original working records, and we publish them as written rather than as a
translation, because a re-typed protocol is not a pre-registration. The
extended manuscript states in English what each document fixed and when
(Appendix F).

`DEVIATIONS_v2.md` is not curated. It includes the incident in which an early
verdict script silently omitted the solver from the `max` set and reported
Δ = +3.00 where the corrected value is −14.00. That error is why the
aggregation is computed over an explicit arm list and checked, and publishing it
is part of the claim.

One entry in that log deserves to be read on its own: **D12, the inference-engine
equivalence gate**, with its raw result under `prereg/engine_gate/`. Before
adopting vLLM for generation, we compared it against HuggingFace `generate` on
the same checkpoint, the same 200 dev items and a verified-identical prompt
fingerprint, with generation and verification deliberately separated so that only
the engine differed:

| | |
|---|---|
| HuggingFace `generate` | **41.00%** |
| vLLM | **38.00%** |
| difference | **−3.00 pp** |
| per-item reward agreement | 91.00% (18/200 flipped) |
| generated text identical | **38.50%** |
| HF-only correct / vLLM-only correct | 12 / 6 (McNemar χ² = 1.39, n.s.) |

The pre-registered Go threshold for this study is **+3 pp**. The inference engine
alone moves accuracy by 3.00 pp, in no systematic direction — so it is not a
correctable offset but noise of exactly the size of the effect under test. The
mechanism is path divergence rather than floating-point jitter: a 0.5B policy on
a ~3000-token prompt in bf16 has tiny logit gaps, one `argmax` flips, and the
whole emitted program diverges. We therefore rejected vLLM and kept HF
`generate` throughout.

All experiments in this study ran on the servers, on RTX A6000s as the paper
states. The pitfall list inside D12 (WSL2, Python versions, attention-backend
selection) is a local setup note for whoever tries this gate again; no reported
number comes from that machine.

The `results_backup/` tree is the subset needed to reproduce every statistic in
the extended version's Section 8 and appendices, not the full 120 MB run
archive. Model checkpoints, reward dumps and training logs are not included.

Server orchestration scripts are deliberately excluded: they contain internal
network addresses.

## Environment

One NVIDIA RTX A6000 (48 GB) per run. PyTorch 2.10, `transformers` 4.57,
`peft` 0.19, `trl` 1.8. The analysis scripts need only the Python standard
library.

## Citation

<!-- Fill in once the arXiv identifier is assigned. -->
