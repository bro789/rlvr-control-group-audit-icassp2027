# Experimental code

30 files, ~9200 lines: the machinery behind the paper, organised so each file
can be read against the claim it supports.

Every file here was copied from the machine that ran the experiments, not from a
working copy. [`CHECKSUMS.md`](CHECKSUMS.md) lists the MD5 of each, so a reader
can confirm these are the files that ran rather than a tidied-up version. We
checked this because an earlier draft of this directory got it wrong: it shipped
one configuration's training code under a directory name claiming to cover two,
and two stale files for a third.

Server orchestration shell scripts are excluded — they contain internal network
addresses. Everything here is path-agnostic Python.

---

## Read these three first

| File | Why it matters |
|---|---|
| [`common/verifier.py`](common/verifier.py) | **Every number in the paper passes through this file.** It defines the reward, the clean-subset filter and the grading of the frozen test — all three are the same function, which is the objection the paper answers in Section 9. 149 lines. |
| [`common/solver_tatqa.py`](common/solver_tatqa.py) | The non-learning arm for CodeTAT-QA. **440 lines** — the figure the paper cites for solver engineering cost is literally this file's length. |
| [`common/gate2_verdict.py`](common/gate2_verdict.py) | Where Δ is computed, and where a past mistake is now a guard. |

### What the verifier enforces

Two pre-registered guards, both readable in ~150 lines:

- **G1, anti-constant.** A program containing no arithmetic and no reference to a
  context value — one that just emits a literal — is rejected regardless of
  whether the literal is correct. This is what makes "zero constant-program
  shortcuts in 57,517 positive samples" a real audit result rather than a
  tautology.
- **G2, tolerance.** Correct means within 1% relative error, or 1e-6 absolute
  when gold is zero. BizBench reports the same metric, which is what makes the
  external calibration in Section 9 like-for-like.

One subtlety worth reading the code for: on CodeTAT-QA the verifier injects
`df = <context>` before executing, but the static guards apply **only to the
model's generated code**, never to the injected data. Otherwise the numbers
inside a large dict literal would let a constant program slip past G1.

### The guard a mistake paid for

`gate2_verdict.py` refuses to compute Δ when any baseline is missing, and says
why:

> *Δ(m) = GRPO − max{baseline set}. A missing baseline collapses the max, lowers
> the bar, and prints a false Go.*

That check exists because it once did not. `prereg/DEVIATIONS_v2.md` entry D16
records a verdict script that silently dropped the solver from the max set and
reported Δ = +3.00 where the corrected value is −14.00. The incident, the fix
and the code are published together.

---

## Layout

```
common/                      byte-identical across all three configurations
  verifier.py                reward, clean filter, frozen-test grading, G1/G2
  solver_tatqa.py            non-learning arm for CodeTAT-QA
  solver_v2_combo.py         non-learning arm for CodeFinQA
  gate2_verdict.py           the Δ decision rule and its missing-baseline guard

config1_codetatqa_0.5b/      Config 1: CodeTAT-QA x Qwen2.5-0.5B
config2_codefinqa_1b/        Config 2: CodeFinQA x Llama-3.2-1B
config3_codetatqa_7b/        Config 3: CodeTAT-QA x Qwen2.5-7B
  gate2.py                   the three training arms: continued SFT, RS-SFT, GRPO
  gate_v2.py                 splits, warm-start training, prompts, dose grid
  solver_arm.py              wires up the solver; scarce vs full retrieval
  gate2_eval.py              dev checkpoint selection
  gate4_v2.py                per-item vectors, McNemar
  merge_lora.py              merges a LoRA adapter into the common branch point
  gate2_passk.py             pass@k, code extraction
  final_test_eval.py         frozen-test protocol

followup/                    the 7B follow-up of Appendix G
  gate2_dyn.py               difficulty-filtered GRPO
  test_eval_tseed.py         frozen-test evaluation for train-seed-suffixed runs
```

The three configuration directories are near-identical files that diverged as
each task was instantiated. They are kept separate because that is how they ran;
diffing them shows exactly what changed between tasks. `gate2.py` is identical
between Configs 1 and 2 and differs at 7B; `gate_v2.py` differs in all three.

---

## Where the design claims live

| Paper claim | File |
|---|---|
| All five arms branch from one warm start $W_m$ | `gate2.py`, every branch loads `merged_m{m}_s{seed}` |
| Every arm sees the same $m$ demonstrations | `gate_v2.py`, split construction |
| The solver may retrieve only from that budget's $m$ gold programs | `solver_arm.py`, `scarce` variant |
| Equal wall clock across training arms | `gate2.py`, the 5700 s cap enforced at step boundaries |
| The frozen test is evaluated once | `final_test_eval.py` — refuses to overwrite an existing result file |
| Δ takes a max over an explicit arm list | `gate2_verdict.py` |
| Engine equivalence gate (Appendix F.2) | `../prereg/engine_gate/vllm_equiv.py` |

## Two things a careful reader will notice

**A fifth arm in a docstring.** `gate2_verdict.py` documents a max set with an
extra member — execution-guarded majority voting, which needs no gold answer and
is therefore deployable. The paper reports a four-arm $\mathcal{B}$. The reported
numbers are unaffected: majority voting at $n{=}16$ scores below the solver at
every pre-registered operating budget (Appendix C), so it never attains the
maximum. We name it because it is visible in the code.

**Missing prerequisites stop the run.** `gate2.py` exits non-zero with an
explicit message if the merged warm start does not exist, rather than producing
an empty run. Worth knowing if you script around it: a wrapper that reads `$?`
inside a string containing a command substitution will read the substitution's
status instead, and see success where there was none.

## One local path, left in deliberately

Configs 1 and 2 name their backbones portably:

```python
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"        # config1
MODEL = "unsloth/Llama-3.2-1B-Instruct"     # config2
```

Config 3 does not. It points at a local snapshot directory, with the reason in a
trailing comment: copying the weights between machines broke the symlinks that
HuggingFace's blob store relies on, so the run was pinned to an unpacked
snapshot instead.

```python
MODEL = "/home/neu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/
         snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
```

We have not rewritten this, because every file here is meant to match the
machine that ran it byte for byte and `CHECKSUMS.md` asserts exactly that. To
run Config 3 elsewhere, substitute:

```python
MODEL = "Qwen/Qwen2.5-7B-Instruct"   # revision a09a35458c702b33eeacc393d103063234e8bc28
```

The hash is the useful part: it pins the exact model revision the 7B results
were produced with, which a portable identifier alone would not.

## Running any of this

These files expect the experiment directory layout (`data/`, `runs_v2/`,
`runs_g2/`, `runs_test/`) and the model weights, neither of which is in this
repository. They are published to be read and checked, not executed end-to-end
from here. The analysis under [`../analysis/`](../analysis/) *is* runnable
against the included data and regenerates every statistic in the paper.
