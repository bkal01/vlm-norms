# vlm-norms

Code for investigating residual stream geometry in vision-language models (VLMs) — specifically whether the norm mismatch between vision adapter outputs and LLM token embeddings creates asymmetric dynamics.

See the [blog post](https://bkal01.github.io/posts/vlm-norms/) for full details.

## Goal

VLM adapters (vision connectors) output tokens with much higher L2 norms than the LLM's own token embeddings (~20x for SmolVLM2). We ask whether this norm mismatch causes vision tokens to dominate or disrupt the residual stream, suppressing text token processing.

## Experimental Setup

- **Models**: SmolVLM2-2.2B, Qwen3-VL-2B (SigLIP-based, no adapter output normalization)
- **Benchmark**: [DatBench](https://huggingface.co/datasets/DatologyAI/DatBench) — 9 categories (chart, counting, document, general, grounding, math, scene, spatial, table), generative format, vision-necessary questions
- **Compute**: Modal (A10G GPU), results saved to a Modal Volume and pulled locally

For each sample, the local runner sweeps visual-token scale factors before the
LLM blocks, runs matched blank-image controls for each scale factor, and also
runs one text-only condition. This supports paired comparisons against the
`alpha = 1.0` baseline and real-image comparisons against blank/text-only
controls without storing raw generation tensors.

## Reproducing

**Run locally** (saves results to `runs/<run_id>`):

```bash
uv run python run.py configs/smolvlm.yaml
uv run python run.py configs/qwen3vl.yaml
```

Configs use an alpha sweep:

```yaml
model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
subsets: counting
num_samples: 1
intervention:
  type: scaled
  alphas: [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 1.0, 3.0]
```

`alpha = 1.0` is required because downstream analysis treats it as the paired
baseline.

Each local run writes raw generated answers to `runs/<run_id>/answers.jsonl`,
DatBench per-sample scores to `runs/<run_id>/scores.jsonl`, and DatBench native
reports to `runs/<run_id>/scores/<subset>_<condition>.json`.

The runner also precomputes scalar logit-control artifacts:

```text
runs/<run_id>/logit_sensitivity.jsonl
runs/<run_id>/condition_logit_comparisons.jsonl
runs/<run_id>/vision_sensitivity.jsonl
```

`logit_sensitivity.jsonl` compares real-image alpha conditions against
`real_alpha_1`. `condition_logit_comparisons.jsonl` contains per-generated-step
KL and greedy-token agreement rows for:

```text
real_vs_blank_same_alpha
real_vs_blank_alpha_1
real_vs_textonly
blank_vs_textonly
```

`vision_sensitivity.jsonl` contains one row per sample with baseline
`real_alpha_1` vs `blank_alpha_1` and text-only KL summaries, generated-text
differences, and a `keep_for_intervention_sweep` flag.

Summarize condition-control KLs with:

```bash
uv run scripts/summarize_condition_logit_comparisons.py <run_id>
```

Per-sample tensor artifacts are saved as:

```text
runs/<run_id>/<subset>/<sample_id>/alpha_0.01/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_0.03/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_0.05/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_0.07/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_0.1/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_0.3/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_1/metrics.pt
runs/<run_id>/<subset>/<sample_id>/alpha_3/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_0.01/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_0.03/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_0.05/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_0.07/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_0.1/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_0.3/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_1/metrics.pt
runs/<run_id>/<subset>/<sample_id>/blank_alpha_3/metrics.pt
runs/<run_id>/<subset>/<sample_id>/textonly/metrics.pt
```

Real-image and blank-image rows in `answers.jsonl` and `scores.jsonl` include an
`alpha` field and conditions such as `real_alpha_0.3` and `blank_alpha_0.3`.
Text-only rows use `condition: textonly` and `alpha: null`.

**Run on Modal** (legacy runner; saves results to the `vlm-norms-runs` volume):

```bash
modal run modal_run.py --num-samples 10
modal run modal_run.py --model Qwen/Qwen3-VL-2B-Instruct --num-samples 10
```

**Pull results locally**:

```bash
modal volume get vlm-norms-runs <run_id> runs/
```

**Run analysis scripts** (operate on `runs/` directory):

```bash
python scripts/initial_norms.py
python scripts/norm_vs_updates.py
python scripts/text_dynamics_vision_vs_textonly.py
python scripts/alignment_ratio.py
```
