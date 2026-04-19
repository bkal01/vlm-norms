# vlm-norms

Code for investigating residual stream geometry in vision-language models (VLMs) — specifically whether the norm mismatch between vision adapter outputs and LLM token embeddings creates asymmetric dynamics.

See the [blog post](https://bkal01.github.io/posts/vlm-norms/) for full details.

## Goal

VLM adapters (vision connectors) output tokens with much higher L2 norms than the LLM's own token embeddings (~20x for SmolVLM2). We ask whether this norm mismatch causes vision tokens to dominate or disrupt the residual stream, suppressing text token processing.

## Experimental Setup

- **Models**: SmolVLM2-2.2B, Qwen3-VL-2B (SigLIP-based, no adapter output normalization)
- **Benchmark**: [DatBench](https://huggingface.co/datasets/DatologyAI/DatBench) — 9 categories (chart, counting, document, general, grounding, math, scene, spatial, table), generative format, vision-necessary questions
- **Compute**: Modal (A10G GPU), results saved to a Modal Volume and pulled locally

For each sample we run two forward passes: one with the image, one text-only (same prompt, no image). This lets us isolate whether any observed text token dynamics are caused by vision tokens or are intrinsic to the LLM.

## Reproducing

**Run on Modal** (saves results to the `vlm-norms-runs` volume):

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
