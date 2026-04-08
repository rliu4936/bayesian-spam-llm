# Bayesian Spam LLM

Few-shot email spam detection using **Bayesian inverse classification** with large language models. Rather than training a discriminative classifier, this project flips the problem: it uses a generative LLM to compute the posterior probability of each label given the email content, enabling effective classification with minimal labeled data.

Built on [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) with a custom LLaMA implementation including KV caching, RoPE, and grouped-query attention.

## How It Works

The core idea is **Bayesian inverse classification** -- instead of learning P(label | email) directly, we use the LLM's generative distribution to compute:

$$
P(Y|X) = \frac{P(X, Y)}{\sum_{Y'} P(X, Y')}
$$

For each email, we compute the joint log-probability of the email concatenated with each candidate label ("spam" / "ham") and normalize to get a posterior distribution. This lets a generative model perform classification without any task-specific classification head.

## Methods

| Method | Description |
|--------|-------------|
| **Zero-shot** | Classify using the pretrained model with no additional training |
| **Naive prompting** | Inject a richer task description into the prompt at inference time |
| **Full fine-tuning** | Fine-tune the entire model on labeled examples before evaluation |
| **Prefix tuning** | Train only a small set of prefix key/value vectors while keeping the base model frozen |
| **Ensemble** | Average predictions across multiple independently fine-tuned models |

## Project Structure

```
.
├── model/
│   ├── llama.py              # LLaMA model implementation
│   ├── prefix_llama.py       # Prefix-tuning variant
│   ├── attention.py          # Grouped-query attention with causal masking
│   ├── cache.py              # KV cache for efficient autoregressive inference
│   ├── positional_encoding.py # Rotary position embeddings (RoPE)
│   ├── mlp.py                # SwiGLU feed-forward layers
│   ├── normalization.py      # RMSNorm
│   └── layers.py             # Decoder layer combining attention + MLP
├── examples/
│   ├── bayes_inverse.py      # Main training and evaluation script
│   ├── save_prob_example.py  # Generate probability predictions
│   └── prep_submission_kaggle.py
├── utils/
│   ├── prompt_template.py    # Prompt construction for Bayesian inverse classification
│   ├── sample.py             # Text generation / sampling utilities
│   ├── weight_utils.py       # Model weight loading
│   ├── download.py           # HuggingFace model downloading
│   └── device.py             # Device selection (CPU/GPU/MPS)
├── report.pdf                # Project report
└── pyproject.toml
```

## Quick Start

### Prerequisites

- Python 3.13+
- [UV](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Setup

```bash
# Install dependencies
uv sync
```

### Running

**Zero-shot classification** (no training required):
```bash
uv run -m examples.bayes_inverse --method zero_shot
```

**Naive prompting**:
```bash
uv run -m examples.bayes_inverse --method naive_prompting
```

**Full fine-tuning**:
```bash
uv run -m examples.bayes_inverse --method full_finetune \
    --num_iterations 100 \
    --learning_rate 1e-5 \
    --num_ensembles 9
```

**Generate predictions**:
```bash
uv run -m examples.save_prob_example
```

## Key Implementation Details

- **Custom LLaMA implementation** -- built from scratch in PyTorch with grouped-query attention, RoPE, SwiGLU MLP, and RMSNorm
- **KV Cache** -- dynamic cache implementation for efficient autoregressive generation, avoiding redundant computation of past key/value states
- **Prefix tuning** -- trains only lightweight prefix key/value parameters per layer while keeping the full model frozen, enabling parameter-efficient adaptation
- **Ensemble averaging** -- trains multiple models with different random seeds and averages their probability outputs for more robust predictions

## Results

Achieved **>85% accuracy** on the test set using an ensemble of 9 fine-tuned models with Bayesian inverse classification. See `report.pdf` for detailed experiments and analysis.
