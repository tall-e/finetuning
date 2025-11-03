# SFT and RL Experiments

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

2. Clone and setup:
```bash
git clone <your-repo-url>
cd sft-rl-experiments
uv sync
```

3. Authenticate:
```bash
uv run huggingface-cli login
uv run wandb login
```

4. Test setup:
```bash
uv run python test_setup.py
```

## Project Structure
```
sft-rl-experiments/
├── configs/          # Training configurations
├── data/            # Dataset storage (not committed)
├── models/          # Model checkpoints (not committed)
├── scripts/         # Training and utility scripts
├── src/             # Source code
│   ├── data/       # Data processing
│   ├── models/     # Model definitions
│   └── trainers/   # Training logic
└── notebooks/       # Analysis notebooks
```

## Models Supported
- Mixtral 8x7B
- Gemma 2B/7B
- Qwen 1.5/2

## Usage

Coming soon...
EOF