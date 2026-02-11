# SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

## Overview

This repository is a **frozen-model ablation** of the [SkillRL paper](https://arxiv.org/abs/2602.08234). The original paper combines skill evolution *with* GRPO fine-tuning on ALFWorld. We isolate a single question:

> **Can an evolving skill library alone — with no weight updates — match or exceed the full pipeline?**

A frozen DeepSeek V3.2 executes all 134 ALFWorld tasks. Between iterations, a teacher LLM analyzes failed trajectories and distills new skills into a hierarchical library. The agent's weights never change; only its skill context evolves.

## Results

| | Overall | Clean | Cool | Heat | Look | Pick | Pick2 | Time |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | 78.36% (105/134) | 87.10% | 80.95% | 65.22% | 72.22% | 79.17% | 82.35% | 807s |
| **Iteration 1** | 89.55% (120/134) | 96.77% | 90.48% | 86.96% | 94.44% | 95.83% | 64.71% | 607s |
| **Iteration 2** | **93.28%** (125/134) | 93.55% | 95.24% | 95.65% | **100.00%** | 87.50% | 88.24% | 703s |


## How It Differs from the Paper

| | Paper (Full SkillRL) | This Implementation |
|---|---|---|
| **Model weights** | Fine-tuned with GRPO | Frozen (no updates) |
| **Skill library** | Evolving hierarchical skills | Same |
| **Teacher system** | Analyzes failures, distills skills | Same |
| **Evaluation** | ALFWorld 134 tasks | Same |
| **Research question** | Do skills + training improve performance? | Do skills *alone* improve performance? |

## Architecture

- **Agent**: Frozen DeepSeek V3.2 (`deepseek-chat`) with tool-calling, executes tasks via think-act-observe loop
- **Skill Library**: Hierarchical — general skills (always injected) + task-specific skills (retrieved via FAISS semantic search, top-K=6)
- **Teacher**: Same DeepSeek V3.2 with a different prompt; analyzes failed trajectories offline and proposes new skills
- **Evolution Loop**: Evaluate all 134 tasks &rarr; collect failures &rarr; teacher distills skills &rarr; update library &rarr; repeat
- **Evaluation**: 10 concurrent workers, atomic JSONL trajectory logging, W&B metrics

## Getting Started

### Prerequisites

- Python 3.12+ (3.14 requires a TextWorld monkey-patch, included)
- [uv](https://docs.astral.sh/uv/) for dependency management
- DeepSeek API key

### Installation

```bash
git clone https://github.com/your-org/SkillRL.git
cd SkillRL
uv sync
```

### Configuration

Set your API key:
```bash
export DEEPSEEK_API_KEY="your-key"
```

### Running

```bash
# Run full evolution loop (eval → analyze → evolve → repeat)
python -m src.main evolve --max-iterations 20

# Run a single evaluation (no evolution)
python -m src.main evaluate

# Resume from a checkpoint
python -m src.main evolve --resume
```

## Next Steps

- **SWE-Bench**: ALFWorld is a relatively constrained environment — the original paper's 32B Qwen model is well-matched to it. Applying this frozen-model skill evolution approach to SWE-Bench would be a more interesting test, where the complexity of real-world software engineering tasks better exercises a capable model like DeepSeek V3.2.

- **Skill library capping**: Currently there is no cap on the number of skills added per iteration. The teacher freely updates and adds skills, which is inefficient and risks filling the prompt with low-value skills. A bounded library with pruning or replacement policies would keep context usage tight.

- **Teacher sampling**: The teacher currently runs on every single trajectory. Running on all failures plus a small sample of successes — or sampling from the full set — could achieve comparable skill quality at a fraction of the cost.

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{xia2026skillrl,
  title={SKILLRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and Liu, Jiaqi and Zeng, Kaide and Wang, Yu and Han, Siwei and Zhou, Yiyang and Zhao, Xujiang and Zhao, Haifeng and Zheng, Zeyu and Xie, Cihang and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}
```
