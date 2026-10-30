
<h1 align="center">Can Test-Time Memory Evolution Outperform </br> Test-Time Training for Scientific Discovery?</h1>

<p align="center">
  <b>Yepeng Liu<sup>*</sup></b>, <b>Shengmai Chen<sup>*</sup></b>, Xuandong Zhao, Chengzhi Liu,<br>
  Dawn Song, Xin Eric Wang, Yuheng Bu
</p>

<p align="center">
  UC Santa Barbara &nbsp;·&nbsp; UC Berkeley<br>
  <sub><sup>*</sup> Equal contribution</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.32046"><img src="docs/assets/paper-badge.svg" height="24" alt="Paper on arXiv"></a>
  &nbsp;
  <a href="https://github.com/yepengliu/TTME"><img src="docs/assets/code-badge.svg" height="24" alt="Source code"></a>
  &nbsp;
  <a href="results/README.md"><img src="docs/assets/results-badge.svg" height="24" alt="Reported constructions"></a>
</p>

<p align="center">
  <sub><a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="docs/usage.md">Usage guide</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a></sub>
</p>

Recent test-time training (TTT) frameworks have enabled open-weight large language models (LLMs) to outperform inference-only frameworks powered by frontier models on scientific discovery tasks. However, TTT introduces additional computational cost for online weight updates during the discovery process, making computational efficiency an important consideration when scaling to larger models. This motivates us to explore an efficient test-time adaptation method for scientific discovery. We introduce Test-Time Memory Evolution (TTME), a plug-and-play test-time adaptation method, as an alternative to TTT. TTME continually updates the external memory by distilling the experience to guide subsequent discovery without updating model weights. To have a clean comparison, we evaluate TTME and TTT under the same discovery framework, replacing only the test-time adaptation method.  Our evaluation shows that TTME outperforms TTT on all five mathematical tasks, while reducing computational cost by $82.8$-$84.6\%$ in FLOPs and $47.7$-$58.4\%$ in runtime. These results establish TTME as an effective and computationally efficient alternative to TTT in the evaluated discovery settings.

## Overview

<p align="center">
  <img src="docs/assets/overview.png" width="850" alt="Figure 2 from the paper: test-time training updates model weights, while TTME updates external memory within the same discovery loop.">
</p>

<p align="center"><sub>Figure 2 from the paper. TTME and TTT use the same discovery framework. TTME updates external memory to guide program generation.</sub></p>

TTME uses two types of memory:

- TTME uses execution and evaluation feedback to update **Guideline Memory**. These guidelines help the model write valid programs.
- TTME stores successful algorithmic ideas in **Strategy Memory**. TTME adapts relevant strategies for later searches.

## Results

The following are the **TTME results reported in Tables 1 and 2**. Arrows indicate
whether a lower (↓) or higher (↑) score is better.

**TTT-Discover framework · 50 search steps**

| Model | AC1 ↓ | AC2 ↑ | Erdős ↓ |
|:--|--:|--:|--:|
| Qwen3-8B | 1.504329 | 0.949248 | 0.380927 |
| gpt-oss-120b | **1.502745** | **0.960784** | **0.380862** |

**ThetaEvolve framework · 65 search steps**

| Model | AC2 ↑ | AC3 ↓ | Hadamard ↑ |
|:--|--:|--:|--:|
| Distill-Qwen3-8B | 0.948148 | 1.464224 | 0.579938 |

Distill-Qwen3-8B denotes DeepSeek-R1-0528-Qwen3-8B. See the [paper](https://arxiv.org/abs/2610.32046)
for the TTT and search-only comparisons.

All **nine reported constructions** are included at full precision, with a
[verification script](scripts/verify_results.py) and [SHA-256 manifest](results/manifest.json).
See the [construction catalog](results/README.md) for each array and its provenance.

## Quick start

### Verify the constructions

Use Python 3.10 or newer. Run the following commands to reproduce the stored
construction scores on a CPU. You do not need to download a model.

```bash
git clone https://github.com/yepengliu/TTME.git
cd TTME
python -m pip install -c requirements.lock -e .
python scripts/verify_results.py
```

The verifier checks the saved arrays, hashes, validity, search budgets and agreement
with the paper. A successful run ends with `9/9 constructions verified`.

### Run a search

Search and evaluation workers require **Linux**. Download a model snapshot at the
[documented revision](docs/usage.md#models). Then run a memory-enabled experiment:

```bash
python -m pip install -c requirements.lock -e ".[programs,serve]"
export MODEL_PATH=models/Qwen3-8B
scripts/local/run_local.sh \
  configs/experiments/ac2_memory_8b.yaml \
  runs/ac2_memory_8b 16
```

The launcher starts a model server and runs the search. When the search ends, the
launcher stops the server. Run all commands from the repository root.

For cluster jobs, resuming runs, model settings and experiment variants, see the
[complete usage guide](docs/usage.md).

## Repository guide

| Location | Contents |
|:--|:--|
| [`ttt/`](ttt/) | Search, memory, archive and evaluation implementation |
| [`configs/`](configs/) | Model settings, baseline experiments and memory ablations |
| [`scripts/`](scripts/) | Local and Slurm launchers; construction verification |
| [`results/`](results/) | The nine paper-reported TTME constructions |
| [`docs/usage.md`](docs/usage.md) | Installation, pinned model revisions and full run instructions |

This release provides the **TTT-Discover search implementation** for AC1, AC2 and
Erdős. The ThetaEvolve results are provided as stored constructions and scoring
code; its search runner is not included in this release.

Generated programs execute with the worker account's filesystem and network permissions.
Run evaluation workers in a container or under a separate account; see the
[execution notes](docs/usage.md#safety).

## Citation

If you find this repository useful for your research or applications, please cite our paper:


```bibtex
@article{liu2026ttme,
  title={Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?},
  author={Liu, Yepeng and Chen, Shengmai and Zhao, Xuandong and Liu, Chengzhi and Song, Dawn and Wang, Xin Eric and Bu, Yuheng},
  journal={arXiv preprint arXiv:2610.32046},
  year={2026}
}
```


## Acknowledgments and license

This repository builds on [TTT-Discover](https://github.com/test-time-training/discover)
and includes evaluation material derived from [ThetaEvolve](https://github.com/ypwang61/ThetaEvolve).
See [LICENSE](LICENSE), [NOTICE](NOTICE) and the [third-party license](LICENSES/Apache-2.0.txt)
for the MIT and Apache-2.0 terms that apply to each component.
