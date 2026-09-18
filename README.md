# Gemma 4 Emotional RLAIF Alignment

Standalone project for training and evaluating Gemma4 models with an emotional
SFT, RM, DPO, and PPO workflow.

Repository name:

```text
gemma-4-emotional-rlaif
```

The project's virtual environment, Hugging Face cache, datasets, configs, logs, adapters,
predictions, and analysis results all live under this directory.

## Scope

### Original campaign

Models:

- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it`

Training stages:

- SFT LoRA, 3 epochs.
- RM LoRA, 1 epoch, initialized from the corresponding SFT adapter.
- DPO LoRA, 1 epoch, initialized from the corresponding SFT adapter.
- DPO LoRA, 3 epochs, initialized from the corresponding SFT adapter.
- PPO LoRA, 1 epoch, initialized from the corresponding SFT adapter and paired RM adapter.
- Test-set prediction after SFT, DPO, RM, and PPO runs.
- Metric analysis for `dpo_1ep`, `dpo_3ep`, and `ppo_1ep` predictions.

## Data Source

Dialogue and DPO preference datasets are downloaded from the Hugging Face dataset
[`mario-rc/aif-emotional-generation`](https://huggingface.co/datasets/mario-rc/aif-emotional-generation).
Store the prepared LlamaFactory JSON files in `datasets/`, next to the
trackable `datasets/dataset_info.json` index.

The preparer copies the dialogue train/test files for PPO, filters rows with
`set == "sft-demonstration"` for SFT, and uses `aif_annotations/train.json` for
DPO. RM train/test files must be supplied separately from the reward-model
preference preparation pipeline; place them in `datasets/` before preparing
data in a standalone clone. Existing files are preserved unless `--force` is used.

Use this check after syncing the large local JSON files:

```bash
scripts/gemma4.py data --verify-only
```

The expected local LlamaFactory dataset files are:

| Local file | Canonical source |
| --- | --- |
| `datasets/sft_demonstration_dataset.json` | `dialogues/train.json`, filtered to SFT demonstrations |
| `datasets/sft_demonstration_dataset_foundation.json` | optional foundation SFT data; not used by the pipeline |
| `datasets/sft_demonstration_dataset_test.json` | `dialogues/test.json`, filtered to SFT demonstrations |
| `datasets/sft_demonstration_dataset_test_history.json` | optional history-aware SFT data; not used by the pipeline |
| `datasets/dpo_preference_dataset.json` | `aif_annotations/train.json` |
| `datasets/rm_preference_dataset.json` | reward-model preferences, supplied separately |
| `datasets/rm_preference_dataset_test.json` | reward-model test preferences, supplied separately |
| `datasets/ppo_unlabeled_prompts_dataset.json` | `dialogues/train.json` |
| `datasets/ppo_unlabeled_prompts_dataset_test.json` | `dialogues/test.json` |

## Layout

```text
gemma-4-emotional-rlaif/
  README.md
  LlamaFactory/                 # cloned LlamaFactory checkout
  vgemma4/                      # project-local virtual environment
  configs/                      # train and predict YAML configs
  datasets/                     # local SFT, RM, DPO, PPO, and test datasets
  saves/                        # adapters, predictions, analyzed per-model outputs
  logs/                         # training and prediction logs
  src/gemma4_project/           # dataset and analysis Python modules
  scripts/
    gemma4.py                   # main CLI entrypoint
    env_gemma4.sh               # environment helper
```

## Setup After Clone

Create or activate a Python 3.12 project environment. If the project
LlamaFactory checkout is absent, obtain its upstream starting point:

```bash
cd gemma-4-emotional-rlaif
python3.12 -m venv vgemma4
source vgemma4/bin/activate
python -m pip install --upgrade pip setuptools wheel
git clone https://github.com/hiyouga/LlamaFactory.git LlamaFactory
git -C LlamaFactory checkout 7af909522a951e3ad9f022ea6f88b6755257eaa5
```

The training environment additionally uses project compatibility patches in
LlamaFactory, including the non-thinking template and trainer fixes. A fresh
upstream clone does not include these patches and is not sufficient to reproduce
the selected models. Preserve the patched checkout when reproducing training.

Install LlamaFactory and project dependencies inside `vgemma4`:

```bash
python -m pip install -e ./LlamaFactory
```

The verified core versions are PyTorch `2.8.0+cu128`, Transformers `5.6.0`,
PEFT `0.18.1` and Accelerate `1.11.0`. PPO additionally uses TRL `0.9.6`,
not the newer TRL dependency selected by the generic editable installation.
Use the project's compatible PPO environment and patches before running PPO;
the installation commands alone do not recreate that environment.

If the node needs explicit CUDA 12.8 PyTorch wheels, install PyTorch first and
then install LlamaFactory:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./LlamaFactory
```

The local CLI expects the checkout at `LlamaFactory/` and writes all runtime
caches under `gemma-4-emotional-rlaif/`.

Then download/prepare the datasets from Hugging Face:

```bash
scripts/gemma4.py data
```

Use `--force` only when you intentionally want to overwrite existing local
dataset files:

```bash
scripts/gemma4.py data --force
```

## Main CLI

Use the single CLI entrypoint:

```bash
scripts/gemma4.py --help
```

Check artifacts from the original campaign (not the selected HF releases):

```bash
scripts/gemma4.py status
scripts/gemma4.py status e2b
scripts/gemma4.py status e4b
```

When a training stage has checkpoints but no final adapter yet, `status` also
reports the newest checkpoint. See the resume limitations below.

## Full Pipeline

Run the original baseline campaign for one model. These commands do not
reproduce the later optimized HF selections; use the published adapters for inference.

```bash
scripts/gemma4.py pipeline e2b --strict
scripts/gemma4.py pipeline e4b --strict
```

Run both models:

```bash
scripts/gemma4.py pipeline all --strict
```

The pipeline currently automates:

1. SFT training and test prediction.
2. RM training and preference-test prediction.
3. DPO training and test prediction for both one and three epochs.
4. PPO training and test prediction.
5. Analysis of SFT, DPO and PPO predictions.

## Idempotent Execution

Training and prediction commands are safe to re-run. The CLI checks for the
expected final artifact before launching each expensive stage:

| Stage | Skip condition |
| --- | --- |
| SFT, RM, DPO and PPO training | The stage's final `adapter_model.safetensors` exists |
| SFT, RM, DPO and PPO prediction | The stage's `generated_predictions.jsonl` exists |

To deliberately rerun an existing stage, add `--force`:

```bash
scripts/gemma4.py train e2b --stage sft --force
scripts/gemma4.py predict e2b --run-name dpo_3ep --force
scripts/gemma4.py pipeline e2b --strict --force
```

## Resume From Checkpoint

If SFT, RM or DPO stops before the final adapter is written, request the latest
available checkpoint with `--resume`:

```bash
scripts/gemma4.py train e2b --stage dpo_3ep --resume
```

The CLI keeps the original YAML unchanged, writes a temporary resume config,
sets `resume_from_checkpoint` to the newest
checkpoint, and disables `overwrite_output_dir` for that resumed run.

The current PPO trainer does not support `--resume`. A plain rerun without
`--resume` uses the base YAML and may overwrite an unfinished output directory.

## Running By Stage

SFT:

```bash
scripts/gemma4.py train e2b --stage sft
scripts/gemma4.py predict e2b --run-name sft_3ep
```

DPO via CLI:

```bash
scripts/gemma4.py train e2b --stage dpo_3ep
scripts/gemma4.py predict e2b --run-name dpo_3ep
```

RM and PPO are also available through the CLI:

```bash
scripts/gemma4.py train e2b --stage rm_1ep
scripts/gemma4.py predict e2b --run-name rm_1ep
scripts/gemma4.py train e2b --stage ppo_1ep
scripts/gemma4.py predict e2b --run-name ppo_1ep
```

## Analysis

Recalculate SFT, DPO and PPO metrics for the original campaign from existing
predictions, without retraining or generating new responses:

```bash
scripts/gemma4.py analyze all --strict
```

`--strict` reports an error if an expected prediction file is missing or its
row count differs from the dataset. Summary files are written under `analysis/`.

## Outputs

Expected artifacts for each model are grouped by training and prediction stage
under `saves/`: adapters (`adapter_model.safetensors`), predictions
(`generated_predictions.jsonl`) and metric summaries. Use `scripts/gemma4.py status`
to inspect the baseline campaign outputs.

Project-level summaries are written to:

```text
analysis/gemma4_summary.json
analysis/gemma4_summary.tsv
analysis/gemma4_summary.md
```

LlamaFactory stdout/stderr for train and prediction stages is mirrored to
`logs/`. The CLI writes the launched command at the top of each log file and
flushes new lines as they arrive, so long-running jobs can be monitored with:

```bash
tail -f logs/dpo_3ep_gemma-4-E2B-it_dpo_3ep.log
```

## Environment

The project-local environment is `vgemma4`.

By default, `scripts/gemma4.py` infers the project root from its own location.
Set `GEMMA4_ROOT` only if you intentionally want to override that path.

The CLI sets these paths inside the project directory before launching LlamaFactory:

- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `XDG_CACHE_HOME`
- `PIP_CACHE_DIR`
- `TMPDIR`
- `PYTHONPATH`

Use project-local Hugging Face login if needed:

```bash
scripts/gemma4.py login
```

## Hugging Face Models

The selected models below are available on Hugging Face. These supersede the original one-/three-epoch baselines for publication. PPO and DPO both continue their matching SFT adapter; PPO does not start from DPO.

| Model | Alignment | Hugging Face |
| --- | --- | --- |
| Gemma-4 E2B IT | PPO | [mario-rc/emotional-rlaif-ppo-gemma-4-e2b-it](https://huggingface.co/mario-rc/emotional-rlaif-ppo-gemma-4-e2b-it) |
| Gemma-4 E2B IT | DPO | [mario-rc/emotional-rlaif-dpo-gemma-4-e2b-it](https://huggingface.co/mario-rc/emotional-rlaif-dpo-gemma-4-e2b-it) |
| Gemma-4 E4B IT | PPO | [mario-rc/emotional-rlaif-ppo-gemma-4-e4b-it](https://huggingface.co/mario-rc/emotional-rlaif-ppo-gemma-4-e4b-it) |
| Gemma-4 E4B IT | DPO | [mario-rc/emotional-rlaif-dpo-gemma-4-e4b-it](https://huggingface.co/mario-rc/emotional-rlaif-dpo-gemma-4-e4b-it) |

See each model card for its inference examples, training parameters and license. The four DPO/PPO adapters were evaluated on the same 392 English dialogue examples from `mario-rc/aif-emotional-generation/dialogues`, split `test`; SFT and RM also have separate stage-specific evaluation sets. E2B selection reused the task test set; its results are not untouched held-out selection evidence.

Use the published tokenizer with `enable_thinking=False` and load the original multimodal base with `AutoModelForImageTextToText`, then the PEFT adapter. Gemma-4 is Apache-2.0.

## Project Origin

This project is based on the original [Mario-RC/aif-emotional-model](https://github.com/Mario-RC/aif-emotional-model) project.

## License

This project is released under the Apache License 2.0. See `LICENSE`.

The license applies to this project's code, configs, and documentation. Third
party projects, model weights, datasets, and Hugging Face artifacts remain under
their own licenses and terms.
