# Gemma 4 Emotional RLAIF DPO

Standalone project for training and evaluating Gemma4 models with an emotional
SFT, RM, DPO, and PPO workflow.

Repository name:

```text
gemma-4-emotional-rlaif-dpo
```

The project is intentionally isolated from the rest of the RLAIF workspace. Its
virtual environment, Hugging Face cache, datasets, configs, logs, adapters,
predictions, and analysis results all live under this directory.

## Scope

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

Training and prediction datasets mirror the canonical phase2 SFT and phase3
RLAIF LlamaFactory datasets. The large JSON files are kept local and ignored by
Git; `datasets/dataset_info.json` is the trackable index.

```text
mario-rc/aif-emotional-generation
```

Use this check after syncing the large local JSON files:

```bash
scripts/gemma4.py data --verify-only
```

The expected local LlamaFactory dataset files are:

| Local file | Canonical source |
| --- | --- |
| `datasets/sft_demonstration_dataset.json` | phase2 SFT |
| `datasets/sft_demonstration_dataset_foundation.json` | phase2 SFT |
| `datasets/sft_demonstration_dataset_test.json` | phase2 SFT |
| `datasets/sft_demonstration_dataset_test_history.json` | phase2 SFT |
| `datasets/dpo_preference_dataset.json` | phase3 RLAIF |
| `datasets/rm_preference_dataset.json` | phase3 RLAIF |
| `datasets/rm_preference_dataset_test.json` | phase3 RLAIF |
| `datasets/ppo_unlabeled_prompts_dataset.json` | phase3 RLAIF |
| `datasets/ppo_unlabeled_prompts_dataset_test.json` | phase3 RLAIF |

## Layout

```text
gemma-4/
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

## Main CLI

Use the single CLI entrypoint:

```bash
scripts/gemma4.py --help
```

Check current artifacts:

```bash
scripts/gemma4.py status
scripts/gemma4.py status e2b
scripts/gemma4.py status e4b
```

When a training stage has checkpoints but no final adapter yet, `status` also
prints the newest `checkpoint-*` directory that can be used with `--resume`.

## Setup After Clone

Create or activate the project environment, then clone LlamaFactory into the
expected local path:

```bash
cd /autofs/thau00a/home/mrodriguez/data/rlaif/gemma-4
python3 -m venv vgemma4
source vgemma4/bin/activate
python -m pip install --upgrade pip setuptools wheel
git clone https://github.com/hiyouga/LlamaFactory.git LlamaFactory
```

Install LlamaFactory and project dependencies inside `vgemma4`:

```bash
python -m pip install -e ./LlamaFactory
```

If the node needs explicit CUDA 12.8 PyTorch wheels, install PyTorch first and
then install LlamaFactory:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./LlamaFactory
```

The local CLI expects the checkout at `LlamaFactory/` and writes all runtime
caches under `gemma-4/`.

Then download/prepare the datasets from Hugging Face:

```bash
scripts/gemma4.py data
```

Use `--force` only when you intentionally want to overwrite existing local
dataset files:

```bash
scripts/gemma4.py data --force
```

## Full Pipeline

Run the complete flow for one model:

```bash
scripts/gemma4.py pipeline e2b --strict
scripts/gemma4.py pipeline e4b --strict
```

Run both models:

```bash
scripts/gemma4.py pipeline all --strict
```

The pipeline currently automates:

1. SFT training.
2. SFT test prediction.
3. DPO training.
4. DPO test prediction.
5. Analysis.

## Idempotent Execution

Training and prediction commands are safe to re-run. The CLI checks for the
expected final artifact before launching each expensive stage:

| Stage | Skip condition |
| --- | --- |
| Gemma4 SFT train | `saves/<model>/lora/sft_3ep/adapter_model.safetensors` exists |
| Gemma4 SFT predict | `saves/<model>/predict/sft_3ep/generated_predictions.jsonl` exists |
| Gemma4 DPO train | `saves/<model>/lora/dpo_3ep/adapter_model.safetensors` exists |
| Gemma4 DPO predict | `saves/<model>/predict/dpo_3ep/generated_predictions.jsonl` exists |

To deliberately rerun an existing stage, add `--force`:

```bash
scripts/gemma4.py train e2b --stage sft --force
scripts/gemma4.py predict e2b --run-name dpo_3ep --force
scripts/gemma4.py pipeline e2b --strict --force
```

## Resume From Checkpoint

If SFT or DPO stops before the final adapter is written, resume from the latest
`checkpoint-*` directory with `--resume`:

```bash
scripts/gemma4.py train e2b --stage dpo --resume
scripts/gemma4.py pipeline e2b --strict --resume
scripts/gemma4.py pipeline all --strict --resume
```

The CLI keeps the original YAML unchanged, writes a temporary resume config
under `tmp/resume_configs/`, sets `resume_from_checkpoint` to the newest
checkpoint, and disables `overwrite_output_dir` for that resumed run.

Use `--resume` after an interrupted SFT/DPO run. A plain rerun without `--resume`
uses the base YAML and may overwrite an unfinished output directory.

## Running By Stage

SFT:

```bash
scripts/gemma4.py train e2b --stage sft
scripts/gemma4.py predict e2b --run-name sft_3ep
```

DPO via CLI:

```bash
scripts/gemma4.py train e2b --stage dpo
scripts/gemma4.py predict e2b --run-name dpo_3ep
```

Direct YAML-driven runs for configs not yet wired into the main CLI can be
launched through LlamaFactory, for example:

```bash
llamafactory-cli train configs/rm_gemma-4-E2B-it_1ep.yaml
llamafactory-cli train configs/ppo_gemma-4-E2B-it_1ep.yaml
llamafactory-cli train configs/predict_gemma-4-E2B-it_rm_1ep.yaml
llamafactory-cli train configs/predict_gemma-4-E2B-it_ppo_1ep.yaml
```

## Outputs

Expected artifacts for each model:

```text
saves/<model>/lora/sft_3ep/adapter_model.safetensors
saves/<model>/predict/sft_3ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/demonstration_data_emotional_balanced_test_results.json

saves/<model>/lora/rm_1ep/adapter_model.safetensors
saves/<model>/predict/rm_1ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/rm_preference_dataset_test_results.json

saves/<model>/lora/dpo_1ep/adapter_model.safetensors
saves/<model>/predict/dpo_1ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/ppo_unlabeled_prompts_dataset_test_results_dpo_1ep.json

saves/<model>/lora/dpo_3ep/adapter_model.safetensors
saves/<model>/predict/dpo_3ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/ppo_unlabeled_prompts_dataset_test_results_dpo_3ep.json

saves/<model>/lora/ppo_1ep/adapter_model.safetensors
saves/<model>/predict/ppo_1ep/generated_predictions.jsonl
saves/<model>/emotional_balanced/ppo_unlabeled_prompts_dataset_test_results_ppo_1ep.json
```

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
tail -f logs/dpo_gemma-4-E2B-it_dpo_3ep.log
```

## Environment

The project-local environment is `vgemma4`.

By default, `scripts/gemma4.py` infers the project root from its own location.
Set `GEMMA4_ROOT` only if you intentionally want to override that path.

The CLI sets these paths inside `gemma-4` before launching LlamaFactory:

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

## License

This project is released under the MIT License. See `LICENSE`.

The license applies to this project's code, configs, and documentation. Third
party projects, model weights, datasets, and Hugging Face artifacts remain under
their own licenses and terms.
