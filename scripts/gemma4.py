#!/usr/bin/env python3
"""Single command-line entrypoint for the Gemma4 training project."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GEMMA4_ROOT", DEFAULT_ROOT)).resolve()
MODELS = {
    "e2b": "gemma-4-E2B-it",
    "e4b": "gemma-4-E4B-it",
}
TRAIN_STAGES = ("sft", "rm_1ep", "dpo_1ep", "dpo_3ep", "ppo_1ep")
PREDICT_STAGES = (
    "predict_sft_3ep",
    "predict_rm_1ep",
    "predict_dpo_1ep",
    "predict_dpo_3ep",
    "predict_ppo_1ep",
)
FULL_STAGES = (
    "sft",
    "predict_sft_3ep",
    "rm_1ep",
    "predict_rm_1ep",
    "dpo_1ep",
    "predict_dpo_1ep",
    "dpo_3ep",
    "predict_dpo_3ep",
    "ppo_1ep",
    "predict_ppo_1ep",
)
PREDICT_RUNS = {
    "sft_3ep": "predict_sft_3ep",
    "rm_1ep": "predict_rm_1ep",
    "dpo_1ep": "predict_dpo_1ep",
    "dpo_3ep": "predict_dpo_3ep",
    "ppo_1ep": "predict_ppo_1ep",
}
TRAIN_CONFIGS = {
    "sft": "sft_{model}.yaml",
    "rm_1ep": "rm_{model}_1ep.yaml",
    "dpo_1ep": "dpo_{model}_1ep.yaml",
    "dpo_3ep": "dpo_{model}_3ep.yaml",
    "ppo_1ep": "ppo_{model}_1ep.yaml",
}
PREDICT_CONFIGS = {
    "predict_sft_3ep": "predict_{model}_sft_3ep.yaml",
    "predict_rm_1ep": "predict_{model}_rm_1ep.yaml",
    "predict_dpo_1ep": "predict_{model}_dpo_1ep.yaml",
    "predict_dpo_3ep": "predict_{model}_dpo_3ep.yaml",
    "predict_ppo_1ep": "predict_{model}_ppo_1ep.yaml",
}
TRAIN_ARTIFACTS = {
    "sft": ("lora", "sft_3ep", "adapter_model.safetensors"),
    "rm_1ep": ("lora", "rm_1ep", "adapter_model.safetensors"),
    "dpo_1ep": ("lora", "dpo_1ep", "adapter_model.safetensors"),
    "dpo_3ep": ("lora", "dpo_3ep", "adapter_model.safetensors"),
    "ppo_1ep": ("lora", "ppo_1ep", "adapter_model.safetensors"),
}
PREDICT_ARTIFACTS = {
    "predict_sft_3ep": ("predict", "sft_3ep", "generated_predictions.jsonl"),
    "predict_rm_1ep": ("predict", "rm_1ep", "generated_predictions.jsonl"),
    "predict_dpo_1ep": ("predict", "dpo_1ep", "generated_predictions.jsonl"),
    "predict_dpo_3ep": ("predict", "dpo_3ep", "generated_predictions.jsonl"),
    "predict_ppo_1ep": ("predict", "ppo_1ep", "generated_predictions.jsonl"),
}
ANALYSIS_ARTIFACTS = {
    "sft_3ep": "demonstration_data_emotional_balanced_test_results.json",
    "dpo_1ep": "ppo_unlabeled_prompts_dataset_test_results_dpo_1ep.json",
    "dpo_3ep": "ppo_unlabeled_prompts_dataset_test_results_dpo_3ep.json",
    "ppo_1ep": "ppo_unlabeled_prompts_dataset_test_results_ppo_1ep.json",
}


def project_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = root / "vgemma4" / "bin"
    cache_root = root / ".cache"
    hf_home = cache_root / "huggingface"

    env["GEMMA4_ROOT"] = str(root)
    env["VIRTUAL_ENV"] = str(root / "vgemma4")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = str(hf_home / "hub")
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["PIP_CACHE_DIR"] = str(cache_root / "pip")
    env["TMPDIR"] = env.get("TMPDIR", "/tmp/gemma4_tmp")
    env["PYTHONPATH"] = f"{root / 'src'}:{env.get('PYTHONPATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for directory in (
        hf_home,
        hf_home / "hub",
        hf_home / "datasets",
        cache_root / "pip",
        root / "logs",
        root / "analysis",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return env


def selected_models(selector: str) -> list[str]:
    if selector == "all":
        return list(MODELS.values())
    return [MODELS[selector]]


def config_for(model_name: str, stage: str) -> Path:
    if stage in TRAIN_CONFIGS:
        filename = TRAIN_CONFIGS[stage].format(model=model_name)
    elif stage in PREDICT_CONFIGS:
        filename = PREDICT_CONFIGS[stage].format(model=model_name)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return ROOT / "configs" / filename


def run_name_for_stage(stage: str) -> str:
    if stage.startswith("predict_"):
        return stage.removeprefix("predict_")
    if stage == "sft":
        return "sft_3ep"
    return stage


def log_for(model_name: str, stage: str) -> Path:
    return ROOT / "logs" / f"{stage}_{model_name}_{run_name_for_stage(stage)}.log"


def artifact_for(model_name: str, stage: str) -> Path:
    if stage in TRAIN_ARTIFACTS:
        parts = TRAIN_ARTIFACTS[stage]
    elif stage in PREDICT_ARTIFACTS:
        parts = PREDICT_ARTIFACTS[stage]
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return ROOT / "saves" / model_name / parts[0] / parts[1] / parts[2]


def latest_checkpoint_for(model_name: str, stage: str) -> Path | None:
    if stage not in TRAIN_STAGES:
        return None

    output_dir = artifact_for(model_name, stage).parent
    if not output_dir.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resume_config_for(config_path: Path, checkpoint_path: Path) -> Path:
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")

    # Keep resume configs portable by storing paths relative to the project root.
    for key in ("adapter_name_or_path", "cache_dir", "dataset_dir", "output_dir"):
        value = config.get(key)
        if isinstance(value, str):
            path_value = Path(value)
            if path_value.is_absolute():
                config[key] = os.path.relpath(path_value, ROOT)

    config["resume_from_checkpoint"] = os.path.relpath(checkpoint_path, ROOT)
    config["overwrite_output_dir"] = False

    output_dir = ROOT / "tmp" / "resume_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / f"{config_path.stem}_resume.yaml"
    with resume_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return resume_path


def skip_existing(label: str, artifact: Path, force: bool) -> bool:
    if artifact.exists() and not force:
        print(f"[SKIP] {label}: {artifact} already exists. Add --force to rerun.")
        return True
    return False


def stream_command(command: list[str], env: dict[str, str], cwd: Path, log_path: Path | None = None) -> int:
    command_text = f"$ {' '.join(command)}"
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8", buffering=1)
        log_file.write(command_text + "\n")
    else:
        log_file = None

    print(command_text)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_file:
                log_file.write(line)
                log_file.flush()
        return process.wait()
    finally:
        if log_file:
            log_file.close()


def run_llamafactory(
    model_name: str,
    stage: str,
    force: bool = False,
    resume: bool = False,
) -> int:
    artifact = artifact_for(model_name, stage)
    if skip_existing(f"{model_name} {stage}", artifact, force):
        return 0
    env = project_env(ROOT)
    config_path = config_for(model_name, stage)
    if resume and stage in TRAIN_STAGES:
        checkpoint_path = latest_checkpoint_for(model_name, stage)
        if checkpoint_path:
            config_path = resume_config_for(config_path, checkpoint_path)
            print(f"[RESUME] {model_name} {stage}: using {checkpoint_path}")
        else:
            print(f"[INFO] {model_name} {stage}: no checkpoint found; starting from the base config.")
    elif resume:
        print(f"[INFO] {model_name} {stage}: --resume only applies to training stages; using the base config.")

    log_path = log_for(model_name, stage)
    command = [str(ROOT / "vgemma4" / "bin" / "llamafactory-cli"), "train", str(config_path)]
    # Run from the project root so relative YAML paths like `datasets/` and
    # `saves/` resolve inside `gemma-4/` rather than inside `LlamaFactory/`.
    return stream_command(command, env=env, cwd=ROOT, log_path=log_path)


def run_module(module: str, args: list[str]) -> int:
    env = project_env(ROOT)
    command = [str(ROOT / "vgemma4" / "bin" / "python"), "-m", module, "--root", str(ROOT), *args]
    return stream_command(command, env=env, cwd=ROOT)


def command_train(args: argparse.Namespace) -> int:
    stages = FULL_STAGES if args.stage == "pipeline" else (args.stage,)
    for model_name in selected_models(args.model):
        for stage in stages:
            code = run_llamafactory(
                model_name,
                stage,
                force=args.force,
                resume=getattr(args, "resume", False),
            )
            if code != 0:
                return code
    return 0


def command_predict(args: argparse.Namespace) -> int:
    run_names = list(PREDICT_RUNS) if args.run_name == "all" else [args.run_name]
    for model_name in selected_models(args.model):
        for run_name in run_names:
            code = run_llamafactory(model_name, PREDICT_RUNS[run_name], force=args.force)
            if code != 0:
                return code
    return 0


def analysis_args(args: argparse.Namespace) -> list[str]:
    models = selected_models(args.model)
    module_args = ["--models", *models]
    if args.strict:
        module_args.append("--strict")
    if args.dataset:
        module_args.extend(["--dataset", args.dataset])
    if args.run_name:
        module_args.extend(["--run-name", args.run_name])
    return module_args


def command_analyze(args: argparse.Namespace) -> int:
    return run_module("gemma4_project.analysis", analysis_args(args))


def command_pipeline(args: argparse.Namespace) -> int:
    train_args = argparse.Namespace(
        model=args.model,
        stage="pipeline",
        force=args.force,
        resume=args.resume,
    )
    code = command_train(train_args)
    if code != 0 or args.skip_analysis:
        return code
    analyze_args = argparse.Namespace(
        model=args.model,
        strict=args.strict,
        dataset=args.dataset,
        run_name=args.run_name,
    )
    return command_analyze(analyze_args)


def command_data(args: argparse.Namespace) -> int:
    module_args = ["--repo-id", args.repo_id]
    if args.revision:
        module_args.extend(["--revision", args.revision])
    if args.force:
        module_args.append("--force")
    if args.verify_only:
        module_args.append("--verify-only")
    return run_module("gemma4_project.data", module_args)


def command_login(args: argparse.Namespace) -> int:
    env = project_env(ROOT)
    command = [str(ROOT / "vgemma4" / "bin" / "hf"), "auth", "login"]
    return stream_command(command, env=env, cwd=ROOT)


def command_status(args: argparse.Namespace) -> int:
    for model_name in selected_models(args.model):
        paths = {
            "sft_adapter": ROOT / "saves" / model_name / "lora" / "sft_3ep" / "adapter_model.safetensors",
            "sft_predictions": ROOT / "saves" / model_name / "predict" / "sft_3ep" / "generated_predictions.jsonl",
            "sft_analysis": ROOT / "saves" / model_name / "emotional_balanced" / ANALYSIS_ARTIFACTS["sft_3ep"],
            "rm_adapter": ROOT / "saves" / model_name / "lora" / "rm_1ep" / "adapter_model.safetensors",
            "rm_predictions": ROOT / "saves" / model_name / "predict" / "rm_1ep" / "generated_predictions.jsonl",
            "dpo_1ep_adapter": ROOT / "saves" / model_name / "lora" / "dpo_1ep" / "adapter_model.safetensors",
            "dpo_1ep_predictions": ROOT / "saves" / model_name / "predict" / "dpo_1ep" / "generated_predictions.jsonl",
            "dpo_1ep_analysis": ROOT / "saves" / model_name / "emotional_balanced" / ANALYSIS_ARTIFACTS["dpo_1ep"],
            "dpo_3ep_adapter": ROOT / "saves" / model_name / "lora" / "dpo_3ep" / "adapter_model.safetensors",
            "dpo_3ep_predictions": ROOT / "saves" / model_name / "predict" / "dpo_3ep" / "generated_predictions.jsonl",
            "dpo_3ep_analysis": ROOT / "saves" / model_name / "emotional_balanced" / ANALYSIS_ARTIFACTS["dpo_3ep"],
            "ppo_1ep_adapter": ROOT / "saves" / model_name / "lora" / "ppo_1ep" / "adapter_model.safetensors",
            "ppo_1ep_predictions": ROOT / "saves" / model_name / "predict" / "ppo_1ep" / "generated_predictions.jsonl",
            "ppo_1ep_analysis": ROOT / "saves" / model_name / "emotional_balanced" / ANALYSIS_ARTIFACTS["ppo_1ep"],
        }
        print(model_name)
        for label, path in paths.items():
            marker = "OK" if path.exists() else "--"
            print(f"  {marker} {label}: {path}")

        for stage in TRAIN_STAGES:
            checkpoint_path = latest_checkpoint_for(model_name, stage)
            if checkpoint_path and not artifact_for(model_name, stage).exists():
                print(f"  OK {stage}_checkpoint: {checkpoint_path}")
    return 0


def add_common_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", choices=[*MODELS, "all"], nargs="?", default="all")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--run-name", choices=[*PREDICT_RUNS, "all"], default="all")
    parser.add_argument("--strict", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Run one stage or the full SFT+RM+DPO+PPO train/predict workflow.",
    )
    train.add_argument("model", choices=[*MODELS, "all"])
    train.add_argument("--stage", choices=[*FULL_STAGES, "pipeline"], default="pipeline")
    train.add_argument("--force", action="store_true", help="Rerun stages even when their expected artifacts exist.")
    train.add_argument("--resume", action="store_true", help="Resume training stages from the latest checkpoint if present.")
    train.set_defaults(func=command_train)

    predict = subparsers.add_parser("predict", help="Run one or more prediction configs on the matching test set.")
    predict.add_argument("model", choices=[*MODELS, "all"], nargs="?", default="all")
    predict.add_argument("--run-name", choices=[*PREDICT_RUNS, "all"], default="all")
    predict.add_argument("--force", action="store_true", help="Regenerate predictions even when they already exist.")
    predict.set_defaults(func=command_predict)

    analyze = subparsers.add_parser("analyze", help="Analyze Gemma4 predictions.")
    add_common_analysis_args(analyze)
    analyze.set_defaults(func=command_analyze)

    pipeline = subparsers.add_parser("pipeline", help="Run the full train/predict workflow and then analysis.")
    pipeline.add_argument("model", choices=[*MODELS, "all"])
    pipeline.add_argument("--skip-analysis", action="store_true")
    pipeline.add_argument("--dataset", default=None)
    pipeline.add_argument("--run-name", choices=[*PREDICT_RUNS, "all"], default="all")
    pipeline.add_argument("--strict", action="store_true")
    pipeline.add_argument("--force", action="store_true", help="Rerun all pipeline stages even when artifacts exist.")
    pipeline.add_argument("--resume", action="store_true", help="Resume training stages from the latest checkpoint if present.")
    pipeline.set_defaults(func=command_pipeline)

    data = subparsers.add_parser("data", help="Download and prepare datasets from Hugging Face.")
    data.add_argument("--repo-id", default="mario-rc/aif-emotional-generation")
    data.add_argument("--revision", default=None)
    data.add_argument("--force", action="store_true", help="Overwrite existing dataset JSON files.")
    data.add_argument("--verify-only", action="store_true", help="Only verify local dataset files.")
    data.set_defaults(func=command_data)

    status = subparsers.add_parser("status", help="Show expected artifacts for each model.")
    status.add_argument("model", choices=[*MODELS, "all"], nargs="?", default="all")
    status.set_defaults(func=command_status)

    login = subparsers.add_parser("login", help="Log in to Hugging Face using the project-local HF cache.")
    login.set_defaults(func=command_login)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
