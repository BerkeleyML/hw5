"""Shared HW5 code: the question format, prompt, answer parser, checkpoint check, and evaluation.

The staff grader imports this file, so everything here is exactly what runs on
your submission. Do not edit it; changes would only make your local results
differ from your grade.

Every question, in the hidden test set and in any CSV you evaluate, is a row
with the columns in ``MCQ_COLUMNS``: ``id, question, A, B, C, D, E, answer``,
where ``answer`` is one of the letters A-E.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

MCQ_COLUMNS = ("id", "question", "A", "B", "C", "D", "E", "answer")
LETTERS = "ABCDE"
TEST_COLUMNS = MCQ_COLUMNS[:-1]  # an unlabeled question CSV
PREDICTION_COLUMNS = ("id", "answer")  # a predictions CSV

EVAL_MAX_NEW_TOKENS = 128  # one sentence, then \boxed{X}, fits easily; generation still stops at EOS
EVAL_BATCH_SIZE = 64  # on Apple MPS, long question sets can run out of memory; pass a smaller batch_size
EVAL_DTYPE = "float32"
MAX_WEIGHT_BYTES = 2_500_000_000

ALLOWED_DTYPES = {"F16", "BF16", "F32"}
ARCH_CONFIG_KEYS = (
    "model_type",
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "tie_word_embeddings",
)
WEIGHT_FILE_RE = re.compile(r"^model(-\d{5}-of-\d{5})?\.safetensors$")
DOWNLOAD_PATTERNS = [
    "config.json",
    "model.safetensors",
    "model-*-of-*.safetensors",
    "model.safetensors.index.json",
]
OTHER_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".gguf")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REPO_RE = re.compile(
    r"^(?:https?://(?:www\.)?huggingface\.co/)?"
    r"(?P<repo_id>[\w.-]+/[\w.-]+?)(?:\.git)?"
    r"(?:/(?:tree|blob|resolve|commit)/.*)?/?$"
)


class Rejected(Exception):
    """The checkpoint at this commit is not the assigned model."""


# --- Questions, prompt, and parser. ---


def build_mcq_prompt(row: dict) -> str:
    """The user message for one question. The model sees it through the chat template."""
    q = str(row["question"]).strip()
    options = "\n".join(f"{letter}. {row[letter]}" for letter in LETTERS)
    return (
        "Choose exactly one correct option from A, B, C, D, and E.\n"
        "Return your answer inside a LaTeX box.\n\n"
        f"{q}\n\n{options}\n\n"
        "In one sentence, state the key idea, then write \\boxed{X}. Do not analyze each option."
    )


_BOXED = re.compile(r"\\boxed\{")
# The box's contents: one capital letter, optionally written \text{B}, (B), or B.
_BOXED_LETTER = re.compile(r"\s*(?:\\text\{\s*)?\(?([A-E])\)?\.?\s*\}?\s*\}")


def parse_choice_from_boxed(text: str | None) -> str | None:
    """The letter in the first \\boxed{...}, else None. Later boxes and unboxed text never count."""
    m = _BOXED.search(text or "")
    if not m:
        return None
    m = _BOXED_LETTER.match(text, m.end())
    return m.group(1) if m else None


def load_mcq_csv(path: str | Path, max_questions: int = 0) -> list[dict]:
    """Reads and validates a question CSV with the ``MCQ_COLUMNS`` columns.

    An optional ``subset`` column is kept and used for per-subset accuracy.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in MCQ_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        keep = list(MCQ_COLUMNS) + (["subset"] if "subset" in reader.fieldnames else [])
        questions = [{c: row[c] for c in keep} for row in reader]
    for q in questions:
        q["answer"] = q["answer"].strip().upper()
        if q["answer"] not in set(LETTERS):
            raise ValueError(f"question {q['id']} has answer {q['answer']!r}; expected A-E")
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate ids")
    if max_questions:
        questions = questions[:max_questions]
    return questions


def write_mcq_csv(questions: list[dict], path: str | Path) -> None:
    """Writes questions in the format ``load_mcq_csv`` reads."""
    columns = list(MCQ_COLUMNS) + (["subset"] if questions and "subset" in questions[0] else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(questions)


def load_test_csv(path: str | Path, max_questions: int = 0) -> list[dict]:
    """Reads an unlabeled question CSV with the ``TEST_COLUMNS`` columns. Any ``answer`` column is ignored."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in TEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        questions = [{c: row[c] for c in TEST_COLUMNS} for row in reader]
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} has duplicate ids")
    if max_questions:
        questions = questions[:max_questions]
    return questions


def write_predictions_csv(predictions: list[dict], path: str | Path) -> Path:
    """Writes ``id, answer`` prediction rows. A blank answer is scored wrong."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)
    return Path(path)


def openbookqa_to_mcq(row: dict, idx: int, prefix: str = "obqa") -> dict:
    """Converts an ``allenai/openbookqa`` row (4 labeled choices, letter answer) to the five-option format.

    The unused fifth option is "none of the above", as in many hidden-test questions.
    """
    labels, texts = list(row["choices"]["label"]), list(row["choices"]["text"])
    choices = texts + ["none of the above"] * (len(LETTERS) - len(texts))
    q = {"id": f"{prefix}_{idx:05d}", "question": row["question_stem"]}
    q.update(zip(LETTERS, choices))
    q["answer"] = LETTERS[labels.index(row["answerKey"])]
    return q


# --- Devices and precision. ---


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def training_precision() -> tuple[dict, str]:
    """``bf16``/``fp16`` flags for ``SFTConfig``/``GRPOConfig`` on this machine, and a description.

    Weights stay float32; these flags only choose the mixed-precision compute dtype.
    """
    import torch

    # 1. CUDA Check
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return {"bf16": True, "fp16": False}, f"bfloat16 mixed precision on {name}"
        return {"bf16": False, "fp16": True}, f"float16 mixed precision on {name}"

    # 2. Apple Silicon MPS Check
    if torch.backends.mps.is_available():
        try:
            import subprocess
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            chip_name = subprocess.check_output(cmd).decode("utf-8").strip()
            if "Apple" in chip_name and "M1" not in chip_name:
                return {"bf16": True, "fp16": False}, f"bfloat16 mixed precision on {chip_name}"
        except Exception:
            pass
        return {"bf16": False, "fp16": True}, "float16 mixed precision on MPS"
    
    # 3. CPU
    return {"bf16": False, "fp16": False}, "float32 on CPU"


# --- Checkpoint check (metadata only, before any weights are downloaded). ---


@dataclass(frozen=True)
class Reference:
    tensor_shapes: dict
    config: dict
    files_dir: str


def parse_repo_id(repo: str) -> str:
    m = REPO_RE.match(repo.strip())
    if not m or m.group("repo_id").split("/")[0] in {"datasets", "spaces"}:
        raise ValueError(f"{repo!r} is not a Hugging Face model repository URL")
    return m.group("repo_id")


def load_reference(files_dir: str | Path | None = None) -> Reference:
    """Tensor shapes and config of the base model, plus a directory with its tokenizer files."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig

    meta = HfApi().get_safetensors_metadata(BASE_MODEL, revision=BASE_REVISION, token=False)
    shapes = {
        name: tuple(t.shape) for f in meta.files_metadata.values() for name, t in f.tensors.items()
    }
    files_dir = snapshot_download(
        BASE_MODEL,
        revision=BASE_REVISION,
        allow_patterns=["*.json", "merges.txt"],
        local_dir=files_dir,
        token=False,
    )
    config = AutoConfig.from_pretrained(files_dir)
    return Reference(
        tensor_shapes=shapes,
        config={k: getattr(config, k, None) for k in ARCH_CONFIG_KEYS},
        files_dir=str(files_dir),
    )


def _describe_tensor_mismatch(shapes: dict, reference: dict) -> str:
    missing = sorted(reference.keys() - shapes.keys())
    extra = sorted(shapes.keys() - reference.keys())
    wrong = sorted(n for n in shapes.keys() & reference.keys() if shapes[n] != reference[n])
    parts = [f"tensors do not match {BASE_MODEL}"]
    for label, names in (("missing", missing), ("unexpected", extra), ("wrong shape", wrong)):
        if names:
            parts.append(f"{len(names)} {label} (e.g. {', '.join(names[:3])})")
    if "model.embed_tokens.weight" in wrong:
        parts.append(
            f"embedding is {shapes['model.embed_tokens.weight']}, expected "
            f"{reference['model.embed_tokens.weight']}; do not call resize_token_embeddings"
        )
    if any(".lora_" in n or ".base_layer." in n for n in extra):
        parts.append("LoRA layers were saved unmerged; call merge_and_unload() before saving")
    return "; ".join(parts)


def check_checkpoint(
    repo: str,
    commit_sha: str,
    token: str | None,
    reference: Reference,
    max_weight_bytes: int = MAX_WEIGHT_BYTES,
    api=None,
) -> str:
    """Raises Rejected for an invalid checkpoint; other exceptions mean it could not be read.

    Reads only repository metadata, config.json, and the safetensors header.
    Returns the repository id.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
    from transformers import AutoConfig

    api = api or HfApi()
    repo_id = parse_repo_id(repo)
    commit_sha = commit_sha.strip().lower()
    if not SHA_RE.match(commit_sha):
        raise ValueError(f"{commit_sha!r} is not a full 40-character commit SHA")
    token = token or False

    try:
        info = api.model_info(repo_id, revision=commit_sha, token=token, files_metadata=True)
    except RevisionNotFoundError as exc:
        raise Rejected(f"commit {commit_sha} does not exist in {repo_id}") from exc
    except (RepositoryNotFoundError, GatedRepoError) as exc:
        raise PermissionError(f"cannot read {repo_id} with the submitted token") from exc

    files = {s.rfilename: s.size for s in info.siblings or []}
    weight_files = [f for f in files if WEIGHT_FILE_RE.match(f)]
    if not weight_files:
        if "adapter_config.json" in files or "adapter_model.safetensors" in files:
            raise Rejected("only a LoRA adapter was uploaded; merge_and_unload() and upload the full model")
        if any(f.endswith(OTHER_WEIGHT_SUFFIXES) for f in files):
            raise Rejected("weights must be saved as model.safetensors (safe_serialization=True)")
        raise Rejected("no model.safetensors at the repository root")
    if "config.json" not in files:
        raise Rejected("no config.json at the repository root")
    weight_bytes = sum(files[f] or 0 for f in weight_files)
    if weight_bytes > max_weight_bytes:
        raise Rejected(
            f"weights are {weight_bytes / 1e9:.2f} GB; {BASE_MODEL} is about 2 GB in float32"
        )

    config_file = hf_hub_download(repo_id, "config.json", revision=commit_sha, token=token)
    config_dict = json.loads(Path(config_file).read_text())
    if config_dict.get("quantization_config"):
        raise Rejected("quantized checkpoints are not accepted; save in float32, bfloat16, or float16")
    try:
        config = AutoConfig.from_pretrained(config_file)
    except (ValueError, KeyError, OSError) as exc:
        raise Rejected(f"config.json is not a {BASE_MODEL} config: {exc}") from exc
    mismatched = [
        f"{k}={getattr(config, k, None)!r}"
        for k in ARCH_CONFIG_KEYS
        if getattr(config, k, None) != reference.config[k]
    ]
    if mismatched:
        raise Rejected(f"config.json does not match {BASE_MODEL}: {', '.join(mismatched)}")

    meta = api.get_safetensors_metadata(repo_id, revision=commit_sha, token=token)
    tensors = {name: t for f in meta.files_metadata.values() for name, t in f.tensors.items()}
    shapes = {name: tuple(t.shape) for name, t in tensors.items()}
    if shapes != reference.tensor_shapes:
        raise Rejected(_describe_tensor_mismatch(shapes, reference.tensor_shapes))
    dtypes = {t.dtype for t in tensors.values()}
    if not dtypes <= ALLOWED_DTYPES:
        raise Rejected(f"tensor dtypes {sorted(dtypes)} are not all in {sorted(ALLOWED_DTYPES)}")
    return repo_id


def download_checkpoint(repo_id: str, commit_sha: str, token: str | None, local_dir: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id,
        revision=commit_sha.strip().lower(),
        token=token or False,
        local_dir=local_dir,
        allow_patterns=DOWNLOAD_PATTERNS,
    )
    return Path(local_dir)


# --- Evaluation. ---


@dataclass
class EvalSetup:
    """Tokenizer, generation settings, and chat-formatted prompts for one question set."""

    tokenizer: Any
    generation_config: Any
    prompts: list[str]
    order: list[int]


def prepare_eval(
    questions: list[dict],
    files_dir: str | Path | None = None,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
) -> EvalSetup:
    """Uses the base model's tokenizer, chat template, and generation config, never a submission's.

    ``files_dir`` is a local copy of the base model files (``Reference.files_dir``);
    by default they are loaded from the Hub at ``BASE_REVISION``.
    """
    from transformers import AutoTokenizer, GenerationConfig

    source, kwargs = (files_dir, {}) if files_dir else (BASE_MODEL, {"revision": BASE_REVISION})
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    tokenizer.padding_side = "left"
    gen_cfg = GenerationConfig.from_pretrained(source, **kwargs)
    gen_cfg.do_sample = False
    gen_cfg.temperature = gen_cfg.top_p = gen_cfg.top_k = None
    gen_cfg.max_new_tokens = max_new_tokens
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_mcq_prompt(q)}], tokenize=False, add_generation_prompt=True
        )
        for q in questions
    ]
    lengths = [len(ids) for ids in tokenizer(prompts)["input_ids"]]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i], reverse=True)
    return EvalSetup(tokenizer, gen_cfg, prompts, order)


def generate_all(model, setup: EvalSetup, batch_size: int = EVAL_BATCH_SIZE) -> list[str]:
    """Greedy generation in length-sorted batches, splitting a batch that runs out of memory."""
    import torch

    device = next(model.parameters()).device
    tokenizer = setup.tokenizer
    generations = [None] * len(setup.prompts)
    pending = [setup.order[i : i + batch_size] for i in range(0, len(setup.order), batch_size)]
    while pending:
        batch = pending.pop()
        try:
            enc = tokenizer([setup.prompts[i] for i in batch], return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                out = model.generate(**enc, generation_config=setup.generation_config)
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pending += [batch[: len(batch) // 2], batch[len(batch) // 2 :]]
            continue
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, text in zip(batch, texts):
            generations[i] = text
    return generations


def score(questions: list[dict], generations: list[str]) -> dict:
    predictions, correct, by_subset = [], 0, {}
    for q, text in zip(questions, generations):
        pred = parse_choice_from_boxed(text)
        ok = pred == q["answer"]
        correct += ok
        if "subset" in q:
            hits, total = by_subset.get(q["subset"], (0, 0))
            by_subset[q["subset"]] = (hits + ok, total + 1)
        predictions.append({"id": q["id"], "prediction": pred, "correct": ok, "generation": text})
    return {
        "accuracy": correct / len(questions),
        "num_correct": correct,
        "num_questions": len(questions),
        "subset_accuracy": {k: hits / total for k, (hits, total) in by_subset.items()} or None,
        "predictions": predictions,
    }


def evaluate_model(
    model,
    questions: list[dict],
    setup: EvalSetup | None = None,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict:
    """Scores a model that is already in memory, with the grader's prompt, decoding, and parser.

    The grader loads your uploaded weights in float32; a model trained in float32
    with mixed precision (as in the HW5 notebooks) is evaluated the same way here.
    """
    setup = setup or prepare_eval(questions)
    was_training = model.training
    model.eval()
    try:
        return score(questions, generate_all(model, setup, batch_size))
    finally:
        model.train(was_training)


def predict(
    model,
    questions: list[dict],
    setup: EvalSetup | None = None,
    batch_size: int = EVAL_BATCH_SIZE,
) -> list[dict]:
    """Answers unlabeled questions exactly as ``evaluate_model`` would, without scoring.

    Returns one ``{"id", "answer", "generation"}`` per question; ``answer`` is "" when no
    ``\\boxed{X}`` was parsed.
    """
    setup = setup or prepare_eval(questions)
    was_training = model.training
    model.eval()
    try:
        generations = generate_all(model, setup, batch_size)
    finally:
        model.train(was_training)
    return [
        {"id": q["id"], "answer": parse_choice_from_boxed(text) or "", "generation": text}
        for q, text in zip(questions, generations)
    ]


def load_checkpoint(local_dir: str | Path, device=None, dtype: str = EVAL_DTYPE):
    import torch
    from transformers import AutoModelForCausalLM

    device = device or pick_device()
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, dtype=getattr(torch, dtype), use_safetensors=True
    )
    return model.to(device).eval()


def score_checkpoint(
    local_dir: str | Path,
    questions: list[dict],
    setup: EvalSetup,
    device=None,
    dtype: str = EVAL_DTYPE,
    batch_size: int = EVAL_BATCH_SIZE,
) -> dict:
    """Loads a downloaded checkpoint and scores it. ``detail`` flags non-finite weights."""
    import torch

    model = load_checkpoint(local_dir, device, dtype)
    try:
        nonfinite = sum(not torch.isfinite(p).all() for p in model.parameters())
        result = score(questions, generate_all(model, setup, batch_size))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["detail"] = f"{nonfinite} weight tensors contain NaN or inf" if nonfinite else ""
    return result


def evaluate_submission(
    repo: str,
    commit_sha: str,
    token: str | None,
    questions: list[dict],
    work_dir: str | Path = "hw5_eval_work",
    log=print,
) -> dict:
    """Runs the grader's steps for one submission on ``questions``, one step at a time.

    Returns a record with ``status`` ``scored``, ``rejected``, or ``error``, like the grader's.
    """
    import shutil

    work_dir = Path(work_dir)
    record = {"repo": repo, "commit_sha": commit_sha.strip().lower()}
    local_dir = work_dir / "checkpoint"
    try:
        log("1/4 Loading the base model's reference metadata and tokenizer files")
        reference = load_reference(work_dir / "base")
        log("2/4 Checking the checkpoint from metadata only (no weights downloaded)")
        repo_id = check_checkpoint(repo, commit_sha, token, reference)
        log(f"3/4 Downloading {repo_id} at {record['commit_sha'][:12]}")
        shutil.rmtree(local_dir, ignore_errors=True)
        download_checkpoint(repo_id, commit_sha, token, local_dir)
        device = pick_device()
        log(f"4/4 Generating answers to {len(questions)} questions in {EVAL_DTYPE} on {device}")
        setup = prepare_eval(questions, reference.files_dir)
        record.update(score_checkpoint(local_dir, questions, setup, device), status="scored")
    except Rejected as exc:
        record.update(status="rejected", detail=str(exc))
    except Exception as exc:
        record.update(status="error", detail=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)
    return record


# --- Saving and uploading a submission. ---


def save_checkpoint(model, tokenizer, out_dir: str | Path) -> Path:
    """Merges LoRA adapters if present and saves a single safetensors checkpoint."""
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    size = sum(p.stat().st_size for p in out_dir.glob("*.safetensors"))
    print(f"Saved {out_dir} ({size / 1e9:.2f} GB of weights)")
    return out_dir


def upload_checkpoint(out_dir: str | Path, repo_id: str) -> str:
    """Uploads ``out_dir`` to a private model repository and returns the new commit SHA.

    Uses the token from ``huggingface_hub.login()``.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, folder_path=str(out_dir), commit_message="HW5 submission")
    print("Repository URL:", f"https://huggingface.co/{repo_id}")
    print("Commit SHA:    ", commit.oid)
    return commit.oid
