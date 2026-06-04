"""
Local-side client that submits training jobs to a RunPod serverless endpoint
and streams the worker's logs back into the FluxGym terminal.

app.py routes training here (instead of running `bash train.sh` locally) when
both RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID are present in the environment
(loaded from .env). Otherwise the local workflow is untouched.
"""
import os
import io
import base64
import time
import zipfile


def is_remote_enabled():
    """True when RunPod credentials + endpoint are configured in the env."""
    return bool(os.environ.get("RUNPOD_API_KEY") and os.environ.get("RUNPOD_ENDPOINT_ID"))


def _zip_dataset_b64(dataset_dir):
    """Zip the (already resized) dataset folder and base64-encode it.

    Files are stored relative to dataset_dir so the worker can recreate
    datasets/<slug>/<files> verbatim.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dataset_dir):
            for name in files:
                full = os.path.join(root, name)
                arc = os.path.relpath(full, dataset_dir)
                zf.write(full, arc)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resolve_hf_repo(lora_name):
    """Build owner/<slug> from HF_REPO_OWNER, or "" to skip the HF push."""
    owner = os.environ.get("HF_REPO_OWNER", "")
    if not owner:
        return ""
    from slugify import slugify
    return f"{owner}/{slugify(lora_name)}"


def submit_training(base_model, lora_name, train_script, train_config,
                    sample_prompts, dataset_dir, local_root):
    """Submit a training job and yield log lines (strings) as they arrive."""
    import runpod

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT_ID"])

    dataset_b64 = _zip_dataset_b64(dataset_dir)
    hf_repo = _resolve_hf_repo(lora_name)
    payload = {
        "base_model": base_model,
        "lora_name": lora_name,
        "train_script": train_script,
        "train_config": train_config,
        "sample_prompts": sample_prompts,
        "local_root": local_root,
        "dataset_zip_b64": dataset_b64,
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "hf_repo": hf_repo,
        "hf_visibility": os.environ.get("HF_VISIBILITY", "public"),
    }

    size_kb = len(dataset_b64) // 1024
    yield f"[RunPod] Dataset packaged ({size_kb} KB). " + (
        f"LoRA will be pushed to https://huggingface.co/{hf_repo}" if hf_repo
        else "No HF_REPO_OWNER set - LoRA will stay on the network volume only."
    )
    if size_kb > 9000:
        yield ("[RunPod] WARNING: encoded dataset is large (>~9MB) and may exceed the "
               "endpoint request limit. Reduce image count/resolution, or stage the "
               "dataset on the network volume / S3 if the job is rejected.")

    job = endpoint.run(payload)
    yield f"[RunPod] Job submitted: id={getattr(job, 'job_id', '?')}"

    # Preferred path: stream incremental output from the generator handler.
    try:
        for output in job.stream():
            for line in _extract_lines(output):
                yield line
        yield "[RunPod] Job complete."
        return
    except Exception as e:
        yield f"[RunPod] Streaming unavailable ({e}); falling back to polling."

    # Fallback: poll status, then dump final output.
    terminal = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")
    while True:
        status = job.status()
        yield f"[RunPod] status={status}"
        if status in terminal:
            break
        time.sleep(5)
    for line in _extract_lines(job.output()):
        yield line
    yield "[RunPod] Job complete."


def _extract_lines(output):
    """Normalize a handler output chunk into a list of log line strings."""
    lines = []
    if output is None:
        return lines
    if isinstance(output, dict):
        msg = output.get("log")
        if msg is None:
            msg = output.get("error") or str(output)
        lines.append(str(msg).rstrip("\n"))
    elif isinstance(output, list):
        for item in output:
            lines.extend(_extract_lines(item))
    else:
        lines.append(str(output).rstrip("\n"))
    return lines
