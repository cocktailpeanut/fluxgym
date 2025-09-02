# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Gradio UI and orchestration (dataset prep, training flags, sampling, HF upload).
- `models/`: Downloaded base model files (`unet`, `vae`, `clip`). Auto‑populated by the app.
- `outputs/`: Training artifacts per LoRA slug (e.g., `outputs/<lora-name>/...`).
- `models.yaml`: Base models list (repo, file, license metadata) used by downloads/publishing.
- `requirements.txt`: Python deps for the UI/orchestration; `sd-scripts/` (kohya) has its own.
- `Dockerfile`, `docker-compose.yml`: Containerized run with GPU.
- Assets: screenshots and icons used by README/UI.

## Build, Test, and Development Commands
- Create venv (Linux/macOS): `python -m venv env && source env/bin/activate`
- Create venv (Windows): `python -m venv env && env\\Scripts\\activate`
- Install deps:
  - `git clone -b sd3 https://github.com/kohya-ss/sd-scripts`
  - `cd sd-scripts && pip install -r requirements.txt && cd ..`
  - `pip install -r requirements.txt`
  - Torch (CUDA 12.1): `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Run locally: `python app.py` then open `http://localhost:7860`.
- Docker: `docker compose up -d --build`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indentation, function/variable `snake_case`, modules/files lowercase with underscores.
- Keep UI labels aligned with underlying `sd-scripts` flags (e.g., `--network_dim`).
- Prefer small, focused functions; avoid side effects in helpers.

## Testing Guidelines
- No formal test suite. Use manual smoke tests:
  - Launch app; verify tabs render and logs stream.
  - Train with a small image set; confirm `outputs/<slug>/` contains `dataset.toml`, logs, and sample images (if enabled).
  - Hugging Face publish: login with token, upload a sample LoRA.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., “add docker compose healthcheck”).
- PRs: clear description, steps to reproduce/verify, screenshots for UI changes, and note if `models.yaml`/docs were updated. Keep scope small.

## Security & Configuration Tips
- Do not commit tokens (`HF_TOKEN`), model weights, or large `outputs/` artifacts.
- Respect model licenses in `models.yaml` when adding new bases.
- GPU differences matter; document VRAM assumptions in PRs that tune defaults.
