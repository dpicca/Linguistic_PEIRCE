# Repository Guidelines

## Project Structure & Module Organization
- `pipeline/`: Orchestrates the end‑to‑end run (`urban_lmm_pipeline.py`). Saves JSON artifacts to `data/pipeline_outputs/`.
- `generation/`: Model wrappers (`gpt.py`, `ollama.py`), factory (`model_factory.py`), and the generator (`urban_lmm.py`).
- `critique/`: Hard checks (ontology + entity structure) and soft checks (parsimony, coherence, uncertainty, fluency, cultural fit).
- `refinement/`: Iterative loop that applies critique feedback to regenerate outputs.
- `prompt/`: Prompt templating utilities and prompt files.
- `data/`: Inputs and outputs; includes `urban_dict_lmm.ttl` and sample JSON.
- `test/`: Unit + E2E tests.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Install spaCy model: `python -m spacy download en_core_web_sm`
- Run pipeline (defaults shown): `python run_pipeline.py --model gpt-oss --provider ollama --num-samples 3`
- Custom files: `python run_pipeline.py --input-file data/factual_statements.json --prompts-file data/creative_prompts.json`
- Run all tests: `python -m unittest discover -s test`
- Run a test file: `python -m unittest test/test_pipeline.py`

## Coding Style & Naming Conventions
- Follow PEP 8 (4‑space indentation). Prefer type hints and docstrings (`"""..."""`).
- Modules and packages: lowercase with underscores. Classes: `CapWords`. Functions/vars: `snake_case`.
- Keep components small and composable; avoid adding new deps unless necessary. Reuse factory (`generation/model_factory.py`).
- Outputs: write JSON to `data/pipeline_outputs/` (timestamped filenames as in `UrbanLMMPipeline.save_pipeline_results`).

## Testing Guidelines
- Framework: Python `unittest` in `test/`. Name tests `test_*.py`; structure with `Test*` classes.
- Fast tests: prefer heuristics/fallback paths (no external API calls). Use temp dirs for IO (see tests’ `tempfile.mkdtemp()`).
- Run E2E: `python -m unittest test/test_pipeline_e2e.py` (may skip model calls or rely on fallbacks if credentials/models missing).

## Commit & Pull Request Guidelines
- Messages: Use imperative, concise subjects (e.g., "Add soft critique fallback"), <=72 chars; include scope/file when helpful.
- PRs: Describe intent, changes, and testing. Link issues. Include example commands and sample output paths.
- Avoid committing secrets; never add API keys to `config.yaml`.

## Security & Configuration Tips
- Configure providers in `config.yaml`. Do not commit real keys. For Ollama, pull models (e.g., `ollama pull llama3`).
- Ensure `en_core_web_sm` is installed; large ontology file lives in `data/urban_dict_lmm.ttl`.

## Architecture Overview
- Flow: facts → prompt → generate → hard/soft critique → refine (loop) → save results. See `README.md` and `figures/pipeline_architecture.txt`.
