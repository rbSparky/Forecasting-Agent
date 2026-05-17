# Deploy Kalibre On Render (Trading Track)

This repo is set up to run the trading agent as a FastAPI web service.

## Required rules now encoded

- Slug prefix: `eval_<team-name>` (auto-enforced).
- `n_ticks`: defaults to `1500`.
- Starting cash: defaults to `10000`.

## 1. Keep secrets out of GitHub

Do **not** commit real keys. This repo already ignores `.env`, `key.txt`, logs, and sqlite files.

Use Render environment variables:

- `PA_SERVER_URL=https://api.aiprophet.dev`
- `PA_SERVER_API_KEY=...`
- `OPENROUTER_API_KEY=...`
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `KALIBRE_TEAM_NAME=HarshadM`

Optional overrides:

- `PA_EXPERIMENT_SLUG` (if omitted, auto-generated as `eval_<team>-<timestamp>`)
- `PA_MAX_TICKS` (default `1500`)

## 2. Render service settings

- Language: Python 3
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 3. Runtime behavior

On startup, `main.py` launches `python -m kalibre.loop` in background using the env vars above.

Health endpoint:

- `GET /healthz` → process status, PID, slug, log path

Restart endpoint:

- `POST /restart`

## 4. Submission

After deploy is healthy, submit your Render URL endpoint at:

- https://www.prophethacks.com/submit-endpoint

Recommended check before submitting:

- Visit `/healthz` and confirm:
  - `"running": true`
  - `"slug"` starts with `eval_<team-name>`
