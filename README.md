# Forecasting-Agent (Trading Track)

Render-hosted FastAPI wrapper that runs the Kalibre trading loop for ProphetHacks.

## Deploy on Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Required Render Environment Variables

Set these manually in Render:

- `PA_SERVER_URL` (example: `https://api.aiprophet.dev`)
- `PA_SERVER_API_KEY` (Prophet Arena server key)
- `OPENROUTER_API_KEY` (OpenRouter key)
- `KALIBRE_TEAM_NAME` (`HarshadM`)

## Recommended Render Environment Variables

- `PA_MAX_TICKS=1500`
- `PA_STARTING_CASH=10000`
- `KALIBRE_ENABLE_LIVE_TRADES=1`
- `KALIBRE_STRATEGY_MODE=forecast_dry_run`
- `KALIBRE_MAX_FORECAST_MARKETS=8`
- `PA_EXPERIMENT_SLUG=eval_harshadm`

## Health endpoint

- `GET /healthz`
- `POST /restart`
