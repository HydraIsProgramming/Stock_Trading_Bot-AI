# MarketSignal Lab — deployment guide

`app.py` is the recruiter-facing version of this coursework. It uses deterministic bundled markets, a chronological train/test split, three forecasting models, a naive baseline, technical indicators, feature analysis, and a transparent strategy backtest. It does not claim that synthetic experiments or image classifiers forecast real markets.

## Run locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces

1. Create a public Gradio Space.
2. Upload `app.py`, `requirements.txt`, and `README.md`, or connect this repository.
3. The YAML block in `README.md` pins Gradio 5.49.1, Python 3.11, and `app.py` as the entry point.
4. After the Space is healthy, use its direct `https://<space-subdomain>.hf.space` URL as the portfolio iframe URL.

No API keys, network data feeds, model downloads, or GPU are required. `requirements-legacy.txt` records the original notebook environment.

