# MarketSignal Intelligence Lab

An interactive machine learning workbench for comparing price-forecasting models, inspecting technical signals, and evaluating a rules-based strategy on data the models did not see during training.

[Open the deployed demo](https://stock-trading-bot-ai.onrender.com)

> This is an educational research project. It uses deterministic simulated markets, does not connect to a brokerage, and is not financial advice.

## Why this project exists

Market forecasting demos can look convincing while hiding data leakage, weak evaluation, or unrealistic profitability assumptions. MarketSignal Lab makes those decisions visible. It preserves time order, compares every model with a naive previous-close baseline, includes trading costs, and reports when a selected model does not beat the baseline.

## Interactive features

- Compare Random Forest, Gradient Boosting, and Ridge Regression forecasts
- Adjust the held-out test window, starting capital, signal threshold, and transaction cost
- Review MAE, RMSE, R-squared, and directional accuracy in one leaderboard
- Inspect actual versus predicted prices, SMA9/SMA21 signals, feature importance, and strategy equity
- Compare the signal strategy with buy-and-hold over the same test period
- Review recent enter and exit decisions, total return, and maximum drawdown

## Evaluation workflow

```text
Deterministic market series
        |
        v
Lagged feature engineering
        |
        v
Chronological train/test split
        |
        v
Three regressors + previous-close baseline
        |
        v
Held-out metrics and strategy simulation
```

The feature pipeline uses the current close, daily return, SMA9, SMA21, EMA12, five-day momentum, ten-day volatility, relative volume, and distance from the longer moving average. The target is the next closing price. Earlier observations train the model; later observations form the unseen evaluation window.

## Tech stack

- Python 3.11
- Gradio
- pandas and NumPy
- scikit-learn
- Matplotlib
- Render

## Run the recruiter demo locally

```bash
git clone https://github.com/HydraIsProgramming/Stock_Trading_Bot-AI.git
cd Stock_Trading_Bot-AI
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and launch:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`. The application requires no API keys, GPU, network data feed, or model download.

## Original coursework

`setup.ipynb` contains the original group research project. It explored a custom one-dimensional CNN and image-based transfer-learning experiments with ResNet50, InceptionV3, and DenseNet121 using historical stock CSV files from the [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset).

The deployable `app.py` is a later recruiter-facing evolution of that work. It uses lightweight tabular models and bundled simulations so the evaluation is reproducible and the demo can run without the large external dataset. `requirements-legacy.txt` records the notebook's original, much larger environment.

## My contribution

This was a group coursework project. My work included:

- Implementing SMA9 and SMA21 indicators
- Improving the plots and Gradio workflow
- Helping repair multi-stock dataset handling
- Coordinating delivery and keeping the team aligned
- Building the current evaluation-focused demo from the original project

Project team: Ranjot Sandhu, Usama Mohiuddin, Rupesh Rangwani, and Rishubh Gusain.

## Repository guide

| Path | Purpose |
| --- | --- |
| `app.py` | Self-contained Gradio recruiter demo |
| `requirements.txt` | Minimal deployment dependencies |
| `render.yaml` | Render service configuration |
| `setup.ipynb` | Original deep-learning coursework notebook |
| `requirements-legacy.txt` | Original notebook environment |
| `CP468 Stock Market AI Report.docx` | Project report |
| `CP468- Artificial Intelligence Spring 2024-Project (1).pdf` | Assignment brief |

## Limitations

- The deployed markets are simulations, not live or historical securities.
- Backtest results are illustrative and are not evidence of future performance.
- The notebook experiments are preserved as coursework and should not be interpreted as a validated trading system.
- The application has no order execution, portfolio custody, authentication, or brokerage integration.

See [README_DEPLOY.md](README_DEPLOY.md) for deployment notes.
