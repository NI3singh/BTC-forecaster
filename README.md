```
██████╗  ████████╗  ██████╗
██╔══██╗ ╚══██╔══╝ ██╔════╝     ·  F O R E C A S T E R
██████╔╝    ██║    ██║
██╔══██╗    ██║    ██║          next-1h & next-4h BTC-USD trend forecasts
██████╔╝    ██║    ╚██████╗     direction · price · range · confidence · why
╚═════╝     ╚═╝     ╚═════╝
```

<h1 align="center">₿ BTC-Forecaster</h1>

<p align="center">
  <b>Honest intraday BTC-USD trend forecasting.</b><br>
  A multi-agent system that predicts whether BTC goes <b>Up / Flat / Down</b> over the
  <b>next 1 hour</b> and <b>next 4 hours</b> — with an approximate price, a price range,
  a confidence score, plain-English reasons, and a <b>self-scoring track record</b> that
  grades every forecast against what actually happened.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-Apache_2.0-green">
  <img alt="Based on TradingAgents" src="https://img.shields.io/badge/forked%20from-TradingAgents-orange">
  <img alt="Status" src="https://img.shields.io/badge/stage-v1%20(agent--based)-yellow">
</p>

---

## What this is

BTC-Forecaster is a **fork of [TradingAgents](https://github.com/TauricResearch/TradingAgents)** — a multi-agent LLM framework that mirrors a real trading desk (technical, news, and sentiment analysts → a bull/bear research debate → a trader → a risk team → a final decision maker).

The upstream project answers *"should I **buy/hold/sell** this stock over the next 1–2 weeks?"* — an **investor** tool on **daily** data.

We kept the whole agent pipeline and **repurposed it into an intraday price forecaster**. Instead of an investment rating, it now answers a **specific, checkable** question:

> **"Which way will BTC-USD move over the next 1 hour and next 4 hours, to roughly what price, and why?"**

The guiding rule of this project is **real, not fake**: every forecast is logged and later **graded against the realized price**, so the accuracy you see is measured — never claimed.

## What changed from TradingAgents

| | TradingAgents (upstream) | **BTC-Forecaster (this repo)** |
|---|---|---|
| **Goal** | Investment decision | Price-trend **forecast** |
| **Output** | `Buy / Hold / Sell` rating | `Up / Flat / Down` + price + range + confidence + reasons |
| **Horizon** | ~1–2 weeks | **Next 1 hour** & **next 4 hours** |
| **Data** | Daily bars | **Hourly (intraday)** bars |
| **Focus** | Any stock / ETF / crypto | **BTC-USD** (intraday) |
| **Self-check** | Decision-reflection memo | **Forecast track record** — scored vs. realized price |
| **You are seen as** | An investor | A **trader / forecaster** |

The agent topology is **unchanged** — only the data layer, the prompts, the final output schema, and the scoring loop were modified, so all the upstream machinery (LangGraph pipeline, multi-provider LLMs, CLI, streaming) still works.

## How it works

```
   Hourly BTC-USD OHLCV + indicators
                │
   ┌────────────▼─────────────┐
   │  Analyst team            │  technical · news · sentiment · (fundamentals = context)
   └────────────┬─────────────┘
   ┌────────────▼─────────────┐
   │  Bull vs. Bear debate    │  argue the next 1–4h direction
   └────────────┬─────────────┘
   ┌────────────▼─────────────┐
   │  Trader → Risk team →    │
   │  Final forecaster        │  emits the Forecast (1h + 4h)
   └────────────┬─────────────┘
                ▼
   Forecast: direction · price · range · confidence · reasons   →  logged to the track record
```

> **Honest note on the current stage (v1).** Today the forecast numbers are the **LLM agents' reasoned judgment** from technical/sentiment/news context — *not yet* a dedicated quantitative model. Raw next-hour direction has a low accuracy ceiling on public data, so the product's value is **calibrated probabilities + an honest range + an explanation + a measured track record**, not a magic price. A fine-tuned time-series model (**Kronos**, a finance-native foundation model for OHLCV data) is planned as the quantitative "brain" in v2 (see [Roadmap](#roadmap)).

## Example

**`analyze`** produces a forecast like this:

```
Primary signal (next 1h): Down

  Horizon   Direction   Approx. price   Expected range       Confidence
  Next 1h   Down        $64,781.00      $64,500 – $65,340    Medium (60%)
  Next 4h   Down        $63,905.00      $63,900 – $65,900    Medium (60%)

Why:   BTC is breaking down with no buy-side absorption below $65,000; VWAP sits
       above spot and RSI at 47 leaves room for downside continuation.
Key levels:            support $64,781 / $63,905 · resistance $65,340 / $65,813
What would invalidate: a 1h close back above $65,900.
```

Later, once the horizons have elapsed, **`score`** grades every logged forecast against the **real** BTC price:

```
Forecast track record (N forecasts logged)

  Horizon   Resolved   Directional acc.   Range coverage   Mean abs % err
  1h        1          100%               0%               1%
  4h        1          100%               100%             1%
```

> A single forecast proves nothing — `100%` on one sample is a coin flip that landed. The track record only becomes meaningful after **dozens** of forecasts across different conditions. That's the whole point of `score`.

## Installation

```bash
git clone https://github.com/NI3singh/BTC-forecaster.git
cd BTC-forecaster

conda create -n btc-forecaster python=3.12
conda activate btc-forecaster

pip install .          # installs all dependencies
```

### API keys

It runs a real multi-agent LLM pipeline, so set **at least one** LLM provider key:

```bash
export OPENAI_API_KEY=...       # OpenAI (GPT)
export ANTHROPIC_API_KEY=...    # Anthropic (Claude)
export GOOGLE_API_KEY=...       # Google (Gemini)
export DEEPSEEK_API_KEY=...     # DeepSeek
# ...OpenRouter, xAI, Qwen, GLM, MiniMax, Azure, Ollama, and any
#    OpenAI-compatible server are also supported (inherited from TradingAgents).
```

Or copy `.env.example` to `.env` and fill in your keys. Market data (BTC-USD hourly OHLCV) comes from Yahoo Finance and needs no key.

## Usage

There are **two commands** — make forecasts now, grade them later.

```bash
# 1) MAKE a forecast  (enter BTC-USD at the prompt). It prints the forecast
#    AND silently logs it to the track record.
python -m cli.main analyze          # or the installed alias:  tradingagents analyze

# 2) ...wait for the horizon to pass (1h / 4h)...

# 3) GRADE all logged forecasts against the realized price.
python -m cli.main score            # or:  tradingagents score
```

Think of it like a weather forecast: `analyze` is *"70% chance of rain tomorrow"*; `score` is checking the next day whether it actually rained, and keeping the logbook honest.

The track record is stored at `~/.tradingagents/forecasts/track_record.jsonl` (override with `TRADINGAGENTS_FORECAST_LOG_PATH`).

## Configuration

Forecasting behavior is set in `tradingagents/default_config.py` (all keys also overridable via `TRADINGAGENTS_*` env vars):

| Key | Default | Meaning |
|---|---|---|
| `data_interval` | `"1h"` | Base OHLCV bar. `"1h"` = intraday forecaster; `"1d"` = original daily behavior. |
| `forecast_horizons` | `[1, 4]` | Horizons in hours the forecaster predicts and scores. |
| `forecast_log_path` | `~/.tradingagents/forecasts/track_record.jsonl` | Append-only log of every forecast + its graded outcome. |

You can also configure the LLM provider, models, and debate rounds — see `default_config.py`.

## Roadmap

This project is staged so each layer is **proven before the next is built**:

- **v1 — agent-based forecaster (this repo, now).** Hourly data + the reframed agent pipeline emit explained 1h/4h forecasts; the `score` loop measures real accuracy.
- **v2 — the quantitative brain (planned).** Fine-tune **Kronos** on BTC hourly data to produce the actual probabilities/price/range; agents shift to explaining and sanity-checking the numbers. Ships only if it beats simple baselines **net of fees**.
- **v3 — the forecast-delivery product (planned).** The full agentic layer wraps the proven model into a polished, explained forecast service.

## Disclaimer

BTC-Forecaster is a **research tool**, not financial, investment, or trading advice. Cryptocurrency is volatile and short-horizon price direction is inherently hard to predict; forecasts are probabilistic and frequently wrong. Never trade on its output without your own due diligence and risk management.

## Credits

Built on **[TradingAgents](https://github.com/TauricResearch/TradingAgents)** by Tauric Research (Apache-2.0). This fork preserves their multi-agent architecture and repurposes it for intraday BTC forecasting. If the underlying framework helps your work, please cite the original:

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138},
}
```
