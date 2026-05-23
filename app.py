"""
app.py — Stock Trading Analysis Dashboard
==========================================
A single-file, production-ready Streamlit app providing:
  • Global Market Pulse header (S&P 500, Nasdaq 100, Dow Jones)
  • Pre-defined stock watchlist via dropdown — no guessing tickers
  • Technical analysis: SMA 20/50, Bollinger Bands, RSI, Volume
  • Context-aware Groq LLM chatbot injected with live market + stock data
  • @st.cache_data(ttl="1h") prevents re-fetching on every chat rerun

Stack: Streamlit · yfinance · Pandas · NumPy · Plotly · Groq (llama-3.3-70b-versatile)
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from .env if present


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

# Pre-defined watchlist — recruiters can explore instantly without knowing symbols
WATCHLIST: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "NFLX", "SPY",  "AMD",
]

# Major indices shown in the Market Pulse header
MARKET_INDICES: dict[str, str] = {
    "S&P 500":    "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones":  "^DJI",
}

# Historical period options for the sidebar
PERIODS: list[str] = ["1mo", "3mo", "6mo", "1y", "2y"]

# Groq model — fast, free, and strong at financial Q&A
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Design tokens (McLaren-inspired dark theme) ──────────────────────────────
BG     = "#0D0D0D"   # page background
PANEL  = "#111111"   # card / sidebar background
GRID   = "#1F1F1F"   # chart grid lines
ORANGE = "#FF8000"   # primary accent
GREEN  = "#22C55E"   # up / bullish
RED    = "#EF4444"   # down / bearish
TEXT   = "#F9FAFB"   # primary text
MUTED  = "#9CA3AF"   # secondary / label text


# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ── Base ─────────────────────────────────────────────── */
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  [data-testid="stSidebar"] {{
      background-color: {PANEL};
      border-right: 1px solid {GRID};
  }}

  /* ── Streamlit metric cards ────────────────────────────── */
  [data-testid="stMetric"] {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 14px 18px;
  }}
  [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: 11px !important; letter-spacing: 0.5px; }}
  [data-testid="stMetricValue"]   {{ color: {TEXT}  !important; font-size: 20px !important; font-weight: 700; }}
  [data-testid="stMetricDelta"]   {{ font-size: 11px !important; }}

  /* ── Market Pulse custom cards ─────────────────────────── */
  .pulse-card  {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 14px 20px;
      text-align: center;
  }}
  .pulse-name  {{ font-size: 10px; color: {MUTED}; text-transform: uppercase; letter-spacing: 1.2px; }}
  .pulse-price {{ font-size: 21px; font-weight: 700; color: {TEXT}; margin: 5px 0 3px; }}
  .pulse-up    {{ font-size: 13px; font-weight: 600; color: {GREEN}; }}
  .pulse-down  {{ font-size: 13px; font-weight: 600; color: {RED};   }}

  /* ── Section label (small orange ALL-CAPS tag) ─────────── */
  .section-label {{
      font-size: 10px; font-weight: 700;
      letter-spacing: 2px; text-transform: uppercase;
      color: {ORANGE}; margin-bottom: 10px;
  }}

  /* ── Chat ──────────────────────────────────────────────── */
  [data-testid="stChatMessage"] {{
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 10px;
      margin-bottom: 6px;
  }}
  [data-testid="stChatInput"] textarea {{
      background: #1A1A1A !important;
      border-color: {ORANGE} !important;
      color: {TEXT} !important;
  }}

  /* ── Sidebar button ────────────────────────────────────── */
  .stButton > button {{
      background: linear-gradient(135deg, {ORANGE}, #FF6B00);
      color: white; border: none;
      border-radius: 8px; font-weight: 600; width: 100%;
  }}
  .stButton > button:hover {{ opacity: 0.88; }}

  /* ── Dropdown / select inputs ──────────────────────────── */
  .stSelectbox > div > div {{
      background: #1A1A1A !important;
      color: {TEXT} !important;
      border-color: {GRID} !important;
  }}
  hr {{ border-color: {GRID}; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# DATA LAYER  — all functions cached with ttl="1h"
#
# WHY CACHE: Streamlit reruns the entire script on every user interaction,
# including every keystroke in the chat box. Without caching, that would
# trigger a fresh yfinance API call on every chat message — slow and wasteful.
# @st.cache_data(ttl="1h") memoises on (ticker, period) so reruns are instant.
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl="1h")
def fetch_market_pulse() -> dict[str, dict]:
    """
    Fetch a 2-day OHLCV snapshot for each major index and compute the
    daily % change. Returns a dict keyed by display name:
        { "S&P 500": {"price": 5321.41, "change_pct": 0.43}, ... }
    Returns None values on any fetch error so the UI degrades gracefully.
    """
    result: dict[str, dict] = {}
    for name, symbol in MARKET_INDICES.items():
        try:
            df = yf.Ticker(symbol).history(period="2d")
            if len(df) >= 2:
                prev = float(df["Close"].iloc[-2])
                curr = float(df["Close"].iloc[-1])
                result[name] = {
                    "price":      round(curr, 2),
                    "change_pct": round((curr - prev) / prev * 100, 2),
                }
            else:
                result[name] = {"price": None, "change_pct": None}
        except Exception:
            result[name] = {"price": None, "change_pct": None}
    return result


@st.cache_data(ttl="1h")
def fetch_and_compute(ticker: str, period: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Download daily OHLCV for `ticker` and compute all technical indicators:
      SMA_20 / SMA_50    — Simple Moving Averages (trend direction)
      BB_upper / BB_lower — Bollinger Bands: SMA_20 ± 2σ (volatility envelope)
      RSI                — 14-day Relative Strength Index (momentum 0–100)
      Volume_SMA20       — 20-day average volume (for spike detection)

    Returns:
        (DataFrame with indicator columns, None)     on success
        (None, error_message_string)                 on failure
    """
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return None, f"No data found for '{ticker}'. Is the symbol correct?"

        # Strip timezone — prevents mismatches between Plotly and Pandas
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.copy()

        # ── Moving averages ────────────────────────────────────────────────────
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()

        # ── Bollinger Bands (±2 standard deviations around the 20-day SMA) ────
        _std         = df["Close"].rolling(20).std()
        df["BB_upper"] = df["SMA_20"] + 2 * _std
        df["BB_lower"] = df["SMA_20"] - 2 * _std

        # ── RSI (Wilder's 14-day method) ───────────────────────────────────────
        # Separates daily gains from daily losses, smooths each over 14 bars,
        # then maps the ratio onto a 0–100 scale.
        delta    = df["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)  # avoid div-by-zero
        df["RSI"] = 100 - (100 / (1 + rs))

        # ── Volume moving average ──────────────────────────────────────────────
        df["Volume_SMA20"] = df["Volume"].rolling(20).mean()

        return df, None

    except Exception as exc:
        return None, str(exc)


def extract_metrics(ticker: str, df: pd.DataFrame) -> dict:
    """
    Flatten the latest row of `df` into a clean metrics dict consumed by both
    the KPI cards and the LLM context builder.

    NOTE: uses `is not None` checks throughout — never bare truthiness — so
    that a value of 0.0 is never misread as "missing".
    """
    latest   = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else latest

    def safe(val) -> float | None:
        """Return a rounded float or None for NaN/missing values."""
        return round(float(val), 2) if pd.notna(val) else None

    # Volume ratio: today vs 20-day average — >2.0x flags institutional activity
    vol_ratio = None
    if pd.notna(latest["Volume_SMA20"]) and latest["Volume_SMA20"] > 0:
        vol_ratio = round(float(latest["Volume"]) / float(latest["Volume_SMA20"]), 2)

    return {
        "ticker":        ticker.upper(),
        "current_price": round(float(latest["Close"]), 2),
        "prev_close":    round(float(prev_row["Close"]), 2),
        "change_pct":    round(
            (float(latest["Close"]) - float(prev_row["Close"]))
            / float(prev_row["Close"]) * 100, 2
        ),
        "sma_20":        safe(latest["SMA_20"]),
        "sma_50":        safe(latest["SMA_50"]),
        "bb_upper":      safe(latest["BB_upper"]),
        "bb_lower":      safe(latest["BB_lower"]),
        "rsi":           safe(latest["RSI"]),
        "volume_ratio":  vol_ratio,
        "period_high":   round(float(df["High"].max()), 2),
        "period_low":    round(float(df["Low"].min()),  2),
    }


# ════════════════════════════════════════════════════════════════════════════════
# LLM CONTEXT BUILDER
# ════════════════════════════════════════════════════════════════════════════════

def build_context_digest(metrics: dict, pulse: dict) -> str:
    """
    Produce a structured plain-text digest injected into the Groq system prompt.

    Combines two data sources so the chatbot can answer both:
      • Asset-specific questions  ("Is NVDA overbought?")
      • Macro market questions    ("How is the Nasdaq doing today?")

    The LLM is instructed never to fabricate numbers outside this digest.
    """
    m     = metrics
    price = m["current_price"]
    chg   = m["change_pct"]
    sma20 = m["sma_20"]
    sma50 = m["sma_50"]

    # ── Trend signal (uses `is not None` — never falsy check on floats) ────────
    if sma20 is not None and sma50 is not None:
        if price > sma20 > sma50:
            trend = "BULLISH — price is above both the 20-day and 50-day SMAs"
        elif price < sma20 < sma50:
            trend = "BEARISH — price is below both the 20-day and 50-day SMAs"
        else:
            trend = "NEUTRAL / MIXED — price is between the two moving averages"
    else:
        trend = "Undetermined (not enough data to calculate one or both SMAs)"

    # ── RSI commentary ─────────────────────────────────────────────────────────
    rsi     = m["rsi"]
    rsi_str = f"{rsi}" if rsi is not None else "N/A"
    if rsi is not None:
        if rsi >= 70:   rsi_str += " → overbought (selling pressure may increase)"
        elif rsi <= 30: rsi_str += " → oversold (buying opportunity may emerge)"

    # ── Volume commentary ──────────────────────────────────────────────────────
    vr = m["volume_ratio"]
    if vr is not None:
        vol_str = f"{vr}x the 20-day average"
        if vr >= 2.0: vol_str += " (unusually high — possible institutional activity)"
        elif vr <= 0.5: vol_str += " (unusually low — low conviction move)"
    else:
        vol_str = "N/A"

    # ── Market Pulse section ───────────────────────────────────────────────────
    pulse_lines = []
    for name, data in pulse.items():
        if data["change_pct"] is not None:
            arrow = "▲" if data["change_pct"] >= 0 else "▼"
            pulse_lines.append(
                f"  {name}: ${data['price']:,.2f}  ({arrow} {abs(data['change_pct'])}%)"
            )
        else:
            pulse_lines.append(f"  {name}: data unavailable")

    def _fmt(val, prefix="$"):
        return f"{prefix}{val}" if val is not None else "N/A"

    return f"""
=== GLOBAL MARKET PULSE (live) ===
{chr(10).join(pulse_lines)}

=== STOCK ANALYSIS: {m['ticker']} ===
Current Price  : ${price:,.2f}  ({"▲" if chg >= 0 else "▼"} {abs(chg)}% today)
Previous Close : ${m['prev_close']:,.2f}
Period High    : ${m['period_high']:,.2f}   |   Period Low: ${m['period_low']:,.2f}

Technical Indicators:
  20-Day SMA    : {_fmt(sma20)}
  50-Day SMA    : {_fmt(sma50)}
  Bollinger Band: Lower {_fmt(m['bb_lower'])}  →  Upper {_fmt(m['bb_upper'])}
  RSI (14-day)  : {rsi_str}
  Volume Ratio  : {vol_str}

Trend Signal   : {trend}
""".strip()


# ════════════════════════════════════════════════════════════════════════════════
# CHART LAYER  — all return plotly Figure objects for st.plotly_chart()
# ════════════════════════════════════════════════════════════════════════════════

# Shared layout dict applied to every chart for visual consistency
_LAYOUT_BASE = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=BG,
    font=dict(family="Inter, sans-serif", color=TEXT, size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=GRID, borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor=GRID, zerolinecolor=GRID,
        showspikes=True, spikecolor=MUTED, spikethickness=1,
    ),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    hovermode="x unified",
)


def chart_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Candlestick OHLC chart with SMA overlays and Bollinger Band shading.

    Layers (bottom → top):
      1. Bollinger Band shaded fill (±2σ orange tint)
      2. Bollinger Band edge dotted lines
      3. Candlestick bars (green up / red down)
      4. SMA 20 solid orange line
      5. SMA 50 dashed gold line
    """
    fig = go.Figure()

    # ── 1 & 2. Bollinger Bands ────────────────────────────────────────────────
    bb_mask = df["BB_upper"].notna() & df["BB_lower"].notna()
    if bb_mask.any():
        bb    = df[bb_mask]
        x_rev = bb.index.to_series()[::-1]

        # Filled region (upper curve down, lower curve up = closed polygon)
        fig.add_trace(go.Scatter(
            x=pd.concat([bb.index.to_series(), x_rev]),
            y=pd.concat([bb["BB_upper"], bb["BB_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,128,0,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bollinger Bands",
            hoverinfo="skip",
        ))
        for col, label in [("BB_upper", "BB Upper"), ("BB_lower", "BB Lower")]:
            fig.add_trace(go.Scatter(
                x=bb.index, y=bb[col],
                line=dict(color="rgba(255,128,0,0.35)", width=1, dash="dot"),
                name=label, showlegend=False,
                hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>",
            ))

    # ── 3. Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED,   width=1), fillcolor=RED),
        name=ticker,
    ))

    # ── 4 & 5. SMA lines ──────────────────────────────────────────────────────
    for col, color, dash, label in [
        ("SMA_20", ORANGE,   "solid", "SMA 20"),
        ("SMA_50", "#FFB347", "dash",  "SMA 50"),
    ]:
        if df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                line=dict(color=color, width=1.5, dash=dash),
                name=label,
                hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>",
            ))

    layout = {**_LAYOUT_BASE}
    layout["title"]  = dict(text=f"{ticker} — Price · SMA · Bollinger Bands", font=dict(size=13))
    layout["yaxis"]  = {**layout["yaxis"], "title": "Price (USD)", "tickprefix": "$"}
    layout["xaxis"]  = {**layout["xaxis"], "rangeslider": dict(visible=False)}
    fig.update_layout(**layout)
    return fig


def chart_rsi(df: pd.DataFrame) -> go.Figure:
    """
    14-day RSI line chart with overbought/oversold reference zones.
    Shows a "not enough data" annotation if RSI cannot be computed.
    """
    fig = go.Figure()
    rsi = df["RSI"].dropna()

    if rsi.empty:
        fig.add_annotation(
            text="Not enough data to calculate RSI (need 14+ trading days)",
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(color=MUTED), showarrow=False,
        )
    else:
        # Shaded zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.09)",  line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(34,197,94,0.09)",  line_width=0)

        # Reference lines at 30 / 50 / 70
        fig.add_hline(y=70, line=dict(color=RED,   width=1, dash="dash"),
                      annotation_text="Overbought 70", annotation_position="top right",
                      annotation_font=dict(color=RED, size=9))
        fig.add_hline(y=30, line=dict(color=GREEN, width=1, dash="dash"),
                      annotation_text="Oversold 30",   annotation_position="bottom right",
                      annotation_font=dict(color=GREEN, size=9))
        fig.add_hline(y=50, line=dict(color=GRID, width=1))

        fig.add_trace(go.Scatter(
            x=rsi.index, y=rsi,
            line=dict(color=ORANGE, width=2),
            name="RSI",
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ))

    layout = {**_LAYOUT_BASE, "showlegend": False}
    layout["title"] = dict(text="RSI — Relative Strength Index (14-day)", font=dict(size=13))
    layout["yaxis"] = {**layout["yaxis"], "range": [0, 100], "title": "RSI"}
    fig.update_layout(**layout)
    return fig


def chart_volume(df: pd.DataFrame) -> go.Figure:
    """
    Daily volume bar chart:
      • Green bars = up-day (Close ≥ Open)
      • Red bars   = down-day
      • Orange line = 20-day average volume (context for spike detection)
    """
    fig = go.Figure()

    # Color each bar by whether it was an up or down day
    bar_colors = [
        GREEN if close >= open_ else RED
        for close, open_ in zip(df["Close"], df["Open"])
    ]

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=bar_colors,
        name="Volume", opacity=0.8,
        hovertemplate="Volume: %{y:,.0f}<extra></extra>",
    ))

    if df["Volume_SMA20"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA20"],
            line=dict(color=ORANGE, width=2),
            name="20-Day Avg",
            hovertemplate="Avg: %{y:,.0f}<extra></extra>",
        ))

    layout = {**_LAYOUT_BASE}
    layout["title"]  = dict(text="Volume vs 20-Day Average", font=dict(size=13))
    layout["yaxis"]  = {**layout["yaxis"], "title": "Volume", "tickformat": ",.0f"}
    layout["bargap"] = 0.15
    fig.update_layout(**layout)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# GROQ CLIENT & CHAT  — client is a singleton cached for the whole session
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_groq_client() -> Groq | None:
    """
    Return an authenticated Groq client, or None if GROQ_API_KEY is missing.
    @st.cache_resource ensures only one Groq connection is created per session,
    regardless of how many times the script reruns.
    """
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None


def get_chat_response(client: Groq, history: list[dict], context: str) -> str:
    """
    Send the conversation history to Groq with live market data injected as
    the system message. Rebuilding the system prompt on every call means that
    switching tickers refreshes the LLM's ground truth without clearing history.

    Args:
        client:  Authenticated Groq client (from get_groq_client).
        history: List of {"role": "user"|"assistant", "content": str} dicts,
                 maintained in st.session_state.chat_history.
        context: Plain-text digest from build_context_digest() — live numbers.

    Returns:
        The assistant's reply as a plain string.
    """
    system_msg = f"""You are an expert stock market analyst embedded in a live trading dashboard.

The following is real-time data currently displayed on the dashboard:

{context}

Your guidelines:
- Ground EVERY numerical answer in the data above. Never fabricate prices or indicators.
- Be concise and direct. Use markdown bullet points for multi-part answers.
- Explain technical terms (SMA, RSI, Bollinger Bands) clearly when asked.
- If asked about news, earnings, or anything outside the provided data, say so explicitly.
- Keep a professional but approachable tone suited to both beginners and experienced traders.
"""
    messages = [{"role": "system", "content": system_msg}, *history]

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.4,   # factual grounding > creativity for finance
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"⚠️ Could not reach the AI service. Error: {exc}"


# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ════════════════════════════════════════════════════════════════════════════════

_SESSION_DEFAULTS = {
    "chat_history":  [],     # list of {role, content} dicts
    "last_ticker":   None,   # tracks ticker switches so chat auto-clears
    "context_digest": "",    # latest LLM context string
}
for _k, _v in _SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — stock picker & period selector
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📈 Stock Dashboard")
    st.markdown("---")

    # Dropdown instead of free-text — works instantly for any recruiter
    ticker = st.selectbox(
        "Select Stock",
        options=WATCHLIST,
        index=0,
        help="Choose from the top-10 watchlist",
    )

    period = st.selectbox(
        "Time Period",
        options=PERIODS,
        index=2,   # default: 6mo
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px; color:{MUTED};'>
    <b style='color:{ORANGE}'>Indicators</b><br>
    SMA 20/50 · Bollinger Bands<br>RSI (14d) · Volume Ratio
    <br><br>
    <b style='color:{ORANGE}'>AI Chatbot</b><br>
    Groq · llama-3.3-70b-versatile<br>
    Context: live metrics + market pulse
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# DATA FETCHING  (cached — runs only when ticker/period actually changes)
# ════════════════════════════════════════════════════════════════════════════════

# Market pulse is independent of the ticker — fetch once and share
pulse = fetch_market_pulse()

# Stock data + indicators for the selected ticker
df, fetch_error = fetch_and_compute(ticker, period)

if df is not None:
    metrics = extract_metrics(ticker, df)
    context = build_context_digest(metrics, pulse)
    st.session_state.context_digest = context

    # Auto-clear chat when user switches to a different stock
    if st.session_state.last_ticker != ticker:
        st.session_state.chat_history = []
        st.session_state.last_ticker  = ticker


# ════════════════════════════════════════════════════════════════════════════════
# GLOBAL MARKET PULSE HEADER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("<div class='section-label'>Global Market Pulse</div>", unsafe_allow_html=True)

pulse_cols = st.columns(len(MARKET_INDICES))
for col, (name, data) in zip(pulse_cols, pulse.items()):
    with col:
        if data["price"] is not None:
            chg_pct = data["change_pct"]
            css_cls = "pulse-up" if chg_pct >= 0 else "pulse-down"
            arrow   = "▲" if chg_pct >= 0 else "▼"
            st.markdown(f"""
            <div class='pulse-card'>
                <div class='pulse-name'>{name}</div>
                <div class='pulse-price'>${data['price']:,.2f}</div>
                <div class='{css_cls}'>{arrow} {abs(chg_pct)}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pulse-card'>
                <div class='pulse-name'>{name}</div>
                <div class='pulse-price' style='font-size:14px; color:{MUTED};'>Unavailable</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT — left 60% charts  |  right 40% chatbot
# ════════════════════════════════════════════════════════════════════════════════

col_charts, col_chat = st.columns([0.60, 0.40], gap="large")


# ── LEFT: Technical Charts ────────────────────────────────────────────────────

with col_charts:

    if fetch_error:
        st.error(f"**Data error:** {fetch_error}")

    elif df is not None:

        # ── KPI snapshot row ──────────────────────────────────────────────────
        st.markdown(
            f"<div class='section-label'>{ticker} — Live Snapshot</div>",
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            chg = metrics["change_pct"]
            st.metric("Price", f"${metrics['current_price']:,.2f}", f"{chg:+.2f}%")

        with k2:
            rsi = metrics["rsi"]
            if rsi is not None:
                rsi_label = "Overbought ⚠" if rsi >= 70 else ("Oversold ⚠" if rsi <= 30 else "Neutral zone")
            else:
                rsi_label = "N/A"
            st.metric("RSI (14d)", f"{rsi:.1f}" if rsi is not None else "N/A", rsi_label)

        with k3:
            sma20 = metrics["sma_20"]
            if sma20 is not None:
                pos = "Above ↑" if metrics["current_price"] > sma20 else "Below ↓"
                st.metric("SMA 20", f"${sma20:,.2f}", pos)
            else:
                st.metric("SMA 20", "N/A", "Insufficient data")

        with k4:
            vr = metrics["volume_ratio"]
            if vr is not None:
                vr_status = "High 🔥" if vr >= 2 else ("Low" if vr <= 0.5 else "Normal")
                st.metric("Volume", vr_status, f"{vr:.2f}x avg")
            else:
                st.metric("Volume", "N/A", "")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Three stacked charts ──────────────────────────────────────────────
        st.plotly_chart(chart_price(df, ticker),  use_container_width=True)
        st.plotly_chart(chart_rsi(df),             use_container_width=True)
        st.plotly_chart(chart_volume(df),           use_container_width=True)


# ── RIGHT: AI Chatbot ─────────────────────────────────────────────────────────

with col_chat:

    st.markdown(
        f"<div class='section-label'>AI Market Analyst · {ticker}</div>",
        unsafe_allow_html=True,
    )

    groq_client = get_groq_client()

    # ── Guard: no API key ─────────────────────────────────────────────────────
    if groq_client is None:
        st.warning(
            "**Groq API key not found.**\n\n"
            "Add `GROQ_API_KEY=your_key` to your `.env` file and restart.\n\n"
            "Get a free key at [console.groq.com](https://console.groq.com).",
            icon="⚠️",
        )

    # ── Guard: data fetch failed ──────────────────────────────────────────────
    elif df is None:
        st.info("Select a valid stock to activate the chatbot.", icon="💡")

    # ── Active chat interface ─────────────────────────────────────────────────
    else:
        st.markdown(
            f"<div style='font-size:11px; color:{MUTED}; margin-bottom:10px;'>"
            f"Model: <b>llama-3.3-70b-versatile</b> · "
            f"Context: live <b style='color:{ORANGE}'>{ticker}</b> metrics + market pulse"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Scrollable message history container
        chat_box = st.container(height=490)
        with chat_box:
            if not st.session_state.chat_history:
                # Empty-state placeholder with starter questions
                st.markdown(
                    f"<div style='color:{MUTED}; font-size:13px; text-align:center; margin-top:48px;'>"
                    f"Ask me anything about <b style='color:{ORANGE}'>{ticker}</b> or the market.<br><br>"
                    f"<span style='color:#6B7280;'><i>Starter questions:</i></span><br>"
                    f"• Is {ticker} in a bullish or bearish trend?<br>"
                    f"• What does the current RSI indicate?<br>"
                    f"• How is the broader market performing today?<br>"
                    f"• Are Bollinger Bands showing high volatility?"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            # Render each message using native Streamlit chat bubbles
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Sticky chat input at the bottom of the right column
        user_input = st.chat_input(f"Ask about {ticker} or the market…")

        if user_input:
            # 1. Record user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # 2. Call Groq with full history + live context digest
            with st.spinner("Analyzing…"):
                reply = get_chat_response(
                    client=groq_client,
                    history=st.session_state.chat_history,
                    context=st.session_state.context_digest,
                )

            # 3. Record assistant reply
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            # 4. Rerun so the new messages render in the chat container
            st.rerun()

        # Clear chat button — only shown when there's history to clear
        if st.session_state.chat_history:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
