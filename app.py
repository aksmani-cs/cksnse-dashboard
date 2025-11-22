import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="NSE Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 NSE Market Dashboard with Option Chain (Yahoo Finance)")
st.caption(
    "Educational / personal use only – data from Yahoo Finance (delayed), "
    "not for live trading decisions."
)

# ---------------------- HELPER FUNCTIONS ---------------------- #

@st.cache_data(ttl=300)
def get_yf_data(tickers, period="3mo", interval="1d"):
    if not tickers:
        return None
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
    )
    return data

def parse_watchlist(text):
    if not text:
        return []
    syms = [s.strip().upper() for s in text.replace("\n", ",").split(",") if s.strip()]
    yf_syms = [s + ".NS" for s in syms]   # NSE stocks on Yahoo: RELIANCE.NS, TCS.NS, etc.:contentReference[oaicite:1]{index=1}
    return syms, yf_syms

def compute_trending_metrics(raw_symbols, yf_symbols, period="1mo"):
    if not yf_symbols:
        return pd.DataFrame()

    data = get_yf_data(yf_symbols, period=period, interval="1d")
    if data is None or data.empty:
        return pd.DataFrame()

    closing = {}
    volume = {}

    # yfinance shape differs for single vs multi-ticker
    if isinstance(data.columns, pd.MultiIndex):
        for s, yf_s in zip(raw_symbols, yf_symbols):
            if (yf_s, "Close") in data.columns:
                closing[s] = data[(yf_s, "Close")]
                volume[s] = data[(yf_s, "Volume")]
    else:
        s = raw_symbols[0]
        closing[s] = data["Close"]
        volume[s] = data["Volume"]

    close_df = pd.DataFrame(closing)
    vol_df = pd.DataFrame(volume)

    latest = close_df.iloc[-1]
    prev = close_df.iloc[-2] if len(close_df) > 1 else close_df.iloc[-1]
    week_back_idx = max(0, len(close_df) - 6)
    month_back_idx = max(0, len(close_df) - 21)

    week_back = close_df.iloc[week_back_idx]
    month_back = close_df.iloc[month_back_idx]

    latest_vol = vol_df.iloc[-1]
    avg_vol_20 = vol_df.tail(20).mean()

    df = pd.DataFrame(index=latest.index)
    df["Last Price"] = latest
    df["1D %"] = (latest - prev) / prev * 100
    df["5D %"] = (latest - week_back) / week_back * 100
    df["1M %"] = (latest - month_back) / month_back * 100
    df["Latest Vol"] = latest_vol
    df["Avg Vol 20D"] = avg_vol_20
    df["Vol Spike"] = df["Latest Vol"] / df["Avg Vol 20D"]

    df["Trend Score"] = (
        df["1D %"] * 0.4 +
        df["5D %"] * 0.3 +
        df["1M %"] * 0.2 +
        (df["Vol Spike"] - 1).fillna(0) * 5
    )

    return df.sort_values("Trend Score", ascending=False)

def map_to_yf_option_symbol(sym: str) -> str:
    """
    Map a user-friendly NSE symbol to Yahoo Finance symbol for options.
    NIFTY -> ^NSEI, BANKNIFTY -> ^NSEBANK, stocks -> SYMBOL.NS
    """
    s = sym.upper().strip()
    if s in ["NIFTY", "NIFTY50", "NIFTY 50"]:
        return "^NSEI"       # NIFTY 50 index on Yahoo:contentReference[oaicite:2]{index=2}
    if s in ["BANKNIFTY", "NIFTYBANK", "BANK NIFTY"]:
        return "^NSEBANK"    # NIFTY BANK index on Yahoo:contentReference[oaicite:3]{index=3}
    if s.startswith("^"):
        return s
    if not s.endswith(".NS"):
        return s + ".NS"
    return s

@st.cache_data(ttl=120)
def get_option_chain_from_yf(yf_symbol: str):
    """
    Use yfinance to get available expiries and option chain for a symbol.
    """
    ticker = yf.Ticker(yf_symbol)
    expiries = list(ticker.options)  # list of expiry dates as strings:contentReference[oaicite:4]{index=4}
    return ticker, expiries

def build_option_chain_df(chain):
    """
    Convert yfinance option_chain result into a combined CE+PE DataFrame.
    """
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    # Keep only relevant columns if present
    ce_cols = ["strike", "openInterest", "lastPrice", "volume"]
    pe_cols = ["strike", "openInterest", "lastPrice", "volume"]

    calls = calls[[c for c in ce_cols if c in calls.columns]]
    puts = puts[[c for c in pe_cols if c in puts.columns]]

    df = pd.merge(
        calls,
        puts,
        on="strike",
        how="outer",
        suffixes=("_CE", "_PE"),
    ).sort_values("strike")

    df.rename(
        columns={
            "strike": "strikePrice",
            "openInterest_CE": "CE_OI",
            "lastPrice_CE": "CE_LTP",
            "volume_CE": "CE_Volume",
            "openInterest_PE": "PE_OI",
            "lastPrice_PE": "PE_LTP",
            "volume_PE": "PE_Volume",
        },
        inplace=True,
    )

    # Fill NaNs with 0 for numeric columns
    for col in ["CE_OI", "CE_LTP", "CE_Volume", "PE_OI", "PE_LTP", "PE_Volume"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

# ---------------------- SIDEBAR ---------------------- #
st.sidebar.header("⚙️ Settings")

default_watchlist = "RELIANCE,TCS,INFY,ICICIBANK,HDFCBANK,SBIN,AXISBANK,LT"
watchlist_text = st.sidebar.text_area(
    "Watchlist symbols (NSE, comma separated)",
    value=default_watchlist,
    height=100,
    help="Example: RELIANCE,TCS,INFY,SBIN",
)

trend_period = st.sidebar.selectbox(
    "Trend lookback for stocks",
    options=["1mo", "3mo", "6mo"],
    index=0,
)

st.sidebar.info(
    "Once this app is deployed, you can open it from any laptop/phone.\n"
    "Just update your watchlist here."
)

# ---------------------- TABS ---------------------- #
tab1, tab2, tab3 = st.tabs(
    ["📈 Market & Stocks", "📊 Gainers / Losers", "🧮 Option Chain"]
)

# ---------------------- TAB 1: MARKET & TRENDING ---------------------- #
with tab1:
    st.subheader("Market Overview & Trending Stocks")

    raw_syms, yf_syms = parse_watchlist(watchlist_text)

    cols = st.columns(2)

    with cols[0]:
        st.markdown("#### 🔍 Trending in Your Watchlist")
        if yf_syms:
            trend_df = compute_trending_metrics(raw_syms, yf_syms, period=trend_period)
            if not trend_df.empty:
                st.dataframe(
                    trend_df.round(2),
                    use_container_width=True,
                )

                top_n = st.slider(
                    "Show top N by Trend Score",
                    3,
                    min(15, len(trend_df)),
                    10,
                )
                chart_df = (
                    trend_df.head(top_n)
                    .reset_index()
                    .rename(columns={"index": "Symbol"})
                )
                fig = px.bar(
                    chart_df,
                    x="Symbol",
                    y="Trend Score",
                    hover_data=["1D %", "5D %", "1M %", "Vol Spike"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Could not fetch data for your watchlist. "
                    "Try different symbols or check connectivity."
                )
        else:
            st.info("Add some symbols in the sidebar to see trending stocks.")

    with cols[1]:
        st.markdown("#### ℹ️ Notes")
        st.write(
            """
            - **Trend Score** combines:
              - 1D / 5D / 1M % returns  
              - 20D volume spike  
            - Higher score ≈ more momentum + activity.  
            - For educational analysis only, **not** a trading signal.
            """
        )

# ---------------------- TAB 2: GAINERS / LOSERS ---------------------- #
with tab2:
    st.subheader("Top Gainers / Losers in Your Watchlist")

    raw_syms, yf_syms = parse_watchlist(watchlist_text)
    if yf_syms:
        trend_df = compute_trending_metrics(raw_syms, yf_syms, period=trend_period)
        if not trend_df.empty:
            today_change = trend_df.sort_values("1D %", ascending=False)

            col_g, col_l = st.columns(2)

            with col_g:
                st.markdown("##### 🔼 Top Gainers (1D %)")
                gainers = today_change.head(10).copy()
                st.dataframe(
                    gainers[["Last Price", "1D %", "5D %", "1M %", "Vol Spike"]].round(2)
                )

            with col_l:
                st.markdown("##### 🔽 Top Losers (1D %)")
                losers = (
                    today_change.tail(10)
                    .sort_values("1D %")
                )
                st.dataframe(
                    losers[["Last Price", "1D %", "5D %", "1M %", "Vol Spike"]].round(2)
                )
        else:
            st.warning(
                "Unable to compute gainers/losers because price data is missing."
            )
    else:
        st.info("Add some symbols in the sidebar to see gainers/losers.")

# ---------------------- TAB 3: OPTION CHAIN (YAHOO) ---------------------- #
with tab3:
    st.subheader("Option Chain Analysis (via Yahoo Finance)")

    st.write(
        """
        Type an **index** (e.g. `NIFTY`, `BANKNIFTY`) or **stock** (e.g. `RELIANCE`, `TCS`).  
        Data comes from Yahoo Finance options API (delayed).:contentReference[oaicite:5]{index=5}
        """
    )

    oc_symbol = st.text_input("Underlying symbol", value="NIFTY").upper().strip()

    if oc_symbol:
        try:
            yf_symbol = map_to_yf_option_symbol(oc_symbol)
            st.caption(f"Using Yahoo symbol: `{yf_symbol}`")

            ticker, expiries = get_option_chain_from_yf(yf_symbol)

            if not expiries:
                st.warning("No option expiries found for this symbol.")
            else:
                sel_expiry = st.selectbox("Select expiry", options=expiries, index=0)

                chain = ticker.option_chain(sel_expiry)
                df_oc = build_option_chain_df(chain)

                if df_oc.empty:
                    st.warning("No option data available for this expiry.")
                else:
                    # PCR using OI
                    ce_oi_sum = df_oc.get("CE_OI", pd.Series(dtype=float)).sum()
                    pe_oi_sum = df_oc.get("PE_OI", pd.Series(dtype=float)).sum()
                    pcr_val = (
                        pe_oi_sum / ce_oi_sum if ce_oi_sum not in [0, np.nan] else np.nan
                    )
                    if not np.isnan(pcr_val):
                        st.metric("Put-Call Ratio (OI)", f"{pcr_val:.2f}")
                    else:
                        st.caption("PCR not available (missing OI data).")

                    st.markdown("#### Option Chain Table")
                    st.dataframe(
                        df_oc.set_index("strikePrice").round(2),
                        use_container_width=True,
                    )

                    # OI charts
                    st.markdown("#### Open Interest by Strike")

                    col1, col2 = st.columns(2)

                    with col1:
                        if "CE_OI" in df_oc.columns:
                            fig_ce = go.Figure()
                            fig_ce.add_trace(
                                go.Bar(
                                    x=df_oc["strikePrice"],
                                    y=df_oc["CE_OI"],
                                    name="CE OI",
                                )
                            )
                            fig_ce.update_layout(
                                xaxis_title="Strike Price",
                                yaxis_title="Call OI",
                            )
                            st.plotly_chart(fig_ce, use_container_width=True)
                        else:
                            st.caption("Call OI not available in data.")

                    with col2:
                        if "PE_OI" in df_oc.columns:
                            fig_pe = go.Figure()
                            fig_pe.add_trace(
                                go.Bar(
                                    x=df_oc["strikePrice"],
                                    y=df_oc["PE_OI"],
                                    name="PE OI",
                                )
                            )
                            fig_pe.update_layout(
                                xaxis_title="Strike Price",
                                yaxis_title="Put OI",
                            )
                            st.plotly_chart(fig_pe, use_container_width=True)
                        else:
                            st.caption("Put OI not available in data.")

                    # Volume / sentiment-style view
                    st.markdown("#### Volume (Liquidity) by Strike")
                    if "CE_Volume" in df_oc.columns or "PE_Volume" in df_oc.columns:
                        fig_vol = go.Figure()
                        if "CE_Volume" in df_oc.columns:
                            fig_vol.add_trace(
                                go.Bar(
                                    x=df_oc["strikePrice"],
                                    y=df_oc["CE_Volume"],
                                    name="CE Volume",
                                )
                            )
                        if "PE_Volume" in df_oc.columns:
                            fig_vol.add_trace(
                                go.Bar(
                                    x=df_oc["strikePrice"],
                                    y=df_oc["PE_Volume"],
                                    name="PE Volume",
                                )
                            )
                        fig_vol.update_layout(
                            barmode="group",
                            xaxis_title="Strike Price",
                            yaxis_title="Volume",
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)

                    st.markdown(
                        """
                        **How to read this:**
                        - Strikes with high **Call OI** → potential **resistance** zones  
                        - Strikes with high **Put OI** → potential **support** zones  
                        - **PCR (OI)** near 1 is neutral; very low/high can signal extreme sentiment.  
                        - Remember: Yahoo data is delayed and may not match live NSE ticks exactly.
                        """
                    )
        except Exception as e:
            st.error(
                "Could not fetch option chain (Yahoo Finance error or rate limit). "
                "Try again after some time or with a different symbol."
            )
            st.exception(e)
