import os
import pandas as pd
import yfinance as yf


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def fetch_prices(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for *tickers* between *start* and *end*.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame, one column per ticker (Adj Close).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = "_".join(sorted(tickers)) + f"_{start}_{end or 'today'}.csv"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if use_cache and os.path.exists(cache_path):
        print(f"[fetcher] Loading from cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[fetcher] Downloading {tickers} …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns when len(tickers) > 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index.name = "Date"
    prices.dropna(how="all", inplace=True)

    if use_cache:
        prices.to_csv(cache_path)
        print(f"[fetcher] Cached to {cache_path}")

    return prices


def load_sample_portfolio() -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Convenience: return prices + weights for a ready-made 5-stock portfolio.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.20, "JPM": 0.15, "XOM": 0.15}
    prices = fetch_prices(tickers, start="2019-01-01")
    return prices, weights
