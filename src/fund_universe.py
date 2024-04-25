import polars as pl

from src.security_symbol import SecuritySedol, SecurityTicker

sedol_list = (
    pl.read_parquet("parquet/base/us_sector_weight.parquet")
    .filter(pl.col("date").dt.year() == 2020)
    .filter(pl.col("date").dt.month() == 1)
    .filter(pl.col("weight") > 0)
    .select(pl.col("sedol7").unique())
    .get_column("sedol7")
    .to_list()
)


SECURITY_SEDOL = [SecuritySedol(id) for id in sedol_list]
ishare_ticker_sector_etf = {
    "Consumer Discretionary": "IYC",
    "Energy": "IYE",
    "Real Estate": "IYR",
    "Materials": "IYM",
    "Utilities": "IDU",
    "Information Technology": "IYW",
    "Communication Services": "IYZ",
    "Health Care": "IYH",
    "Industrials": "IYJ",
    "Consumer Staples": "IYK",
    "Financials": "IYF",
}

# market weight ETF
ISHARE_SECTOR_ETF_TICKER = [
    SecurityTicker(v, k) for k, v in ishare_ticker_sector_etf.items()
]
