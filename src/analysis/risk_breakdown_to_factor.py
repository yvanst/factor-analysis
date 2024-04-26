import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.benchmark import Benchmark
from src.fund_universe import ISHARE_SECTOR_ETF_TICKER
from src.market import Market
from src.security_symbol import SecuritySedol


class RiskBreakdownToFactor:
    def __init__(self, portforlio, benchmark) -> None:
        self.portfolio = portforlio
        self.portforlio_performance = portforlio.value_book.select("date", "value")
        self.benchmark = benchmark
        self.month_df = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date") >= self.portfolio.start_date)
            .filter(pl.col("date") <= self.portfolio.end_date)
            .select("date")
            .unique()
            .collect()
            .sort(pl.col("date"))
        )

    def get_stock_beta_against_factors(self):
        index_tickers = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date") >= self.portfolio.start_date)
            .filter(pl.col("date") <= self.portfolio.end_date)
            .select(["sedol7", "date", "weight"])
            .collect()
        )
        security_df = index_tickers.rename({"sedol7": "security"}).filter(
            pl.col("weight") > 0
        )
        ticker_return_df = self.get_securities_monthly_return_df(security_df)
        ticker_return_df = ticker_return_df.pivot(
            index="date", columns="security", values="current_1mo_return"
        ).sort(pl.col("date"))
        factor_orth = self.get_orthogonalized_factor()
        return

    def get_orthogonalized_factor(self):
        benchmark_return_df: pl.DataFrame = (
            self.benchmark.get_performance_return_from_month_start()
        )
        benchmark_return_df = benchmark_return_df.group_by(
            pl.col("date").dt.month_start()
        ).agg(((pl.col("return") + 1).product() - 1).alias("return"))
        benchmark_return_df = (
            benchmark_return_df.with_columns(pl.lit(1).alias("intercept"))
            .rename({"return": "benchmark_return"})
            .sort(pl.col("date"))
        )
        sector_return_df = self.get_sector_monthly_return_df().sort(pl.col("date"))
        factor_before = pl.concat(
            [
                benchmark_return_df.select("intercept", "benchmark_return"),
                sector_return_df.select(pl.all().exclude("date")),
            ],
            how="horizontal",
        )
        factor_orth, beta = self.regress(
            benchmark_return_df.select("intercept", "benchmark_return"),
            sector_return_df.select(pl.all().exclude("date")),
        )
        # heatmap_before = sns.heatmap(
        #     factor_before.corr(),
        #     annot=True,
        #     xticklabels=factor_orth.columns,
        #     yticklabels=factor_orth.columns,
        # )
        # heatmap_before.set_title("factor correlation before orthogonalization")
        # heatmap = sns.heatmap(
        #     factor_orth.corr(),
        #     annot=True,
        #     xticklabels=factor_orth.columns,
        #     yticklabels=factor_orth.columns,
        # )
        # heatmap.set_title("factor correlation after orthogonalization")
        return factor_orth

    def get_sector_monthly_return_df(self):
        start_date = datetime.date(
            self.portfolio.start_date.year, self.portfolio.start_date.month, 1
        )
        market = Market(
            ISHARE_SECTOR_ETF_TICKER,
            start_date,
            self.portfolio.end_date,
        )
        sector_df_list = []
        for sector_ticker in ISHARE_SECTOR_ETF_TICKER:
            security_df = pl.DataFrame(
                {
                    "date": self.month_df,
                    "sector": sector_ticker.sector,
                    "ticker": sector_ticker.ticker,
                }
            )
            security_df = security_df.with_columns(
                pl.col("date").dt.month_start().alias("current_start_date"),
                pl.col("date").dt.month_end().alias("current_end_date"),
            )

            security_df = security_df.with_columns(
                pl.struct(["ticker", "current_start_date", "current_end_date"])
                .map_elements(
                    lambda x: market.query_ticker_range_return(
                        SecurityTicker(x["ticker"]),
                        x["current_start_date"],
                        x["current_end_date"],
                    ),
                    return_dtype=pl.Float32,
                )
                .alias("current_1mo_return")
            )
            sector_df_list.append(security_df)
        sector_performance_df = pl.concat(sector_df_list)
        sector_performance_df = sector_performance_df.pivot(
            index="date", columns="sector", values="current_1mo_return"
        )
        return sector_performance_df

    def get_securities_monthly_return_df(self, security_df: pl.DataFrame):
        start_date = datetime.date(
            self.portfolio.start_date.year, self.portfolio.start_date.month, 1
        )
        market = Market(
            [SecuritySedol(security_df.get_column("security").item(0))],
            start_date,
            self.portfolio.end_date,
        )
        security_df = security_df.with_columns(
            pl.col("date").dt.month_start().alias("current_start_date"),
            pl.col("date").dt.month_end().alias("current_end_date"),
        )

        security_df = security_df.with_columns(
            pl.struct(["security", "current_start_date", "current_end_date"])
            .map_elements(
                lambda x: market.query_sedol_range_return(
                    x["security"],
                    x["current_start_date"],
                    x["current_end_date"],
                ),
                return_dtype=pl.Float32,
            )
            .alias("current_1mo_return")
        )
        return security_df

    def regress(self, X: pl.DataFrame, Y: pl.DataFrame):
        X_inv = np.linalg.pinv(X.to_numpy())
        B = X_inv.dot(Y.to_numpy())
        residual = Y.to_numpy() - X.to_numpy().dot(B)
        residual = pl.from_numpy(residual, schema=Y.schema)
        factors = pl.concat([X, residual], how="horizontal")
        return factors, pl.DataFrame(B, schema=Y.schema)


if __name__ == "__main__":
    import datetime

    from src.backtest import BackTest
    from src.benchmark import Benchmark
    from src.factor.roe import RoeFactor
    from src.fund_universe import SECURITY_SEDOL
    from src.market import Market
    from src.portfolio import Portfolio
    from src.rebalance import Rebalance
    from src.security_symbol import SecurityTicker

    cfg = pl.Config()
    cfg.set_tbl_rows(100)
    cfg.set_tbl_cols(100)

    start_date = datetime.date(2012, 12, 31)
    end_date = datetime.date(2013, 10, 31)
    security_universe = SECURITY_SEDOL
    rebalance_period = 1
    rebalance_interval = "1mo"
    Factor = RoeFactor
    market = Market(security_universe, start_date, end_date)
    long_factor = Factor(security_universe, "long")
    long_portfolio = Portfolio(100.0, start_date, end_date)
    long_factor.set_portfolio_at_start(long_portfolio)

    rebalance = Rebalance(
        rebalance_period, long_portfolio, long_factor, rebalance_interval
    )

    backtest = BackTest(long_portfolio, market, rebalance)
    backtest.run()
    benchmark = Benchmark(SecurityTicker("^SPX", "index"), start_date, end_date)
    # benchmark_performance = benchmark.get_performance()

    df = RiskBreakdownToFactor(
        long_portfolio, benchmark
    ).get_stock_beta_against_factors()
