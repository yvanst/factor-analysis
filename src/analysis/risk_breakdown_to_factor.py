import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.benchmark import Benchmark
from src.fund_universe import ISHARE_SECTOR_ETF_TICKER
from src.market import Market
from src.security_symbol import SecuritySedol
from src.analysis.risk_breakdown_with_weight import RiskBreakdownWithWeight


class RiskBreakdownToFactor:
    def __init__(self, portforlio, benchmark) -> None:
        self.portfolio = portforlio
        self.portforlio_performance = portforlio.value_book.select("date", "value")
        self.holding_snapshot = portforlio.holding_snapshots[
            datetime.date(portforlio.end_date.year, portforlio.end_date.month, 1)
        ]
        self.benchmark = benchmark
        # TODO: filter on 60
        self.month_range = 60
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
        security_df = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date") >= self.portfolio.start_date)
            .filter(pl.col("date") <= self.portfolio.end_date)
            .select(["sedol7", "date", "weight"])
            .collect()
            .rename({"sedol7": "security"})
            .filter(pl.col("weight") > 0)
        )
        ticker_return_df = self.get_securities_monthly_return_df(security_df)
        ticker_return_df = ticker_return_df.pivot(
            index="date", columns="security", values="current_1mo_return"
        ).sort(pl.col("date"))
        ticker_cols = list(
            filter(lambda x: x != "date", sorted(ticker_return_df.columns))
        )
        self.ticker_return_df = ticker_return_df
        factor_orth = self.get_orthogonalized_factor(use_intercept=False)
        residual, beta = self.regress(factor_orth, ticker_return_df.select(ticker_cols))
        exlcude_cols = [
            "benchmark_return",
            "Consumer Discretionary",
            "Energy",
            "Real Estate",
            "Materials",
            "Utilities",
            "Information Technology",
            "Communication Services",
            "Health Care",
            "Industrials",
            "Consumer Staples",
            "Financials",
        ]
        idiosyncratic_variance = residual.drop(exlcude_cols).select(
            pl.all().pow(2).sum() / (self.month_range - len(exlcude_cols))
        )
        idiosyncratic_variance = pl.DataFrame(
            np.diag(idiosyncratic_variance.to_numpy().reshape(-1)),
            schema=idiosyncratic_variance.schema,
        )

        risk_breakdown_with_weight = RiskBreakdownWithWeight(
            self.holding_snapshot, self.portfolio.start_date, self.portfolio.end_date
        )
        security_weight = (
            risk_breakdown_with_weight.get_security_with_weight()
            .sort(pl.col("security"))
            .select("portfolio_weight", "benchmark_weight", "active_weight")
            .fill_null(0)
        )
        # TODO: why nan
        beta_numpy = beta.fill_nan(0).to_numpy()
        security_weight_numpy = security_weight.to_numpy()
        factor_loading = beta_numpy.dot(security_weight_numpy)
        factor_loading = pl.DataFrame(factor_loading, schema=security_weight.schema)

        #### total risk attribution
        portfolio_weight = security_weight.get_column("portfolio_weight").to_numpy()
        F = factor_orth.to_pandas().cov()
        systematic_risk = (
            portfolio_weight.dot(beta_numpy.transpose())
            .dot(F)
            .dot(beta_numpy)
            .dot(portfolio_weight)
            * 12
        )
        idiosyncratic_variance_numpy = idiosyncratic_variance.fill_nan(0).to_numpy()
        stock_specific_risk = (
            portfolio_weight.dot(idiosyncratic_variance_numpy).dot(portfolio_weight)
            * 12
        )
        total_risk = np.sqrt(systematic_risk + stock_specific_risk)
        total_risk_df1 = pl.DataFrame(
            {
                "total_risk": total_risk,
                "systematic_risk": systematic_risk,
                "stock_specific_risk": stock_specific_risk,
            }
        )

        #### total risk attribution

        return

    def get_orthogonalized_factor(self, use_intercept=True):
        benchmark_return_df: pl.DataFrame = (
            self.benchmark.get_performance_return_from_month_start()
        )
        benchmark_return_df = (
            benchmark_return_df.group_by(pl.col("date").dt.month_start())
            .agg(((pl.col("return") + 1).product() - 1).alias("return"))
            .rename({"return": "benchmark_return"})
            .sort(pl.col("date"))
        )
        benchmark_select = ["benchmark_return"]
        if use_intercept:
            benchmark_return_df = benchmark_return_df.with_columns(
                pl.lit(1).alias("intercept")
            )
            benchmark_select = ["intercept", "benchmark_return"]

        sector_return_df = self.get_sector_monthly_return_df().sort(pl.col("date"))
        self.benchmark_return_df = benchmark_return_df
        self.sector_return_df = sector_return_df
        factor_orth, beta = self.regress(
            benchmark_return_df.select(benchmark_select),
            sector_return_df.select(pl.all().exclude("date")),
        )

        # factor_before = pl.concat(
        #     [
        #         benchmark_return_df.select(benchmark_select),
        #         sector_return_df.select(pl.all().exclude("date")),
        #     ],
        #     how="horizontal",
        # )
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
