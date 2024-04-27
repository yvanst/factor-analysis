import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from src.analysis.security_weight_util import SecurityWeightUtil
from src.benchmark import Benchmark
from src.fund_universe import ISHARE_SECTOR_ETF_TICKER
from src.market import Market
from src.security_symbol import SecuritySedol


class RiskBreakdownToFactor:
    def __init__(self, portforlio, benchmark, end_date) -> None:
        self.holding_snapshot = portforlio.holding_snapshots[end_date]
        self.month_range = 60
        # 60 months before end_date
        if end_date.month == 2 and end_date.day == 29:
            end_date = datetime.date(end_date.year, 2, 28)
        self.start_date = datetime.date(end_date.year - 5, end_date.month, end_date.day)
        self.end_date = end_date
        self.benchmark = Benchmark(benchmark.benchmark, self.start_date, self.end_date)
        self.month_df = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date") >= self.start_date)
            .filter(pl.col("date") <= self.end_date)
            .select("date")
            .unique()
            .collect()
            .sort(pl.col("date"))
        )

    def calculate_stock_beta_against_factors(self):
        security_df = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date") >= self.start_date)
            .filter(pl.col("date") <= self.end_date)
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
        residual, beta = self.regress(
            factor_orth, ticker_return_df.select(ticker_cols).to_pandas()
        )
        index = [
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
        sector_list = index[1:]
        K = len(index)

        idiosyncratic_variance = residual.drop(index, axis=1).apply(
            lambda x: sum(x**2) / (self.month_range - K), axis=0
        )
        idiosyncratic_variance = pd.DataFrame(
            np.diag(idiosyncratic_variance),
            idiosyncratic_variance.index,
            idiosyncratic_variance.index,
        )
        # TODO:
        idiosyncratic_variance = idiosyncratic_variance.fillna(0)
        self.idiosyncratic_variance = idiosyncratic_variance

        risk_breakdown_with_weight = SecurityWeightUtil(
            self.holding_snapshot, self.start_date, self.end_date
        )
        security_weight = (
            risk_breakdown_with_weight.get_security_with_weight()
            .sort(pl.col("security"))
            .fill_null(0)  # TODO
            .to_pandas()
            .drop("date", axis=1)
            .set_index("security")
        )
        self.security_weight = security_weight
        self.benchmark_weight = security_weight["benchmark_weight"]
        # TODO: why nan
        beta = beta.fillna(0)
        self.beta = beta
        self.factor_loading = beta.dot(security_weight)

        #### total risk attribution 1
        F = factor_orth.cov()
        self.F = F
        portfolio_weight = security_weight["portfolio_weight"]
        self.portfolio_weight = portfolio_weight
        systematic_risk = (
            portfolio_weight.dot(beta.T).dot(F).dot(beta).dot(portfolio_weight) * 12
        )
        stock_specific_risk = (
            portfolio_weight.dot(idiosyncratic_variance).dot(portfolio_weight) * 12
        )
        total_risk = np.sqrt(systematic_risk + stock_specific_risk)
        self.total_risk_attribution_df1 = pl.DataFrame(
            {
                "total_risk": total_risk,
                "systematic_risk": systematic_risk,
                "stock_specific_risk": stock_specific_risk,
            }
        )

        #### total risk attribution 2
        F_benchmark = F.loc[
            np.isin(F.index.values, ["benchmark_return"]), "benchmark_return"
        ].to_frame()
        beta_benchmark = beta.loc[np.isin(beta.index.values, "benchmark_return"), :]
        total_risk_benchmark = (
            portfolio_weight.dot(beta_benchmark.T)
            .dot(F_benchmark)
            .dot(beta_benchmark)
            .dot(portfolio_weight)
            * 12
        )

        F_sector = F.loc[np.isin(F.index.values, sector_list), sector_list]
        beta_sector = beta.loc[np.isin(beta.index.values, sector_list), :]
        total_risk_sector = (
            portfolio_weight.dot(beta_sector.T)
            .dot(F_sector)
            .dot(beta_sector)
            .dot(portfolio_weight)
            * 12
        )
        self.total_risk_attribution_df2 = pl.DataFrame(
            {
                "systematic_risk": systematic_risk,
                "total_risk_benchmark": total_risk_benchmark,
                "total_risk_sector": total_risk_sector,
            }
        )

        #### total risk attribution 3
        total_risk_i = []
        for i in range(F.shape[1]):
            F_i = pd.DataFrame(0, F.columns, F.index)
            F_i.iloc[:, i] = F.iloc[:, i]
            total_risk_i.append(
                portfolio_weight.dot(beta.T).dot(F_i).dot(beta).dot(portfolio_weight)
                * 12
            )
        self.total_risk_attribution_df3 = pd.Series(total_risk_i, F.index)

        #### MCTR
        V = (beta.T.dot(F).dot(beta) + idiosyncratic_variance) * 12
        self.MCTR = V.dot(portfolio_weight) / total_risk

        #### tracking error attribution 1
        active_weight = security_weight["active_weight"]
        self.active_weight = active_weight
        systematic_active_risk = (
            active_weight.dot(beta.T).dot(F).dot(beta).dot(active_weight) * 12
        )
        stock_specific_active_risk = (
            active_weight.dot(idiosyncratic_variance).dot(active_weight) * 12
        )
        tracking_error = np.sqrt(systematic_active_risk + stock_specific_active_risk)
        self.tracking_error_attribution_df1 = pl.DataFrame(
            {
                "systematic_active_risk": systematic_active_risk,
                "stock_specific_active_risk": stock_specific_active_risk,
                "tracking_error": tracking_error,
            }
        )

        #### tracking error attribution 2
        benchmark_active_risk = (
            active_weight.dot(beta_benchmark.T)
            .dot(F_benchmark)
            .dot(beta_benchmark)
            .dot(active_weight)
            * 12
        )
        sector_active_risk = (
            active_weight.dot(beta_sector.T)
            .dot(F_sector)
            .dot(beta_sector)
            .dot(active_weight)
            * 12
        )
        self.tracking_error_attribution_df2 = pl.DataFrame(
            {
                "systematic_active_risk": systematic_active_risk,
                "benchmark_active_risk": benchmark_active_risk,
                "sector_active_risk": sector_active_risk,
            }
        )
        #### tracking error attribution 3
        active_risk_i = []
        for i in range(F.shape[1]):
            F_i = pd.DataFrame(0, F.columns, F.index)
            F_i.iloc[:, i] = F.iloc[:, i]
            active_risk_i.append(
                active_weight.dot(beta.T).dot(F_i).dot(beta).dot(active_weight) * 12
            )
        self.tracking_error_attribution_df3 = pd.Series(active_risk_i, F.index)

        #### MCAR
        self.MCAR = V.dot(active_weight) / tracking_error
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
            benchmark_return_df.select(benchmark_select).to_pandas(),
            sector_return_df.select(pl.all().exclude("date")).to_pandas(),
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
        market = Market(
            ISHARE_SECTOR_ETF_TICKER,
            self.start_date,
            self.end_date,
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
        market = Market(
            [SecuritySedol(security_df.get_column("security").item(0))],
            self.start_date,
            self.end_date,
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

    def regress(self, X: pd.DataFrame, Y: pd.DataFrame):
        X_inv = pd.DataFrame(np.linalg.pinv(X.values), X.columns, X.index)
        B = X_inv.dot(Y)
        residual = Y - X.dot(B)
        factors = pd.concat([X, residual], axis=1)
        return factors, B


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
        long_portfolio, benchmark, long_portfolio.end_date
    ).calculate_stock_beta_against_factors()
