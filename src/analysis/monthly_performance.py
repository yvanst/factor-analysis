import math

import polars as pl


class MonthlyPerformance:
    def __init__(self):
        pass

    def get_annualized_stat(
        self, portfolio_performance: pl.DataFrame, benchmark_performance: pl.DataFrame
    ):
        portfolio_monthly_performance = (
            self.get_monthly_performance(portfolio_performance)
            .select("date", "monthly_return")
            .rename({"monthly_return": "portfolio_monthly_return"})
        )
        benchmark_monthly_performance = (
            self.get_monthly_performance(benchmark_performance)
            .select("date", "monthly_return")
            .rename({"monthly_return": "benchmark_monthly_return"})
        )
        stat_df = portfolio_monthly_performance.join(
            benchmark_monthly_performance, on="date", how="inner"
        ).with_columns(
            (
                pl.col("portfolio_monthly_return") - pl.col("benchmark_monthly_return")
            ).alias("relative_monthly_return")
        )
        stat_df = stat_df.select(
            (pl.col("portfolio_monthly_return").mean() * 12).alias("annualized_return"),
            (pl.col("portfolio_monthly_return").std() * math.sqrt(12)).alias(
                "annualized_volatility"
            ),
            (pl.col("relative_monthly_return").std() * math.sqrt(12)).alias(
                "tracking_error"
            ),
        )
        return stat_df

    def get_monthly_performance(self, monthly_performance: pl.DataFrame):
        month_end_date = monthly_performance.groupby(
            pl.col("date").dt.month_start().alias("month_start")
        ).agg(
            pl.col("date").min().alias("range_start"),
            pl.col("date").max().alias("range_end"),
        )
        monthly_performance = (
            monthly_performance.join(
                month_end_date, left_on="date", right_on="range_start", how="left"
            )
            .join(month_end_date, left_on="date", right_on="range_end", how="left")
            .filter(
                (pl.col("range_start").is_not_null())
                | (pl.col("range_end").is_not_null())
            )
        )
        monthly_performance = (
            monthly_performance.groupby(pl.col("date").dt.month_start().alias("date"))
            .agg(
                pl.when(pl.col("range_start").is_not_null())
                .then(pl.col("value"))
                .otherwise(None)
                .max()
                .alias("month_start_value"),
                pl.when(pl.col("range_end").is_not_null())
                .then(pl.col("value"))
                .otherwise(None)
                .max()
                .alias("month_end_value"),
            )
            .with_columns(
                (pl.col("month_end_value") / pl.col("month_start_value") - 1).alias(
                    "monthly_return"
                )
            )
        )
        return monthly_performance


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
    portfolio_performance = long_portfolio.value_book.select("date", "value")
    benchmark = Benchmark(SecurityTicker("^SPX", "index"), start_date, end_date)
    benchmark_performance = benchmark.get_performance()

    monthly_performance = MonthlyPerformance()
    stat = monthly_performance.get_annualized_stat(
        portfolio_performance, benchmark_performance
    )
