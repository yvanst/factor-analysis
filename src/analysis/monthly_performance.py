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
            (pl.col("portfolio_monthly_return").mean() * 12).alias(
                "monthly_averaged_return(annualized)"
            ),
            (pl.col("portfolio_monthly_return").std() * math.sqrt(12)).alias(
                "monthly_averaged_volatility(annualized)"
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
        monthly_performance = monthly_performance.with_columns(
            pl.col("date").dt.month_start().alias("month_start")
        ).join(month_end_date, on="month_start", how="inner")

        monthly_performance = (
            monthly_performance.groupby(pl.col("month_start"))
            .agg(
                pl.when(pl.col("range_start") == pl.col("date"))
                .then(pl.col("value"))
                .otherwise(None)
                .max()
                .alias("month_start_value"),
                pl.when(pl.col("range_end") == pl.col("date"))
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
