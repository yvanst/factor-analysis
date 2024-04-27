import polars as pl
from scipy.stats import norm


class Var:
    def __init__(self, portfolio_performance):
        self.portfolio_performance: pl.DataFrame = portfolio_performance
        self.hist_days = portfolio_performance.shape[0]
        self.rolling_days_df = self.get_rolling_days_df()
        self.pcts = [0.9, 0.95, 0.99]

    def get_rolling_days_df(self):
        portfolio_performance = (
            self.portfolio_performance.sort(pl.col("date"), descending=True)
            .with_row_index(name="index", offset=0)
            .filter(pl.col("index") < 1000)
        )
        portfolio_performance = portfolio_performance.sort(pl.col("date"))
        portfolio_performance = portfolio_performance.with_columns(
            (pl.col("value") / pl.col("value").shift(1) - 1).alias("1-day return")
        )
        one_day_return_series = (
            portfolio_performance.sort(pl.col("date"), descending=True)
            .get_column("1-day return")
            .sort(nulls_last=True)
        )
        five_day_return_series = (
            portfolio_performance.rolling(index_column="date", period="5d")
            .agg(((pl.col("1-day return") + 1).product() - 1).alias("5-day return"))
            .sort(pl.col("date"), descending=True)
            .with_row_index(name="index", offset=0)
            .get_column("5-day return")
            .sort(nulls_last=True)
        )
        ten_day_return_series = (
            portfolio_performance.rolling(index_column="date", period="10d")
            .agg(((pl.col("1-day return") + 1).product() - 1).alias("10-day return"))
            .sort(pl.col("date"), descending=True)
            .with_row_index(name="index", offset=0)
            .get_column("10-day return")
            .sort(nulls_last=True)
        )
        twenty_one_day_return_series = (
            portfolio_performance.rolling(index_column="date", period="21d")
            .agg(((pl.col("1-day return") + 1).product() - 1).alias("21-day return"))
            .sort(pl.col("date"), descending=True)
            .with_row_index(name="index", offset=0)
            .get_column("21-day return")
            .sort(nulls_last=True)
        )
        rolling_days_df = pl.DataFrame(
            [
                one_day_return_series,
                five_day_return_series,
                ten_day_return_series,
                twenty_one_day_return_series,
            ]
        ).with_row_index(name="index", offset=0)
        return rolling_days_df

    def get_imperical_var(self):
        var = self.rolling_days_df
        target_rows = [int(self.hist_days * (1 - pct)) for pct in self.pcts]
        var = var.filter(pl.col("index").is_in(target_rows))
        var = pl.concat(
            [var.sort("index", descending=True), pl.DataFrame({"pct": self.pcts})],
            how="horizontal",
        )
        var = var.select(
            "pct", "1-day return", "5-day return", "10-day return", "21-day return"
        ).sort(pl.col("pct"))
        return var

    def get_normal_distribution_var(self):
        var = self.rolling_days_df
        var = var.select(
            pl.col("1-day return").mean().alias("1-day mean"),
            pl.col("1-day return").std().alias("1-day std"),
            pl.col("5-day return").mean().alias("5-day mean"),
            pl.col("5-day return").std().alias("5-day std"),
            pl.col("10-day return").mean().alias("10-day mean"),
            pl.col("10-day return").std().alias("10-day std"),
            pl.col("21-day return").mean().alias("21-day mean"),
            pl.col("21-day return").std().alias("21-day std"),
        )
        one_day_return_series = [
            norm.ppf(
                1 - pct,
                loc=var.get_column("1-day mean").item(0),
                scale=var.get_column("1-day std").item(0),
            )
            for pct in self.pcts
        ]
        five_day_return_series = [
            norm.ppf(
                1 - pct,
                loc=var.get_column("5-day mean").item(0),
                scale=var.get_column("5-day std").item(0),
            )
            for pct in self.pcts
        ]
        ten_day_return_series = [
            norm.ppf(
                1 - pct,
                loc=var.get_column("10-day mean").item(0),
                scale=var.get_column("10-day std").item(0),
            )
            for pct in self.pcts
        ]
        twenty_one_day_return_series = [
            norm.ppf(
                1 - pct,
                loc=var.get_column("21-day mean").item(0),
                scale=var.get_column("21-day std").item(0),
            )
            for pct in self.pcts
        ]
        var = pl.DataFrame(
            {
                "pct": self.pcts,
                "1-day return": one_day_return_series,
                "5-day return": five_day_return_series,
                "10-day return": ten_day_return_series,
                "21-day return": twenty_one_day_return_series,
            }
        )

        return var


if __name__ == "__main__":
    import datetime

    from src.benchmark import Benchmark
    from src.security_symbol import SecurityTicker

    start_date = datetime.date(2012, 12, 31)
    end_date = datetime.date(2023, 10, 31)
    benchmark = Benchmark(SecurityTicker("^SPX", "index"), start_date, end_date)
    benchmark_performance = benchmark.get_performance()
    var = Var(benchmark_performance)
    print("normal distribution:")
    print(var.get_normal_distribution_var())
    print("imperical value:")
    print(var.get_imperical_var())
