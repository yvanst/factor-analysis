import polars as pl

from src.market import Market
from src.security_symbol import SecuritySedol


class RiskBreakdownWithWeight:
    def __init__(self, holding_snapshot, start_date, end_date):
        self.holding_snapshot: pl.DataFrame = holding_snapshot
        self.start_date = start_date
        self.end_date = end_date
        self.sector_info_df = (
            pl.scan_parquet("parquet/base/us_sector_info.parquet")
            .filter(pl.col("date") >= start_date)
            .filter(pl.col("date") <= end_date)
            .select(["sedol7", "sector"])
            .collect()
        )
        self.benchmark_sector_weight_df = self.benchmark_sector_construction()

    def get_snapshot_holding_sector_and_forward_return(self):
        market = Market(
            [SecuritySedol(holding_snapshot.get_column("security").item(0))],
            self.start_date,
            self.end_date,
        )
        holding_snapshot = self.holding_snapshot.join(
            self.sector_info_df.filter(pl.col("date").dt.year() == self.end_date.year)
            .filter(pl.col("date").dt.month() == self.end_date.month)
            .rename({"sedol7": "security"}),
            on="security",
            how="inner",
        )
        holding_snapshot = holding_snapshot.with_columns(
            pl.col("date").dt.month_end().alias("forward_start_date"),
            (pl.col("date").dt.month_end() + pl.duration(days=1))
            .dt.month_end()
            .alias("forward_end_date"),
        )
        holding_snapshot = holding_snapshot.with_columns(
            pl.struct(["security", "forward_start_date", "forward_end_date"])
            .map_elements(
                lambda x: market.query_sedol_range_return(
                    x["security"],
                    x["forward_start_date"],
                    x["forward_end_date"],
                )
            )
            .alias("forward_1mo_return")
        )
        assert len(holding_snapshot) == 20
        return holding_snapshot

    def benchmark_sector_construction(self):
        sector_info = (
            pl.scan_parquet("parquet/base/us_sector_info.parquet")
            .filter(pl.col("date").dt.year() == self.end_date.year)
            .filter(pl.col("date").dt.month() == self.end_date.month)
            .select(["sedol7", "date", "sector"])
            .collect()
        )

        sector_weight = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date").dt.year() == self.end_date.year)
            .filter(pl.col("date").dt.month() == self.end_date.month)
            .select(["sedol7", "weight"])
            .collect()
        )

        merge = (
            sector_info.join(sector_weight, on=["sedol7"], how="inner")
            .select(["sedol7", "sector", "weight"])
            .filter(pl.col("sector") != "--")
        )

        sector_weight_df = (
            merge.group_by("sector")
            .agg((pl.col("weight").sum() / 100).alias("benchmark_weight"))
            .filter(pl.col("benchmark_weight") > 0)
        )
        assert sector_weight_df.get_column("benchmark_weight").sum() - 1 < 1e-5
        return sector_weight_df

    def risk_break_down_to_sector(self):
        holding_snapshot = self.get_snapshot_holding_sector_and_forward_return()
        holding_snapshot = holding_snapshot.group_by("sector").agg(
            pl.col("weight").sum().alias("portfolio_weight"),
        )
        sector_df = self.benchmark_sector_weight_df.join(
            holding_snapshot, on="sector", how="left"
        )
        sector_df = (
            sector_df.with_columns(
                pl.coalesce(pl.col("portfolio_weight"), 0).alias("portfolio_weight")
            )
            .with_columns(
                (pl.col("portfolio_weight") - pl.col("benchmark_weight")).alias(
                    "active_weight"
                )
            )
            .select("sector", "portfolio_weight", "benchmark_weight", "active_weight")
            .sort(pl.col("active_weight"), descending=True)
        )
        return sector_df

    def get_security_with_weight(self):
        # security universe: stock weight > 0 in [start_date, end_date]
        # security weight: stock weight in end_date
        security_df = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date") >= self.start_date)
            .filter(pl.col("date") <= self.end_date)
            .select(["sedol7", "date", "weight"])
            .collect()
            .rename({"sedol7": "security"})
            .filter(pl.col("weight") > 0)
        )
        latest_date = (
            security_df.select(pl.col("date").max()).get_column("date").item(0)
        )
        security_df = security_df.group_by(pl.col("security")).agg(
            pl.when(pl.col("date") == latest_date)
            .then(pl.col("weight") / 100)
            .otherwise(0)
            .max()
            .alias("benchmark_weight")
        )
        security_df = (
            security_df.join(self.holding_snapshot, on="security", how="left")
            .with_columns(pl.coalesce(pl.col("weight"), 0).alias("portfolio_weight"))
            .with_columns(pl.lit(latest_date).alias("date"))
            .with_columns(
                (pl.col("portfolio_weight") - pl.col("benchmark_weight")).alias(
                    "active_weight"
                )
            )
            .sort(pl.col("active_weight"))
            .select(
                "date",
                "security",
                "benchmark_weight",
                "portfolio_weight",
                "active_weight",
            )
        )
        return security_df


if __name__ == "__main__":
    import datetime

    from src.fund_universe import SECURITY_SEDOL

    cfg = pl.Config()
    cfg.set_tbl_rows(100)

    start_date = datetime.date(2012, 12, 31)
    end_date = datetime.date(2023, 10, 31)
    security_universe = SECURITY_SEDOL

    df = pl.DataFrame(
        {
            "security": [
                "2517382",
                "2215460",
                "2434209",
                "BLP1HW5",
                "2795393",
                "2648806",
                "2232793",
                "2491839",
                "2182553",
                "2130109",
                "2480138",
                "2459020",
                "2536763",
                "2065308",
                "2457552",
                "BNZHB81",
                "2293819",
                "BYV2325",
                "2550707",
                "2681511",
            ],
            "weight": 0.05,
            "date": datetime.date(2013, 10, 31),
        }
    )
    start_date = datetime.date(2012, 12, 31)
    end_date = datetime.date(2013, 10, 31)
    risk_break = RiskBreakdownWithWeight(df, start_date, end_date)
    # risk_break.risk_break_down_to_sector()
    risk_break.get_security_with_weight()
