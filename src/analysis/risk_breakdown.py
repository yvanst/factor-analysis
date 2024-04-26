import polars as pl


class RiskBreakdown:
    def __init__(self, holding_snapshot, market, date):
        self.holding_snapshot: pl.DataFrame = holding_snapshot
        self.market = market
        self.date = date
        self.sector_info_df = (
            pl.scan_parquet("parquet/base/us_sector_info.parquet")
            .filter(pl.col("date").dt.year() == date.year)
            .filter(pl.col("date").dt.month() == date.month)
            .select(["sedol7", "sector"])
            .collect()
        )
        self.benchmark_sector_weight_df = self.benchmark_sector_construction()

    def break_down_to_sectors(self):
        holding_snapshot = self.holding_snapshot.join(
            self.sector_info_df.rename({"sedol7": "security"}),
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
                lambda x: self.market.query_sedol_range_return(
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
            .filter(pl.col("date").dt.year() == self.date.year)
            .filter(pl.col("date").dt.month() == self.date.month)
            .select(["sedol7", "date", "sector"])
            .collect()
        )

        sector_weight = (
            pl.scan_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date").dt.year() == self.date.year)
            .filter(pl.col("date").dt.month() == self.date.month)
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

    def source_of_risk(self):
        holding_snapshot = self.break_down_to_sectors()
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


if __name__ == "__main__":
    import datetime

    from src.fund_universe import SECURITY_SEDOL
    from src.market import Market

    cfg = pl.Config()
    cfg.set_tbl_rows(100)

    start_date = datetime.date(2012, 12, 31)
    end_date = datetime.date(2023, 10, 31)
    security_universe = SECURITY_SEDOL
    market = Market(security_universe, start_date, end_date)

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
            "date": datetime.date(2016, 1, 29),
        }
    )
    risk_break = RiskBreakdown(df, market, datetime.date(2016, 1, 29))
    risk_break.source_of_risk()
