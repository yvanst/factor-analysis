import datetime
from abc import ABC, abstractmethod

import polars as pl


class BaseImpl(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def impl_security_signal(self, date):
        raise NotImplementedError()

    @abstractmethod
    def impl_security_z_score(self, observe_date):
        raise NotImplementedError()

    def get_security_list(self, observe_date):
        z_score_df = self.impl_security_z_score(observe_date)
        ordered_security = (
            z_score_df.sort("z-score", descending=True).get_column("sedol7").to_list()
        )
        return ordered_security

    def get_security_z_score(self, total_signal_df):
        latest_month = (
            total_signal_df.select(pl.col("date").max()).get_column("date").item(0)
        )
        latest_signal_df = total_signal_df.filter(pl.col("date") == latest_month)

        stat_signal_df = (
            total_signal_df
            # .filter(pl.col("date") != latest_month)
            .filter(pl.col("signal").is_not_null())
            .filter(pl.col("signal").is_not_nan())
            .group_by(["sedol7"])
            .agg(
                (pl.col("signal").std().alias("std")),
                (pl.col("signal").mean().alias("mean")),
            )
            .filter(pl.col("std").is_not_null())
        )

        assert len(stat_signal_df.filter(pl.col("std").is_null())) == 0
        assert len(stat_signal_df.filter(pl.col("mean").is_null())) == 0

        merge_df = latest_signal_df.join(
            stat_signal_df, on="sedol7", how="inner"
        ).with_columns(
            ((pl.col("signal") - pl.col("mean")) / pl.col("std")).alias("z-score")
        )
        return merge_df
