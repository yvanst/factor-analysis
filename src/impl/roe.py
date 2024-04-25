import datetime

import polars as pl

from src.impl.base_impl import BaseImpl


class RoeImpl(BaseImpl):
    def __init__(self, security_universe_sedol_ids) -> None:
        # hyper parameter: generate z-score using data in the last n years
        self.z_score_year_range = 10
        self.table = f"parquet/roe/us_security_roe_ntm_monthly.parquet"
        self.security_universe_sedol_ids = security_universe_sedol_ids

    def impl_security_z_score(self, observe_date):
        total_df_list = []
        if observe_date.month == 2 and observe_date.day == 29:
            observe_date = datetime.date(observe_date.year, 2, 28)
        for delta in range(self.z_score_year_range):
            date = datetime.date(
                observe_date.year - delta, observe_date.month, observe_date.day
            )
            security_signal_df = self.impl_security_signal(date)
            total_df_list.append(security_signal_df)
        total_signal_df = pl.concat(total_df_list)
        z_score_df = self.get_security_z_score(total_signal_df)
        return z_score_df

    def impl_security_signal(self, date):
        cur_month = datetime.date(date.year, date.month, 1)
        signal_df = (
            pl.scan_parquet(self.table)
            .filter(pl.col("roe").is_not_null())
            .filter(pl.col("date").dt.year() == date.year)
            .filter(pl.col("date").dt.month() == date.month)
            .filter(pl.col("sedol7").is_in(self.security_universe_sedol_ids))
            .collect()
        )
        # rewrite the date column to unify the date in the same month
        signal_df = signal_df.with_columns(pl.lit(cur_month).alias("date"))
        signal_df = signal_df.rename({"roe": "signal"})
        return signal_df
