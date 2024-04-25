import polars as pl

from src.security_symbol import SecuritySedol
from src.factor.base_factor import BaseFactor
from src.impl.roe import RoeImpl


class RoeFactor(BaseFactor):
    def __init__(self, security_universe, factor_type):
        super().__init__(security_universe, factor_type)

    def get_security_list(self, date):
        sedol_list = (
            pl.read_parquet("parquet/base/us_sector_weight.parquet")
            .filter(pl.col("date").dt.year() == date.year)
            .filter(pl.col("date").dt.month() == date.month)
            .filter(pl.col("weight") > 0)
            .select(pl.col("sedol7").unique())
            .get_column("sedol7")
            .to_list()
        )
        security_list = list(RoeImpl(sedol_list).get_security_list(date))
        security_list = [SecuritySedol(id) for id in security_list]
        return security_list
