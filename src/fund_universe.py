import polars as pl
from src.security_symbol import SecuritySedol

sedol_list = (
    pl.read_parquet("parquet/base/us_sector_weight.parquet")
    .filter(pl.col("date").dt.year() == 2020)
    .filter(pl.col("date").dt.month() == 1)
    .filter(pl.col("weight") > 0)
    .select(pl.col("sedol7").unique())
    .get_column("sedol7")
    .to_list()
)


SECURITY_SEDOL = [SecuritySedol(id) for id in sedol_list]
