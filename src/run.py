import datetime

from src.analysis.hit_rate import HitRate
from src.analysis.information_coefficient import InformationCoefficient
from src.analysis.metric import Metric
from src.analysis.plot import Plot
from src.backtest import BackTest
from src.benchmark import Benchmark
from src.factor.cape import CapeFactor
from src.factor.roe import RoeFactor
from src.fund_universe import SECURITY_SEDOL
from src.market import Market
from src.portfolio import Portfolio
from src.rebalance import Rebalance
from src.security_symbol import SecurityTicker

start_date = datetime.date(2012, 12, 31)
end_date = datetime.date(2023, 10, 31)
security_universe = SECURITY_SEDOL
rebalance_period = 1
rebalance_interval = "1mo"
Factor = RoeFactor
market = Market(security_universe, start_date, end_date)

### Long factor
long_factor = Factor(security_universe, "long")
long_portfolio = Portfolio(100.0, start_date, end_date)
long_factor.set_portfolio_at_start(long_portfolio)

rebalance = Rebalance(rebalance_period, long_portfolio, long_factor, rebalance_interval)

backtest = BackTest(long_portfolio, market, rebalance)
backtest.run()

### Short factor
short_factor = Factor(security_universe, "short")
short_portfolio = Portfolio(100.0, start_date, end_date)
short_factor.set_portfolio_at_start(short_portfolio)

rebalance = Rebalance(
    rebalance_period, short_portfolio, short_factor, rebalance_interval
)

backtest = BackTest(short_portfolio, market, rebalance)
backtest.run()

### plot
benchmark = Benchmark(SecurityTicker("^SPX", "index"), start_date, end_date)

benchmark_performance = benchmark.get_performance()

# metric = Metric(long_portfolio, benchmark_performance)
# print(f"portfolio annulized return: {metric.portfolio_annualized_return()}")
# print(
#     f"portfolio annulized return relative to benchmark: {metric.annualized_return_relative_to_benchmark()}"
# )
# print(f"information ratio: {metric.information_ratio()}")
# print(f"average monthly turnover: {metric.avg_monthly_turnover()}")
# print(f"sharpe ratio(with risk-free rate 0.04): {metric.sharpe_ratio()}")


plot = Plot(
    long_portfolio,
    short_portfolio,
    benchmark_performance,
    "SPX",
)
plot.draw()

# ie = InformationCoefficient(long_portfolio, long_factor, market, rebalance_period)
# ie.get_information_coefficient()
# ie.draw()

# hr = HitRate(long_portfolio, long_factor, market, rebalance_period, benchmark)
# hr.get_hit_rate()
# hr.draw()
