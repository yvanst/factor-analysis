import datetime

from src.analysis.plot import Plot
from src.backtest import BackTest
from src.benchmark import Benchmark
from src.factor.roe import RoeFactor
from src.fund_universe import SECURITY_SEDOL
from src.market import Market
from src.portfolio import Portfolio
from src.rebalance import Rebalance
from src.security_symbol import SecurityTicker
from src.analysis.risk_breakdown_to_factor import RiskBreakdownToFactor

start_date = datetime.date(2015, 8, 30)
end_date = datetime.date(2015, 10, 31)
security_universe = SECURITY_SEDOL
rebalance_period = 1
rebalance_interval = "1mo"
# could be EQUAL|MIN_TE|MVO
weight_strategy = "MVO"
Factor = RoeFactor
market = Market(security_universe, start_date, end_date)
benchmark = Benchmark(SecurityTicker("^SPX", "index"), start_date, end_date)
benchmark_performance = benchmark.get_performance()

### equal weight factor
equal_factor = Factor(security_universe, "long")
equal_portfolio = Portfolio(100.0, start_date, end_date)
equal_factor.set_portfolio_at_start(equal_portfolio)

rebalance = Rebalance(
    rebalance_period,
    equal_portfolio,
    equal_factor,
    benchmark,
    rebalance_interval,
    "EQUAL",
)

backtest = BackTest(equal_portfolio, market, rebalance)
backtest.run()

# rb = RiskBreakdownToFactor(equal_portfolio, benchmark, end_date)
# rb.tracking_error_breakdown_analysis()
# rb.total_risk_breakdown_analysis()
# rb.plot_correlation()

### minimum tracking error factor
min_te_factor = Factor(security_universe, "long")
min_te_portfolio = Portfolio(100.0, start_date, end_date)
min_te_factor.set_portfolio_at_start(min_te_portfolio)

rebalance = Rebalance(
    rebalance_period,
    min_te_portfolio,
    min_te_factor,
    benchmark,
    rebalance_interval,
    "MIN_TE",
)

backtest = BackTest(min_te_portfolio, market, rebalance)
backtest.run()


### MVO factor
mvo_factor = Factor(security_universe, "long")
mvo_portfolio = Portfolio(100.0, start_date, end_date)
mvo_factor.set_portfolio_at_start(mvo_portfolio)

rebalance = Rebalance(
    rebalance_period,
    mvo_portfolio,
    mvo_factor,
    benchmark,
    rebalance_interval,
    "MVO",
)

backtest = BackTest(mvo_portfolio, market, rebalance)
backtest.run()


### plot
plot = Plot(
    equal_portfolio,
    min_te_portfolio,
    mvo_portfolio,
    benchmark_performance,
    "SPX",
)
plot.plot_total_risk()
plot.plot_performance()
