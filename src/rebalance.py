import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl

from src.analysis.risk_breakdown_to_factor import RiskBreakdownToFactor


class Rebalance:
    def __init__(
        self,
        period,
        portfolio,
        factor,
        benchmark,
        interval="1d",
        disable_rebalance=False,
    ) -> None:
        self.period = period
        self.portfolio = portfolio
        self.factor = factor
        self.benchmark = benchmark
        # could be "1d" or "1mo"
        # if "1mon", then rebalance happens at the last market open day of the month
        self.interval = interval
        self.disable_rebalance = disable_rebalance

    def check_and_run(self, iter_index, prev_rebalance_index):
        if self.disable_rebalance:
            return False
        if iter_index + 1 >= len(self.portfolio.date_df):
            return False

        if self.interval == "1d":
            if iter_index % self.period == 0:
                self.run(iter_index)
                return True
            else:
                return False
        elif self.interval == "1mo":
            prev_rebalance_date = self.portfolio.date_df.item(prev_rebalance_index, 0)
            cur_date = self.portfolio.date_df.item(iter_index, 0)
            next_date = self.portfolio.date_df.item(iter_index + 1, 0)
            if self.interval == "1mo" and cur_date.month == next_date.month:
                return False
            diff = (
                cur_date.month - prev_rebalance_date.month
                if cur_date.year == prev_rebalance_date.year
                else cur_date.month + 12 - prev_rebalance_date.month
            )
            if self.interval == "1mo" and diff == self.period:
                self.run(iter_index)
                return True
            else:
                return False
        else:
            raise ValueError(f"no implementation for {self.interval}")

    def run(self, iter_index):
        cur_date = self.portfolio.date_df.item(iter_index, 0)
        position = self.factor.get_position(cur_date)

        residual = 0
        valid_count = 0
        new_position = []
        for s, w in position:
            valid_count += 1
            if valid_count < len(position):
                new_position.append((s, w))
            else:
                new_position.append((s, w - 0.01))  # rounding error

        if residual > 0:
            residual -= 0.01  # rounding error
            new_position = [
                (s, w + round(residual / valid_count, 3)) if w != 0 else (s, 0)
                for s, w in new_position
            ]

        position_change = []

        new_securities = [s for s, _ in new_position]
        for security in self.portfolio.security_book.keys():
            original_weight = self.portfolio.get_security_weight(security, iter_index)
            if security not in new_securities and original_weight > 0:
                position_change.append((security, -original_weight))

        for security, weight in new_position:
            original_weight = self.portfolio.get_security_weight(security, iter_index)
            position_change.append((security, weight - original_weight))

        # sold first and then buy
        position_change.sort(key=lambda p: p[1])
        print(
            f"rebalance on {cur_date}: {list(map(lambda t: (t[0].display(), round(t[1],3)), position_change[:3]))}..."
        )

        turnover = sum((map(lambda t: abs(t[1]), position_change)))
        self.portfolio.value_book[iter_index]["turnover"] = turnover
        # sector = ",".join((map(lambda t: t[0].sector, position_change)))
        # self.portfolio.value_book[iter_index]["sector"] = sector

        for security, weight_change in position_change:
            if weight_change < 0:
                self.portfolio.reduce_security_weight(
                    security, abs(weight_change), iter_index
                )
            if weight_change > 0:
                self.portfolio.add_security_weight(security, weight_change, iter_index)
        self.portfolio.append_holding_snapshot(iter_index)

    def minimum_tracking_error_portfolio_weight(self, cur_date):
        rb = RiskBreakdownToFactor(self.portfolio, self.benchmark, cur_date)
        rb.calculate_stock_beta_against_factors()

        benchmark_weight = np.asarray(rb.benchmark_weight)
        n = len(benchmark_weight)
        w = cp.Variable(n)

        BFB = np.asmatrix(rb.beta.T) @ rb.F @ rb.beta
        D = np.asmatrix(rb.idiosyncratic_variance)

        risk = cp.quad_form(w - benchmark_weight, BFB) + cp.quad_form(
            w - benchmark_weight, D
        )
        problem = cp.Problem(
            cp.Minimize(risk),
            [sum(w) == 1, w >= 0, w <= [1 if i > 0 else 0 for i in benchmark_weight]],
        )
        tracking_error = (problem.solve() * 12) ** 0.5

        active_risk_i = []
        for i in range(rb.F.shape[1]):
            F_i = pd.DataFrame(0, rb.F.columns, rb.F.index)
            F_i.iloc[:, i] = rb.F.iloc[:, i]
            active_risk_i.append(
                (w.value - rb.benchmark_weight)
                @ (rb.beta.T)
                @ (rb.F_i)
                @ (rb.beta)
                @ (w.value - rb.benchmark_weight)
                * 12
            )
        active_risk_i = pd.Series(active_risk_i, rb.F.index)
        active_factor_loading = (w.value - rb.benchmark_weight) @ rb.beta.T
        factor_loading = pd.Series(w.value @ rb.beta.T)
        return
