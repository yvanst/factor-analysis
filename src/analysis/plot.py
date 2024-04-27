import matplotlib.pyplot as plt
from src.portfolio import Portfolio


class Plot:
    def __init__(
        self,
        equal_portfolio: Portfolio,
        min_te_portfolio: Portfolio,
        mvo_portfolio: Portfolio,
        benchmark,
        benchmark_label,
    ):
        self.equal_portfolio = equal_portfolio
        self.min_te_portfolio = min_te_portfolio
        self.mvo_portfolio = mvo_portfolio
        self.dates = equal_portfolio.date_df.get_column("date")
        self.benchmark_value = benchmark.get_column("value")
        self.benchmark_label = benchmark_label

    def plot_performance(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        equal_portfolio_value = self.equal_portfolio.value_book.get_column("value")
        min_te_portfolio_value = self.min_te_portfolio.value_book.get_column("value")
        mvo_portfolio_value = self.mvo_portfolio.value_book.get_column("value")
        portfolios = [
            mvo_portfolio_value,
            min_te_portfolio_value,
            equal_portfolio_value,
        ]
        portfolio_labels = [
            f"MVO - {self.benchmark_label}",
            f"MIN_TE - {self.benchmark_label}",
            f"EQUAL - {self.benchmark_label}",
        ]
        colors = ["tab:pink", "tab:green", "tab:red"]
        for p, l, c in zip(portfolios, portfolio_labels, colors):
            relative_value = p - self.benchmark_value
            ax.plot(self.dates, relative_value, label=l, color=c)

        ax.plot(
            self.dates,
            self.benchmark_value - self.benchmark_value,
            label=f"{self.benchmark_label} - {self.benchmark_label}",
            color="tab:blue",
        )

        step = self.dates.shape[0] // 30
        ax.set_xticks(
            ticks=self.dates[::step],
            labels=self.dates[::step],
            rotation=90,
        )
        ax.grid(True)
        ax.legend()
        ax.set_title("Portofolio Return Relative to Benchmark")
        plt.show()

    def plot_total_risk(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        dates = self.equal_portfolio.total_risk_df.get_column("date")
        equal_portfolio_value = self.equal_portfolio.total_risk_df.get_column(
            "total_risk"
        )
        min_te_portfolio_value = self.min_te_portfolio.total_risk_df.get_column(
            "total_risk"
        )
        mvo_portfolio_value = self.mvo_portfolio.total_risk_df.get_column("total_risk")
        portfolios = [
            mvo_portfolio_value,
            min_te_portfolio_value,
            equal_portfolio_value,
        ]
        portfolio_labels = [
            f"MVO",
            f"MIN_TE",
            f"EQUAL",
        ]
        colors = ["tab:pink", "tab:green", "tab:red"]
        for p, l, c in zip(portfolios, portfolio_labels, colors):
            ax.plot(dates, p, label=l, color=c)

        ax.grid(True)
        ax.legend()
        ax.set_title("Total Risk")
        plt.show()

    def plot_tracking_error(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        dates = self.equal_portfolio.tracking_error_df.get_column("date")
        equal_portfolio_value = self.equal_portfolio.tracking_error_df.get_column(
            "tracking_error"
        )
        min_te_portfolio_value = self.min_te_portfolio.tracking_error_df.get_column(
            "tracking_error"
        )
        mvo_portfolio_value = self.mvo_portfolio.tracking_error_df.get_column(
            "tracking_error"
        )
        portfolios = [
            mvo_portfolio_value,
            min_te_portfolio_value,
            equal_portfolio_value,
        ]
        portfolio_labels = [
            f"MVO",
            f"MIN_TE",
            f"EQUAL",
        ]
        colors = ["tab:pink", "tab:green", "tab:red"]
        for p, l, c in zip(portfolios, portfolio_labels, colors):
            ax.plot(dates, p, label=l, color=c)

        ax.grid(True)
        ax.legend()
        ax.set_title("Tracking Error")
        plt.show()
