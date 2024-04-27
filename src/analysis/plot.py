import matplotlib.pyplot as plt


class Plot:
    def __init__(
        self,
        equal_portfolio,
        min_te_portfolio,
        mvo_portfolio,
        benchmark,
        benchmark_label,
    ):
        self.equal_portfolio = equal_portfolio
        self.min_te_portfolio = min_te_portfolio
        self.mvo_portfolio = mvo_portfolio
        self.equal_portfolio_value = equal_portfolio.value_book.get_column("value")
        self.min_te_portfolio_value = min_te_portfolio.value_book.get_column("value")
        self.mvo_portfolio_value = mvo_portfolio.value_book.get_column("value")
        self.dates = equal_portfolio.date_df.get_column("date")
        self.benchmark_value = benchmark.get_column("value")
        self.benchmark_label = benchmark_label
        _, self.ax = plt.subplots(1, 1, figsize=(10, 5))

    def draw(self):
        portfolios = [
            self.equal_portfolio_value,
            self.min_te_portfolio_value,
            self.mvo_portfolio_value,
        ]
        portfolio_labels = [
            f"EQUAL - {self.benchmark_label}",
            f"MIN_TE - {self.benchmark_label}",
            f"MVO - {self.benchmark_label}",
        ]
        colors = ["tab:red", "tab:green", "tab:pink"]
        for p, l, c in zip(portfolios, portfolio_labels, colors):
            relative_value = p - self.benchmark_value
            self.ax.plot(self.dates, relative_value, label=l, color=c)

        self.ax.plot(
            self.dates,
            self.benchmark_value - self.benchmark_value,
            label=f"{self.benchmark_label} - {self.benchmark_label}",
            color="tab:blue",
        )

        step = self.dates.shape[0] // 30
        self.ax.set_xticks(
            ticks=self.dates[::step],
            labels=self.dates[::step],
            rotation=90,
        )
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_title("Portofolio Return Relative to Benchmark")
        plt.show()
