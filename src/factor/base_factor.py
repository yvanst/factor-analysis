from abc import ABC, abstractmethod


class BaseFactor(ABC):
    def __init__(self, security_universe, factor_type):
        self.security_universe = security_universe
        self.factor_type = factor_type
        self.num = 20

    def set_portfolio_at_start(self, portfolio):
        position = self.get_position(portfolio.start_date)
        print(
            f"initially buy on {portfolio.start_date}: {list(map(lambda t: (t[0].display(), round(t[1],3)), position[:3]))}..."
        )
        for security, weight in position:
            portfolio.add_security_weight(security, weight, 0)

    def get_position(self, date):
        security_list = self.get_security_list(date)
        if self.factor_type == "long":
            target_security = security_list[: self.num]
        elif self.factor_type == "short":
            target_security = list(reversed(security_list))[: self.num]
        else:
            raise ValueError(f"no implementation for {self.factor_type}")
        weight = 1 / len(target_security)
        return [(s, weight) for s in target_security]

    @abstractmethod
    def get_security_list(self, date):
        pass
