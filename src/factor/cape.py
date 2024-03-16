from src.factor.base_factor import BaseFactor
from src.impl.cape import CapeImpl
from src.security_symbol import SecuritySedol


class CapeFactor(BaseFactor):
    def __init__(self, security_universe, factor_type):
        super().__init__(security_universe, factor_type)

    def get_security_list(self, date):
        """
        get the sorted sector based on the signal
        """
        security_list = list(CapeImpl().get_security_list(date))
        security_list = [SecuritySedol(id) for id in security_list]
        return security_list
