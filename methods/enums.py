import enum
from datetime import datetime as dt

class ENUM_TYPE_OB(enum.Enum):
    obBuy = 1
    obSell = 2

# print(ENUM_TYPE_OB)
    
class ENUM_TYPE_FIND(enum.Enum):
    onlyBuy = 0
    onlySell = 1
    both = 2

class dataOB():
    def __init__(self, eto:ENUM_TYPE_OB, dtime:dt) -> None:
        self.eto = eto
        self.dt = dt