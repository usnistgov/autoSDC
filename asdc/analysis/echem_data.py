import pandas as pd
from enum import IntEnum

class Status(IntEnum):
    OK = 0
    WARN = 1
    RETRY = 2
    FAIL = 3

class EchemData(pd.DataFrame):

    def check_quality(self):
        """ run diagnostics, return status that explains what to do and why.

        maybe status could be like
        OK -- no errors, proceed
        Warn -- errors, nothing to be done, so proceed?
        Retry -- error, redo the measurement to try to get a replicate
        Fail -- error, no point in performing subsequent work
        """
        raise NotImplementedError
