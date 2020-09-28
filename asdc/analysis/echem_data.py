import pandas as pd

class EchemData(pd.DataFrame):

    def check_quality(self):
        raise NotImplementedError
