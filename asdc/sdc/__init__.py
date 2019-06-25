""" SDC interface """

# if clr module (pythonnet) is not available, load the SDC shims
from . import experiment

try:
    from . import position
    from . import potentiostat

except ModuleNotFoundError:

    from .shims import position
    from .shims import potentiostat
