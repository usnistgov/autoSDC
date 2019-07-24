""" SDC interface """

# if clr module (pythonnet) is not available, load the SDC shims
from . import experiment

try:
    from . import pump
    from . import position
    from . import potentiostat

except ModuleNotFoundError:

    from .shims import pump
    from .shims import position
    from .shims import potentiostat
