""" SDC interface """

from __future__ import absolute_import

try:
    raise ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError

# if clr module (pythonnet) is not available, load the SDC shims
from . import experiment
from . import pump

try:
    from . import position
    from . import potentiostat

except ModuleNotFoundError:

    from .shims import pump
    from .shims import position
    from .shims import potentiostat
