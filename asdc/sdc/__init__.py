""" SDC interface """

from __future__ import absolute_import

import sys

# try:
#     raise ModuleNotFoundError
# except NameError:
#     ModuleNotFoundError = ImportError


# if clr module (pythonnet) is not available, load the SDC shims
from . import pump
from . import orion
from . import reglo
from . import experiment
from . import microcontroller

if sys.platform == 'win32':
    from . import position
    from . import potentiostat

else:
# except ModuleNotFoundError:

    from .shims import pump
    from .shims import position
    from .shims import potentiostat
