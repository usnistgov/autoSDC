import sys

if sys.platform == 'win32':

    from . import position
    from . import experiment
    from . import potentiostat

else:

    from .shims import position
    from .shims import experiment
    from .shims import potentiostat
