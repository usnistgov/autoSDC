import os
import sys
import shutil
import typing
import pathlib
import subprocess

EPICS_VERSION = 'CA-3.15.6'

# keep Epics DLLs and executables under ${REPOSITORY_ROOT}/lib
SDC_LIB = pathlib.Path(__file__).resolve().parent.parent / 'lib'
EPICS_LIB = SDC_LIB / EPICS_VERSION
EPICS_EXTRA_BIN = pathlib.Path('home/phoebus/ANJ/win32/bin')
sys.path.append(EPICS_LIB)

print(EPICS_LIB)

CA_GET = EPICS_LIB / 'caget.exe'
CA_PUT = EPICS_LIB / 'caput.exe'
CA_REPEATER = EPICS_LIB / EPICS_EXTRA_BIN / 'caRepeater.exe'

print(f"subprocess.Popen(['{CA_REPEATER}'])")
if CA_REPEATER.is_file():
    subprocess.Popen([CA_REPEATER])

def caget(key: str) -> str:
    response = subprocess.check_output([CA_GET, key])
    print(response)
    return response

def caput(key: str, value: str):
    response = subprocess.check_output([CA_GET, key, value])
    print(response)
    return response

def scp_get_files(pattern: str, remotehost: str = '6bm', dest='./'):
    """ grab file(s) with scp.

    make sure `remotehost` is included in `~/.ssh/config`, and that the
    `localhost` machine can use the `publickey` method to access `remotehost`
    """
    subprocess.check_call(['scp', f'{remotehost}:{pattern}', dest])
