#!/usr/bin/env python3

import re
import sys
from typing import Tuple


def version() -> Tuple[int, int, int]:
    try:
        version = re.findall('^\d+\.\d+\.\d+', sys.version)[0]
        senior, minor, patch = re.findall('\d+', version)
        return (int(senior), int(minor), int(patch))
    except:
        return (0, 0, 0)


# major, minor, patch = version()


if sys.version_info < (3, 6):
    sys.exit('ERROR: Python 3.6 or above is required.')
else:
    print('Python %d.%d.%d passed checking.' % sys.version_info[:3])
