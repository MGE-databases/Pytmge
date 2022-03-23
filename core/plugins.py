# coding: utf-8
# Copyright (c) pytmge Development Team.

"""

"""

import sys
import math


def progressbar(current, total):
    if True:
        percent = '{:.2%}'.format(current / total)
        sys.stdout.write('\r[%-50s] %s' % ('=' * math.floor(current * 50 / total), percent))
        sys.stdout.flush()
        if current == total:
            print()
    return
