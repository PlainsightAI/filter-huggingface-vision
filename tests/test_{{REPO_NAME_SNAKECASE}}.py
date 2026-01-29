#!/usr/bin/env python

import logging
import multiprocessing
import os
import sys
import unittest

from {{REPO_NAME_SNAKECASE}}.filter import {{REPO_NAME_PASCALCASE}}

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv('LOG_LEVEL') or 'INFO').upper())))

VERBOSE   = '-v' in sys.argv or '--verbose' in sys.argv
LOG_LEVEL = logger.getEffectiveLevel()


class Test{{REPO_NAME_PASCALCASE}}(unittest.TestCase):
    def test_{{REPO_NAME_SNAKECASE}}(self):
        if VERBOSE and LOG_LEVEL <= logging.WARNING:
            print()

        # TODO: test here

        pass


try:
    multiprocessing.set_start_method('spawn')  # CUDA doesn't like fork()
except Exception:
    pass

if __name__ == '__main__':
    unittest.main()
