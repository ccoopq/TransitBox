#!/usr/bin/env python3
"""Git-only credential helper. Reads a temporary process environment, never a file."""
import os
import sys

prompt=' '.join(sys.argv[1:]).lower()
print('x-access-token' if 'username' in prompt else os.environ.get('TRANSITBOX_GIT_TOKEN',''))
