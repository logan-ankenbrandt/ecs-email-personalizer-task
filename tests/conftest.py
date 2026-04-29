"""Path shim so `from utils.slop_validation import ...` resolves under pytest.

The email-personalizer-task process has no installed package layout; modules
live at the project root (utils/, harness/, etc.). Inserting the project root
on sys.path lets the test suite run from any cwd.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
