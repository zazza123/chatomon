"""`python -m chatomon.studio` is `chatomon studio`."""

import sys

from chatomon._cli import main

sys.exit(main(["studio", *sys.argv[1:]]))
