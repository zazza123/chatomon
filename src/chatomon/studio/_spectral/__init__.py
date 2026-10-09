"""The studio's add-on for Spectral: an atom answers the requests of an evaluation.

It is loaded only when the `spectral` extra is installed. The rest of the studio
imports `Bridge` and `available` from here and nothing else.
"""

from chatomon.studio._spectral._bridge import Bridge
from chatomon.studio._spectral._relay import available

__all__ = ["Bridge", "available"]
