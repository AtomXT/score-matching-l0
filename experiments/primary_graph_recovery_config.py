"""Registered design constants for the primary Gaussian graph-recovery study."""

from __future__ import annotations


NUMBER_OF_REPLICATIONS = 10

EVALUATION_SETTINGS = {
    "roc": (("erdos_renyi", 500, 250),),
}
PANEL_SETTINGS = EVALUATION_SETTINGS
