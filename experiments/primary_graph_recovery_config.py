"""Registered design constants for the primary Gaussian graph-recovery study."""

from __future__ import annotations


NUMBER_OF_REPLICATIONS = 10

# Each evaluation configuration is owned by one Quest array.  The central
# Erdos--Renyi setting is reused across manuscript panels but fitted only once.
EVALUATION_SETTINGS = {
    "sample_size": (
        ("erdos_renyi", 40, 20),
        ("erdos_renyi", 40, 40),
        ("erdos_renyi", 40, 80),
        ("erdos_renyi", 40, 160),
    ),
    "dimension": (("erdos_renyi", 60, 120),),
    "topology": (("scale_free", 40, 80),),
}

# These are the configurations displayed in each manuscript panel.  The
# evaluation ownership above is deliberately nonoverlapping, whereas the
# display settings below reuse the shared central configuration.
SHARED_CONFIGURATION = ("erdos_renyi", 40, 80)
PANEL_SETTINGS = {
    "sample_size": EVALUATION_SETTINGS["sample_size"],
    "dimension": (SHARED_CONFIGURATION, *EVALUATION_SETTINGS["dimension"]),
    "topology": (SHARED_CONFIGURATION, *EVALUATION_SETTINGS["topology"]),
}
