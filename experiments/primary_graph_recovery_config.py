"""Registered design constants for the primary Gaussian graph-recovery study."""

from __future__ import annotations


METHODS = ("sm_l0", "sm_l1", "graphl0", "glasso")
NUMBER_OF_REPLICATIONS = 10
CALIBRATION_CONFIGURATION = ("erdos_renyi", 20, 40)
PENALTY_CONSTANT_GRID = (
    0.03125,
    0.044,
    0.0625,
    0.088,
    0.125,
    0.177,
    0.25,
    0.354,
    0.5,
    0.707,
    1.0,
    1.414,
    2.0,
    2.828,
    4.0,
)

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


def comma_join(values: tuple[object, ...]) -> str:
    """Format a registered tuple for a command-line list argument."""
    return ",".join(
        f"{value:g}" if isinstance(value, float) else str(value)
        for value in values
    )


def configuration_filter(settings: tuple[tuple[str, int, int], ...]) -> str:
    """Format exact configuration filters for the common runner."""
    return ";".join(f"{topology}:{p}:{n}" for topology, p, n in settings)


if __name__ == "__main__":
    print(comma_join(PENALTY_CONSTANT_GRID))
