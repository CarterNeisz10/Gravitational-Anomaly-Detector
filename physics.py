"""Physical interpretation of validated gravitational strain candidates."""


LIGO_ARM_LENGTH_METERS = 4000.0
SPEED_OF_LIGHT = 299_792_458.0
GRAVITATIONAL_CONSTANT = 6.67430e-11

EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_METERS = 6_371_000.0

REFERENCE_ALTITUDES = {
    "Sea level": 0.0,
    "ISS": 400_000.0,
    "Low Earth orbit": 2_000_000.0,
    "GPS satellite": 20_200_000.0,
    "Geostationary orbit": 35_786_000.0,
}

def calculate_length_perturbation(
    strain,
    arm_length=LIGO_ARM_LENGTH_METERS,
):
    """
    Convert dimensionless gravitational strain into an equivalent
    differential interferometer arm-length perturbation.

    Delta L = h * L
    """
    return strain * arm_length

def gravitational_clock_rate(altitude_meters):
    """
    Calculate the gravitational clock rate at an altitude above Earth
    using the Schwarzschild metric.

    Returns dτ/dt.
    """
    radius = EARTH_RADIUS_METERS + altitude_meters

    return (
        1
        - (
            2
            * GRAVITATIONAL_CONSTANT
            * EARTH_MASS_KG
        )
        / (
            radius
            * SPEED_OF_LIGHT ** 2
        )
    ) ** 0.5

def calculate_altitude_time_dilation():
    """
    Compare gravitational clock rates at reference altitudes
    with a clock at sea level.
    """
    sea_level_rate = gravitational_clock_rate(0.0)

    results = {}

    for name, altitude in REFERENCE_ALTITUDES.items():
        clock_rate = gravitational_clock_rate(
            altitude
        )

        relative_rate = (
            clock_rate / sea_level_rate
        )

        seconds_difference_per_day = (
            relative_rate - 1.0
        ) * 86400.0

        results[name] = {
            "altitude_m": altitude,
            "clock_rate": clock_rate,
            "seconds_per_day_vs_sea_level":
                seconds_difference_per_day,
        }

    return results

def analyze_candidate_physics(
    peak_strain,
):
    """Calculate physical quantities for a validated strain candidate."""
    peak_length_change = calculate_length_perturbation(
        peak_strain
    )
    time_dilation = calculate_altitude_time_dilation()

    print("\nPhysics analysis:")
    print(
        "Peak strain:",
        f"{peak_strain:.6e}",
    )
    print(
        "Equivalent differential arm-length change:",
        f"{peak_length_change:.6e} m",
    )

    print("\nEarth gravitational time dilation:")

    for name, result in time_dilation.items():
        difference = result[
            "seconds_per_day_vs_sea_level"
        ]

        return {
            "peak_strain": peak_strain,
            "peak_length_change_m": peak_length_change,
            "earth_time_dilation": time_dilation,
        }

    return {
        "peak_strain": peak_strain,
        "peak_length_change_m": peak_length_change,
    }

if __name__ == "__main__":
    results = calculate_altitude_time_dilation()

    print("Earth gravitational time dilation")
    print("----------------------------------")

    for name, result in results.items():
        difference = result[
            "seconds_per_day_vs_sea_level"
        ]

        difference_microseconds = (
                difference * 1_000_000
        )

        print(
            f"{name}: "
            f"{difference_microseconds:.3f} "
            "microseconds/day faster than sea level"
        )