"""
Physical interpretation of validated gravitational strain candidates.

This module converts measured gravitational-wave strain into an equivalent
LIGO differential arm-length perturbation. It also calculates gravitational
time dilation at several reference altitudes above Earth using the
Schwarzschild metric.

The altitude calculations represent gravitational time dilation only and do
not include special-relativistic effects caused by orbital velocity.
"""


# Physical constants
LIGO_ARM_LENGTH_METERS = 4000.0
SPEED_OF_LIGHT = 299_792_458.0
GRAVITATIONAL_CONSTANT = 6.67430e-11

EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_METERS = 6_371_000.0
SECONDS_PER_DAY = 86_400.0
MICROSECONDS_PER_SECOND = 1_000_000.0

# Representative altitudes above Earth's surface
REFERENCE_ALTITUDES = {
    "Sea level": 0.0,
    "ISS": 400_000.0,
    "Low Earth orbit": 2_000_000.0,
    "GPS satellite": 20_200_000.0,
    "Geostationary orbit": 35_786_000.0,
}


"""
    Convert gravitational strain into an equivalent LIGO arm-length change.

    Gravitational-wave strain is dimensionless and relates differential
    interferometer arm-length change to the detector arm length through:

    Delta L = h * L

    Args:
        strain: Dimensionless gravitational-wave strain.
        arm_length: Interferometer arm length in meters.

    Returns:
        float: Equivalent differential arm-length change in meters.
"""
def calculate_length_perturbation(strain, arm_length=LIGO_ARM_LENGTH_METERS):
    return strain * arm_length


"""
    Calculate the gravitational clock rate at an altitude above Earth.

    The calculation uses the Schwarzschild metric for a stationary clock at
    a given distance from Earth's center. Orbital velocity and other
    special-relativistic effects are not included.

    Args:
        altitude_meters: Altitude above Earth's surface in meters.

    Returns:
        float: Proper-time rate dτ/dt at the specified altitude.
"""
def gravitational_clock_rate(altitude_meters):
    radius = EARTH_RADIUS_METERS + altitude_meters

    schwarzschild_term = (
        2 * GRAVITATIONAL_CONSTANT * EARTH_MASS_KG
        / (radius * SPEED_OF_LIGHT ** 2)
    )

    return (1 - schwarzschild_term) ** 0.5


"""
    Compare gravitational clock rates at several reference altitudes.

    Each clock rate is compared with a stationary clock at sea level. The
    resulting difference is expressed as seconds gained per day relative
    to the sea-level clock.

    Returns:
        dict: Clock rates and daily gravitational time differences for each
        reference altitude.
"""
def calculate_altitude_time_dilation():
    sea_level_rate = gravitational_clock_rate(0.0)
    results = {}

    for name, altitude in REFERENCE_ALTITUDES.items():
        clock_rate = gravitational_clock_rate(altitude)
        relative_rate = clock_rate / sea_level_rate

        seconds_difference_per_day = (
            relative_rate - 1.0
        ) * SECONDS_PER_DAY

        results[name] = {
            "altitude_m": altitude,
            "clock_rate": clock_rate,
            "seconds_per_day_vs_sea_level": seconds_difference_per_day,
        }

    return results


"""
    Calculate and display physical quantities for a validated strain candidate.

    The candidate's peak strain is converted into an equivalent LIGO
    differential arm-length change. Gravitational clock-rate differences at
    several Earth altitudes are also calculated as physical reference values.

    Args:
        peak_strain: Peak calibrated strain of the validated candidate.

    Returns:
        dict: Candidate strain, equivalent arm-length change, and Earth
        gravitational time-dilation reference values.
"""
def analyze_candidate_physics(peak_strain):
    peak_length_change = calculate_length_perturbation(peak_strain)
    time_dilation = calculate_altitude_time_dilation()

    print("\nPhysics analysis:")
    print("Peak strain:", f"{peak_strain:.6e}")
    print(
        "Equivalent differential arm-length change:",
        f"{peak_length_change:.6e} m",
    )

    print("\nEarth gravitational time dilation (gravity only):")

    for name, result in time_dilation.items():
        difference_microseconds = (
            result["seconds_per_day_vs_sea_level"]
            * MICROSECONDS_PER_SECOND
        )

        print(
            f"{name}: {difference_microseconds:.3f} "
            "microseconds/day faster than sea level"
        )

    return {
        "peak_strain": peak_strain,
        "peak_length_change_m": peak_length_change,
        "earth_time_dilation": time_dilation,
    }


if __name__ == "__main__":
    results = calculate_altitude_time_dilation()

    print("Earth gravitational time dilation (gravity only)")
    print("------------------------------------------------")

    for name, result in results.items():
        difference_microseconds = (
            result["seconds_per_day_vs_sea_level"]
            * MICROSECONDS_PER_SECOND
        )

        print(
            f"{name}: {difference_microseconds:.3f} "
            "microseconds/day faster than sea level"
        )