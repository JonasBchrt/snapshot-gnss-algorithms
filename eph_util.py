"""GNSS utility functions, mostly based on satellite ephemerides.

Author: Jonas Beuchert
"""

try:
    import autograd.numpy as np
except(ImportError):
    print("""Package 'autograd' not found. 'autograd.numpy' is necessary for
          coarse-time navigation via maximum-likelihood estimation. Falling
          back to 'numpy'.""")
    import numpy as np
import pymap3d as pm
try:
    import mkl_fft as fft_lib
except(ImportError):
    print("""Package 'mkl_fft' not found. Consider installing 'mkl_fft' with
          'conda install -c intel mkl_fft' for faster FFT and IFFT. Falling
          back to 'numpy.fft'.""")
    import numpy.fft as fft_lib


def get_sat_pos_vel_acc(t, eph):
    """Calculate positions, velocities, and accelerations of satellites.

    Accepts arrays for t / eph, i.e., can calculate multiple points in time
    / multiple satellites at once.

    Does not interpolate GLONASS.

    Implemented according to
    Thompson, Blair F., et al. “Computing GPS Satellite Velocity and
    Acceleration from the Broadcast Navigation Message.” Annual of
    Navigation, vol. 66, no. 4, 2019, pp. 769–779.
    https://www.gps.gov/technical/icwg/meetings/2019/09/GPS-SV-velocity-and-acceleration.pdf

    Inputs:
        t - GPS time(s) [s] (ignored for SBAS)
        eph - Ephemeris as array(s)

    Outputs:
        positions - Satellite position(s) in ECEF XYZ as array(s) [m]
        velocities - Satellite velocity/ies in ECEF XYZ as array(s) [m/s]
        accelerations - Sat. acceleration(s) in ECEF XYZ as array(s) [m/s^2]

    Author: Jonas Beuchert
    """
    if not np.isnan(eph[2]).any():  # No SBAS / GLONASS

        t = np.mod(t, 7 * 24 * 60 * 60)

        cic = eph[13]  # "cic"]
        crs = eph[10]  # "crs"]
        Omega0 = eph[15]  # "Omega0"]
        Deltan = eph[4]  # "Deltan"]
        cis = eph[14]  # "cis"]
        M0 = eph[2]  # "M0"]
        i0 = eph[11]  # "i0"]
        cuc = eph[7]  # "cuc"]
        crc = eph[9]  # "crc"]
        e = eph[5]  # "e"]
        Omega = eph[6]  # "Omega"]
        cus = eph[8]  # "cus"]
        OmegaDot = eph[16]  # "OmegaDot"]
        sqrtA = eph[3]  # "sqrtA"]
        IDOT = eph[12]  # "IDOT"]
        toe = eph[20]  # "toe"]

        # Broadcast Navigation User Equations

        # WGS 84 value of the earth’s gravitational constant for GPS user [m^3/s^2]
        mu = 3.986005e14

        # WGS 84 value of the earth’s rotation rate [rad/s]
        OmegaeDot = 7.2921151467e-5

        # Semi-major axis
        A = sqrtA ** 2

        # Computed mean motion [rad/s]
        n0 = np.sqrt(mu / A ** 3)

        # Time from ephemeris reference epoch
        tk = np.array(t - toe)
        # t is GPS system time at time of transmission, i.e., GPS time corrected
        # for transit time (range/speed of light). Furthermore, tk shall be the
        # actual total time difference between the time t and the epoch time toe,
        # and must account for beginning or end of week crossovers. That is, if tk
        # is greater than 302,400 seconds, subtract 604,800 seconds from tk. If tk
        # is less than -302,400 seconds, add 604,800 seconds to tk.
        with np.nditer(tk, op_flags=["readwrite"]) as it:
            for tk_i in it:
                if tk_i > 302400:
                    tk_i[...] = tk_i - 604800
                elif tk_i < -302400:
                    tk_i[...] = tk_i + 604800

        # Corrected mean motion
        n = n0 + Deltan

        # Mean anomaly
        Mk = M0 + n * tk

        # Kepler’s equation (Mk = Ek - e*np.sin(Ek)) solved for eccentric anomaly
        # (Ek) by iteration:
        # Initial value [rad]
        Ek = Mk
        # Refined value, three iterations, (j = 0,1,2)
        for j in range(3):
            Ek = Ek + (Mk - Ek + e * np.sin(Ek)) / (1 - e * np.cos(Ek))

        # True anomaly (unambiguous quadrant)
        nuk = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(Ek / 2))

        # Argument of Latitude
        Phik = nuk + Omega

        # Argument of Latitude Correction
        deltauk = cus * np.sin(2 * Phik) + cuc * np.cos(2 * Phik)

        # Radius Correction
        deltark = crs * np.sin(2 * Phik) + crc * np.cos(2 * Phik)

        # Inclination Correction
        deltaik = cis * np.sin(2 * Phik) + cic * np.cos(2 * Phik)

        # Corrected Argument of Latitude
        uk = Phik + deltauk

        # Corrected Radius
        rk = A * (1 - e * np.cos(Ek)) + deltark

        # Corrected Inclination
        ik = i0 + deltaik + IDOT * tk

        # Positions in Orbital Plane
        xkDash = rk * np.cos(uk)
        ykDash = rk * np.sin(uk)

        # Corrected longitude of ascending node
        Omegak = Omega0 + (OmegaDot - OmegaeDot) * tk - OmegaeDot * toe

        # Earth-fixed coordinates
        xk = xkDash * np.cos(Omegak) - ykDash * np.cos(ik) * np.sin(Omegak)
        yk = xkDash * np.sin(Omegak) + ykDash * np.cos(ik) * np.cos(Omegak)
        zk = ykDash * np.sin(ik)

        # SV Velocity

        # Eccentric anomaly rate
        EkDot = n / (1 - e * np.cos(Ek))

        # True anomaly rate
        nukDot = EkDot * np.sqrt(1 - e ** 2) / (1 - e * np.cos(Ek))

        # Corrected Inclination rate
        dik_dt = IDOT + 2 * nukDot * (
            cis * np.cos(2 * Phik) - cic * np.sin(2 * Phik)
        )

        # Corrected Argument of Latitude rate
        ukDot = nukDot + 2 * nukDot * (
            cus * np.cos(2 * Phik) - cuc * np.sin(2 * Phik)
        )

        # Corrected Radius rate
        rkDot = e * A * EkDot * np.sin(Ek) + 2 * nukDot * (
            crs * np.cos(2 * Phik) - crc * np.sin(2 * Phik)
        )

        # Longitude of ascending node rate
        OmegakDot = OmegaDot - OmegaeDot

        # In-plane x velocity
        xkDashDot = rkDot * np.cos(uk) - rk * ukDot * np.sin(uk)

        # In-plane y velocity
        ykDashDot = rkDot * np.sin(uk) + rk * ukDot * np.cos(uk)

        # Earth-fixed x velocity [m/s]
        xkDot = (
            -xkDash * OmegakDot * np.sin(Omegak)
            + xkDashDot * np.cos(Omegak)
            - ykDashDot * np.sin(Omegak) * np.cos(ik)
            - ykDash
            * (
                OmegakDot * np.cos(Omegak) * np.cos(ik)
                - dik_dt * np.sin(Omegak) * np.sin(ik)
            )
        )

        # Earth-fixed y velocity [m/s]
        ykDot = (
            xkDash * OmegakDot * np.cos(Omegak)
            + xkDashDot * np.sin(Omegak)
            + ykDashDot * np.cos(Omegak) * np.cos(ik)
            - ykDash
            * (
                OmegakDot * np.sin(Omegak) * np.cos(ik)
                + dik_dt * np.cos(Omegak) * np.sin(ik)
            )
        )

        # Earth-fixed z velocity [m/s]
        zkDot = ykDash * dik_dt * np.cos(ik) + ykDashDot * np.sin(ik)

        # SV Acceleration

        # WGS 84 Earth equatorial radius [m]
        RE = 6378137.0

        # Oblate Earth gravity coefficient
        J2 = 0.0010826262

        # Oblate Earth acceleration factor
        F = -3 / 2 * J2 * mu / rk ** 2 * (RE / rk) ** 2

        # Earth-fixed x acceleration [m/s^2]
        xkDotDot = (
            -mu * xk / rk ** 3
            + F * ((1 - 5 * (zk / rk) ** 2) * xk / rk)
            + 2 * ykDot * OmegaeDot
            + xk * OmegaeDot ** 2
        )

        # Earth-fixed y acceleration [m/s^2]
        ykDotDot = (
            -mu * yk / rk ** 3
            + F * ((1 - 5 * (zk / rk) ** 2) * yk / rk)
            - 2 * xkDot * OmegaeDot
            + yk * OmegaeDot ** 2
        )

        # Earth-fixed z acceleration [m/s^2]
        zkDotDot = -mu * zk / rk ** 3 + F * ((3 - 5 * (zk / rk) ** 2) * zk / rk)

        positions = np.array([xk, yk, zk]).T
        velocities = np.array([xkDot, ykDot, zkDot]).T
        accelerations = np.array([xkDotDot, ykDotDot, zkDotDot]).T

    else:  # SBAS

        positions = 1.0e3 * np.array([eph[3], eph[6], eph[9]]).T
        velocities = 1.0e3 * np.array([eph[4], eph[7], eph[10]]).T
        accelerations = 1.0e3 * np.array([eph[5], eph[8], eph[11]]).T

        if isinstance(t, np.ndarray) and len(eph.shape) == 1:
            n_times = t.shape[0]
            positions = np.tile(positions, (n_times, 1))
            velocities = np.tile(velocities, (n_times, 1))
            accelerations = np.tile(accelerations, (n_times, 1))

    return positions, velocities, accelerations


def get_sat_pos_vel(t, eph):
    """Calculate positions and velocities of satellites.

    Accepts arrays for t / eph, i.e., can calculate multiple points in time
    / multiple satellites at once.

    Does not interpolate GLONASS.

    Implemented according to
    Thompson, Blair F., et al. “Computing GPS Satellite Velocity and
    Acceleration from the Broadcast Navigation Message.” Annual of
    Navigation, vol. 66, no. 4, 2019, pp. 769–779.
    https://www.gps.gov/technical/icwg/meetings/2019/09/GPS-SV-velocity-and-acceleration.pdf

    Inputs:
        t - GPS time(s) [s] (ignored for SBAS)
        eph - Ephemeris as array(s)

    Outputs:
        positions - Satellite position(s) in ECEF XYZ as array(s) [m]
        velocities - Satellite velocity/ies in ECEF XYZ as array(s) [m/s]

    Author: Jonas Beuchert
    """
    if not np.isnan(eph[2]).any():  # No SBAS / GLONASS

        t = np.mod(t, 7 * 24 * 60 * 60)

        cic = eph[13]  # "cic"]
        crs = eph[10]  # "crs"]
        Omega0 = eph[15]  # "Omega0"]
        Deltan = eph[4]  # "Deltan"]
        cis = eph[14]  # "cis"]
        M0 = eph[2]  # "M0"]
        i0 = eph[11]  # "i0"]
        cuc = eph[7]  # "cuc"]
        crc = eph[9]  # "crc"]
        e = eph[5]  # "e"]
        Omega = eph[6]  # "Omega"]
        cus = eph[8]  # "cus"]
        OmegaDot = eph[16]  # "OmegaDot"]
        sqrtA = eph[3]  # "sqrtA"]
        IDOT = eph[12]  # "IDOT"]
        toe = eph[20]  # "toe"]

        # Broadcast Navigation User Equations

        # WGS 84 value of the earth’s gravitational constant for GPS user [m^3/s^2]
        mu = 3.986005e14

        # WGS 84 value of the earth’s rotation rate [rad/s]
        OmegaeDot = 7.2921151467e-5

        # Semi-major axis
        A = sqrtA ** 2

        # Computed mean motion [rad/s]
        n0 = np.sqrt(mu / A ** 3)

        # Time from ephemeris reference epoch
        tk = np.array(t - toe)
        # t is GPS system time at time of transmission, i.e., GPS time corrected
        # for transit time (range/speed of light). Furthermore, tk shall be the
        # actual total time difference between the time t and the epoch time toe,
        # and must account for beginning or end of week crossovers. That is, if tk
        # is greater than 302,400 seconds, subtract 604,800 seconds from tk. If tk
        # is less than -302,400 seconds, add 604,800 seconds to tk.
        with np.nditer(tk, op_flags=["readwrite"]) as it:
            for tk_i in it:
                if tk_i > 302400:
                    tk_i[...] = tk_i - 604800
                elif tk_i < -302400:
                    tk_i[...] = tk_i + 604800

        # Corrected mean motion
        n = n0 + Deltan

        # Mean anomaly
        Mk = M0 + n * tk

        # Kepler’s equation (Mk = Ek - e*np.sin(Ek)) solved for eccentric anomaly
        # (Ek) by iteration:
        # Initial value [rad]
        Ek = Mk
        # Refined value, three iterations, (j = 0,1,2)
        for j in range(3):
            Ek = Ek + (Mk - Ek + e * np.sin(Ek)) / (1 - e * np.cos(Ek))

        # True anomaly (unambiguous quadrant)
        nuk = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(Ek / 2))

        # Argument of Latitude
        Phik = nuk + Omega

        # Argument of Latitude Correction
        deltauk = cus * np.sin(2 * Phik) + cuc * np.cos(2 * Phik)

        # Radius Correction
        deltark = crs * np.sin(2 * Phik) + crc * np.cos(2 * Phik)

        # Inclination Correction
        deltaik = cis * np.sin(2 * Phik) + cic * np.cos(2 * Phik)

        # Corrected Argument of Latitude
        uk = Phik + deltauk

        # Corrected Radius
        rk = A * (1 - e * np.cos(Ek)) + deltark

        # Corrected Inclination
        ik = i0 + deltaik + IDOT * tk

        # Positions in Orbital Plane
        xkDash = rk * np.cos(uk)
        ykDash = rk * np.sin(uk)

        # Corrected longitude of ascending node
        Omegak = Omega0 + (OmegaDot - OmegaeDot) * tk - OmegaeDot * toe

        # Earth-fixed coordinates
        xk = xkDash * np.cos(Omegak) - ykDash * np.cos(ik) * np.sin(Omegak)
        yk = xkDash * np.sin(Omegak) + ykDash * np.cos(ik) * np.cos(Omegak)
        zk = ykDash * np.sin(ik)

        # SV Velocity

        # Eccentric anomaly rate
        EkDot = n / (1 - e * np.cos(Ek))

        # True anomaly rate
        nukDot = EkDot * np.sqrt(1 - e ** 2) / (1 - e * np.cos(Ek))

        # Corrected Inclination rate
        dik_dt = IDOT + 2 * nukDot * (
            cis * np.cos(2 * Phik) - cic * np.sin(2 * Phik)
        )

        # Corrected Argument of Latitude rate
        ukDot = nukDot + 2 * nukDot * (
            cus * np.cos(2 * Phik) - cuc * np.sin(2 * Phik)
        )

        # Corrected Radius rate
        rkDot = e * A * EkDot * np.sin(Ek) + 2 * nukDot * (
            crs * np.cos(2 * Phik) - crc * np.sin(2 * Phik)
        )

        # Longitude of ascending node rate
        OmegakDot = OmegaDot - OmegaeDot

        # In-plane x velocity
        xkDashDot = rkDot * np.cos(uk) - rk * ukDot * np.sin(uk)

        # In-plane y velocity
        ykDashDot = rkDot * np.sin(uk) + rk * ukDot * np.cos(uk)

        # Earth-fixed x velocity [m/s]
        xkDot = (
            -xkDash * OmegakDot * np.sin(Omegak)
            + xkDashDot * np.cos(Omegak)
            - ykDashDot * np.sin(Omegak) * np.cos(ik)
            - ykDash
            * (
                OmegakDot * np.cos(Omegak) * np.cos(ik)
                - dik_dt * np.sin(Omegak) * np.sin(ik)
            )
        )

        # Earth-fixed y velocity [m/s]
        ykDot = (
            xkDash * OmegakDot * np.cos(Omegak)
            + xkDashDot * np.sin(Omegak)
            + ykDashDot * np.cos(Omegak) * np.cos(ik)
            - ykDash
            * (
                OmegakDot * np.sin(Omegak) * np.cos(ik)
                + dik_dt * np.cos(Omegak) * np.sin(ik)
            )
        )

        # Earth-fixed z velocity [m/s]
        zkDot = ykDash * dik_dt * np.cos(ik) + ykDashDot * np.sin(ik)

        positions = np.array([xk, yk, zk]).T
        velocities = np.array([xkDot, ykDot, zkDot]).T

    else:  # SBAS

        positions = 1.0e3 * np.array([eph[3], eph[6], eph[9]]).T
        velocities = 1.0e3 * np.array([eph[4], eph[7], eph[10]]).T

        if isinstance(t, np.ndarray) and len(eph.shape) == 1:
            n_times = t.shape[0]
            positions = np.tile(positions, (n_times, 1))
            velocities = np.tile(velocities, (n_times, 1))

    return positions, velocities


def get_sat_pos(t, eph):
    """Calculate positions of satellites.

    Accepts arrays for t / eph, i.e., can calculate multiple points in time
    / multiple satellites at once.

    Does not interpolate GLONASS.

    Implemented according to
    Thompson, Blair F., et al. “Computing GPS Satellite Velocity and
    Acceleration from the Broadcast Navigation Message.” Annual of
    Navigation, vol. 66, no. 4, 2019, pp. 769–779.
    https://www.gps.gov/technical/icwg/meetings/2019/09/GPS-SV-velocity-and-acceleration.pdf

    Inputs:
        t - GPS time(s) [s] (ignored for SBAS)
        eph - Ephemeris as array(s)

    Outputs:
        positions - Satellite position(s) in ECEF XYZ as array(s) [m]

    Author: Jonas Beuchert
    """
    if not np.isnan(eph[2]).any():  # No SBAS / GLONASS

        t = np.mod(t, 7 * 24 * 60 * 60)

        cic = eph[13]  # "cic"]
        crs = eph[10]  # "crs"]
        Omega0 = eph[15]  # "Omega0"]
        Deltan = eph[4]  # "Deltan"]
        cis = eph[14]  # "cis"]
        M0 = eph[2]  # "M0"]
        i0 = eph[11]  # "i0"]
        cuc = eph[7]  # "cuc"]
        crc = eph[9]  # "crc"]
        e = eph[5]  # "e"]
        Omega = eph[6]  # "Omega"]
        cus = eph[8]  # "cus"]
        OmegaDot = eph[16]  # "OmegaDot"]
        sqrtA = eph[3]  # "sqrtA"]
        IDOT = eph[12]  # "IDOT"]
        toe = eph[20]  # "toe"]

        # Broadcast Navigation User Equations

        # WGS 84 value of the earth’s gravitational constant for GPS user [m^3/s^2]
        mu = 3.986005e14

        # WGS 84 value of the earth’s rotation rate [rad/s]
        OmegaeDot = 7.2921151467e-5

        # Semi-major axis
        A = sqrtA ** 2

        # Computed mean motion [rad/s]
        n0 = np.sqrt(mu / A ** 3)

        # Time from ephemeris reference epoch
        tk = np.array(t - toe)
        # t is GPS system time at time of transmission, i.e., GPS time corrected
        # for transit time (range/speed of light). Furthermore, tk shall be the
        # actual total time difference between the time t and the epoch time toe,
        # and must account for beginning or end of week crossovers. That is, if tk
        # is greater than 302,400 seconds, subtract 604,800 seconds from tk. If tk
        # is less than -302,400 seconds, add 604,800 seconds to tk.
        try:
            with np.nditer(tk, op_flags=["readwrite"]) as it:
                for tk_i in it:
                    if tk_i > 302400:
                        tk_i[...] = tk_i - 604800
                    elif tk_i < -302400:
                        tk_i[...] = tk_i + 604800
        except TypeError:
            for idx in np.arange(tk.shape[0]):
                if tk[idx] > 302400:
                    tk[idx] = tk[idx] - 604800
                elif tk[idx] < -302400:
                    tk[idx] = tk[idx] + 604800

        # Corrected mean motion
        n = n0 + Deltan

        # Mean anomaly
        Mk = M0 + n * tk

        # Kepler’s equation (Mk = Ek - e*np.sin(Ek)) solved for eccentric anomaly
        # (Ek) by iteration:
        # Initial value [rad]
        Ek = Mk
        # Refined value, three iterations, (j = 0,1,2)
        for j in range(3):
            Ek = Ek + (Mk - Ek + e * np.sin(Ek)) / (1 - e * np.cos(Ek))

        # True anomaly (unambiguous quadrant)
        nuk = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(Ek / 2))

        # Argument of Latitude
        Phik = nuk + Omega

        # Argument of Latitude Correction
        deltauk = cus * np.sin(2 * Phik) + cuc * np.cos(2 * Phik)

        # Radius Correction
        deltark = crs * np.sin(2 * Phik) + crc * np.cos(2 * Phik)

        # Inclination Correction
        deltaik = cis * np.sin(2 * Phik) + cic * np.cos(2 * Phik)

        # Corrected Argument of Latitude
        uk = Phik + deltauk

        # Corrected Radius
        rk = A * (1 - e * np.cos(Ek)) + deltark

        # Corrected Inclination
        ik = i0 + deltaik + IDOT * tk

        # Positions in Orbital Plane
        xkDash = rk * np.cos(uk)
        ykDash = rk * np.sin(uk)

        # Corrected longitude of ascending node
        Omegak = Omega0 + (OmegaDot - OmegaeDot) * tk - OmegaeDot * toe

        # Earth-fixed coordinates
        xk = xkDash * np.cos(Omegak) - ykDash * np.cos(ik) * np.sin(Omegak)
        yk = xkDash * np.sin(Omegak) + ykDash * np.cos(ik) * np.cos(Omegak)
        zk = ykDash * np.sin(ik)

        positions = np.array([xk, yk, zk]).T

    else:  # SBAS

        positions = 1.0e3 * np.array([eph[3], eph[6], eph[9]]).T

        if isinstance(t, np.ndarray) and len(eph.shape) == 1:
            n_times = t.shape[0]
            positions = np.tile(positions, (n_times, 1))

    return positions


def get_sat_pos_sp3(gps_time, sp3, sv_list, system=None):
    """Calculate positions of satellites from precise orbits (SP3 file).

    Inputs:
        gps_time - GPS times [s] as numpy array
        sp3 - Precise orbit supporting points as pandas.DataFrame from read_sp3
        sv_list - Satellite indices (PRNs)
        system - Character representing satellite navigation system:
                     'G' - GPS
                     'S' - SBAS
                     'R' - GLONASS
                     'E' - Galileo
                     'C' - BeiDou
                     'J' - QZSS
                     'I' - NavIC
                     None - Use character of 1st SP3 entry (default)
    Output:
        position - Satellite positions in ECEF XYZ as Nx3 array [m]

    Author: Jonas Beuchert
    Based on https://github.com/GNSSpy-Project/gnsspy/blob/fce079af37d585dc757c56539a98cc0dfe66f9de/gnsspy/position/interpolation.py
    """
    def coord_interp(parameter):
        """
        Interpolation of SP3 coordinates.

        Fit polynomial to 4 hours (14400 seconds) period of SP3 Cartesian
        coordinates and return to the interpolated coordinates.

        Input:
            parameter - Polynomial coefficients from numpy polyfit function
                        (numpy array)
        Output:
            interp_coord - Interpolated coordinates (numpy array)
        """
        epoch = 0.0
        time = np.array([
            epoch**deg for deg in range(len(parameter)-1, -1, -1)
            ])
        return np.matmul(parameter, time)

    import pandas as pd

    # Degree of polynomial interpolation, lower than 11 not recommended, above
    # 16 not applicable for 15 minute intervals
    poly_degree = 16

    # Convert time
    referenceDate = np.datetime64('1980-01-06')  # GPS reference date
    utc = np.timedelta64(int((gps_time[0]) * 1e9), 'ns') + referenceDate

    # Check if numpy array has been passed for sv_list
    if isinstance(sv_list, np.ndarray):
        # Convert numpy array to list
        sv_list = sv_list.astype(int).tolist()
    # Check if system character must be read from SP3
    if system is None:
        # Use system character of 1st entry
        system = sp3.index[0][1][0]
    # Convert sv_list to strings
    sv_list = [system + "{:02d}".format(sv) for sv in sv_list]

    # Get time stamps
    epoch_values = sp3.index.get_level_values("Epoch").unique()
    # Difference between 2 time stamps
    deltaT = epoch_values[1]-epoch_values[0]

    # Get 17 data points
    epoch_start = utc - np.timedelta64(2, 'h')
    epoch_stop = utc + np.timedelta64(2, 'h') + deltaT
    sp3_temp = sp3.loc[(slice(epoch_start, epoch_stop))].copy()
    sp3_temp = sp3_temp.reorder_levels(["SV", "Epoch"])

    # Initialize result
    epoch_interp_List = np.zeros(shape=(len(sv_list), 3))
    # Iterate over all satellites
    for svIndex, sv in enumerate(sv_list):
        fitTime = np.array([
            (np.datetime64(t) - referenceDate) / np.timedelta64(1, 's')
            - gps_time[svIndex]
            for t in sp3_temp.loc[sv_list[0]].index.get_level_values("Epoch")
            ])
        epoch_number = len(sp3_temp.loc[sv])
        if epoch_number <= poly_degree:
            print("Warning: Not enough epochs to predict for satellite",
                  sv, "| Epoch Count:", epoch_number, " - Polynomial Degree:",
                  poly_degree)
            epoch_interp_List[svIndex, :] = np.full(shape=3, fill_value=None)
            continue
        # if epoch_number != 17:
        #     fitTime = [(sp3_temp.loc[sv].index[t]
        #                - sp3_temp.loc[sv].index[0]).seconds
        #                for t in range(epoch_number)]
        # Fit sp3 coordinates to 16 deg polynomial
        fitX = np.polyfit(fitTime, sp3_temp.loc[sv].X.copy(), deg=poly_degree)
        fitY = np.polyfit(fitTime, sp3_temp.loc[sv].Y.copy(), deg=poly_degree)
        fitZ = np.polyfit(fitTime, sp3_temp.loc[sv].Z.copy(), deg=poly_degree)

        # sidereal_day = 0.99726956634
        # period = sidereal_day
        # P0 = 2.0*np.pi / period
        # gps_day_sec = np.mod(gps_time, 24*60*60)
        # gps_rel_time = gps_day_sec / 86400.0
        # Timei = fitTime + gps_day_sec
        # Timei = Timei / 86400.0
        # Xi = sp3_temp.loc[sv].X.copy()
        # Yi = sp3_temp.loc[sv].Y.copy()
        # Zi = sp3_temp.loc[sv].Z.copy()
        # A = np.zeros((poly_degree+1, poly_degree+1))
        # A[:, 0] = np.ones(poly_degree+1)
        # B = np.zeros(poly_degree+1)
        # B[0] = 1.0
        # ND = np.int((poly_degree) / 2)
        # for i in np.arange(ND):
        #     kk = 1 + i*2
        #     P = P0 * (i+1)
        #     A[:, kk] = np.sin(P*Timei)
        #     A[:, kk+1] = np.cos(P*Timei)
        #     B[kk] = np.sin(P*gps_rel_time)
        #     B[kk+1] = np.cos(P*gps_rel_time)
        # XCoeffs = np.linalg.lstsq(A, Xi, rcond=None)[0]
        # YCoeffs = np.linalg.lstsq(A, Yi, rcond=None)[0]
        # ZCoeffs = np.linalg.lstsq(A, Zi, rcond=None)[0]
        # epoch_interp_List[svIndex, :] = 1000.0*np.array(
        #     [B@XCoeffs, B@YCoeffs, B@ZCoeffs]
        #     )

        # Interpolate coordinates
        x_interp = coord_interp(fitX) * 1000  # km to m
        # x_velocity = _np.array([(x_interp[i+1]-x_interp[i])/interval if (i+1)<len(x_interp) else 0 for i in range(len(x_interp))])
        y_interp = coord_interp(fitY) * 1000  # km to m
        # y_velocity = _np.array([(y_interp[i+1]-y_interp[i])/interval if (i+1)<len(y_interp) else 0 for i in range(len(y_interp))])
        z_interp = coord_interp(fitZ) * 1000  # km to m
        # z_velocity = _np.array([(z_interp[i+1]-z_interp[i])/interval if (i+1)<len(z_interp) else 0 for i in range(len(z_interp))])
        sv_interp = np.vstack((x_interp, y_interp, z_interp))
        epoch_interp_List[svIndex, :] = sv_interp[:, 0]
        # Restore original fitTime in case it has changed
        # fitTime = np.linspace(0.0, deltaT.seconds*16.0, 17)

    return epoch_interp_List


def find_eph(eph, sv, time):
    """Find the proper column in ephemeris array.

    Inputs:
        eph - Ephemeris array
        sv - Satellite index (PRNs)
        time - GPS time of week [s]

    Output:
        icol - Column index, NaN if ephemeris does not contain satellite
    """
    icol = 0
    isat = np.where(eph[0] == sv)[0]
    n = isat.size
    if n == 0:
        return np.NaN
    icol = isat[0]
    dtmin = eph[20, icol] - time
    for t in isat:
        dt = eph[20, t] - time
        if dt < 0:
            if abs(dt) < abs(dtmin):
                icol = t
                dtmin = dt
    return icol


def check_t(time):
    """Account for beginning or end of week crossover.

    Input:
        time        - Time [s]

    Output:
        corrTime    - Corrected time [s]
    """
    half_week = 302400  # [s]

    corrTime = time

    if time > half_week:
        corrTime = time - 2 * half_week
    elif time < -half_week:
        corrTime = time + 2 * half_week
    return corrTime


def check_t_vectorized(time):
    """Account for beginning or end of week crossover.

    Input:
        time        - Time [s], numpy.ndarray

    Output:
        corrTime    - Corrected time [s]
    """
    half_week = 302400  # [s]

    corrTime = time

    corrTime[time > half_week] = time[time > half_week] - 2 * half_week
    corrTime[time < -half_week] = corrTime[time < -half_week] + 2 * half_week
    return corrTime


def get_sat_clk_corr(transmit_time, prn, eph):
    """Compute satellite clock correction time.

    Without relativistic correction.

    Ephemeris provided as array.

    Inputs:
        transmit_time - Actual time when signal was transmitted [s]
        prn - Satellite's PRN index (array)
        eph - Ephemeris array

    Output:
        satClockCorr - Satellite clock corrections [s]

    Author: Jonas Beuchert
    """
    # GPS time with respect to 1980 to time of week (TOW)
    transmit_time = np.mod(transmit_time, 7 * 24 * 60 * 60)
    # Get ephemerides (find column of ephemeris matrix that matches satellite
    # index and time)
    if eph.ndim > 1 and eph.shape[1] != prn.shape[0]:
        col = np.array([find_eph(eph, prn_i, transmit_time) for prn_i in prn])
        eph = eph[:, col]  # Extract column
    # Find initial satellite clock correction
    # Find time difference
    dt = np.array([check_t(transmit_time - eph_20) for eph_20 in eph[20]])
    # Calculate clock correction
    satClkCorr = (eph[1] * dt + eph[19]) * dt + eph[19]  # - eph.T_GD
    # Apply correction
    time = transmit_time - satClkCorr
    # Find time difference
    dt = np.array([check_t(t_eph_20) for t_eph_20 in time - eph[20]])
    # Calculate clock correction
    return (eph[1] * dt + eph[19]) * dt + eph[18]
    # - eph.T_GD


def get_sat_clk_corr_vectorized(transmit_time, prn, eph):
    """Compute satellite clock correction time.

    Without relativistic correction.

    Navigation data provided as 2D NumPy array; transmission time and PRNs
    provided as 1D NumPy array.

    Inputs:
        transmit_time - Actual times when signals were transmitted [s] (Nx1 array)
        prn - Satellite's PRN indices (Nx1 array)
        eph - Matching navigation data (21xN array)

    Output:
        sat_clock_corr - Satellite clock corrections [s] (Nx1 array)

    Author: Jonas Beuchert
    """
    # GPS time with respect to 1980 to time of week (TOW)
    transmit_time = np.mod(transmit_time, 7 * 24 * 60 * 60)
    # Get ephemerides (find column of ephemeris matrix that matches satellite
    # index and time)
    if  eph.shape[1] != prn.shape[0]:
        col = np.array([find_eph(eph, prn_i, transmit_time) for prn_i in prn])
        eph = eph[:, col]  # Extract column
    # Find initial satellite clock correction
    # Find time difference
    dt = check_t_vectorized(transmit_time - eph[20])
    # Calculate clock correction
    satClkCorr = (eph[1] * dt + eph[19]) * dt + eph[19]  # - eph.T_GD
    # Apply correction
    time = transmit_time - satClkCorr
    # Find time difference
    dt = check_t_vectorized(time - eph[20])
    # Calculate clock correction
    return (eph[1] * dt + eph[19]) * dt + eph[18]
    # - eph.T_GD


def get_visible_sats(ht, p, eph, elev_mask=0, prn_list=range(1, 33)):
    """Estimate set of visible satellites.

    Ephemeris provided as array.

    Inputs:
      ht - Receiver time hypothesis [s]
      p - Receiver position hypothesis (latitude, longitude, elevation)
      eph - Ephemeris as matrix
      elev_mask - [Optional] Elevation mask: minimum elevation for satellite
                 to be considered to be visible [degrees], default 0
      prn_list - [Optional] PRNs of satellites to search for, default 1-32
    Output:
      visSat - Indices of visible satellites

    Author: Jonas Beuchert
    """
    ht = np.mod(ht, 7 * 24 * 60 * 60)
    # Crude transmit time estimate [s]
    t = ht - 76.5e-3
    # Empty array for result
    visSat = np.array([], dtype=int)
    # Loop over all satellite indices
    for prn in prn_list:
        col = find_eph(eph, prn, t)
        if not np.isnan(col):
            ephSat = eph[:, col]
            # Get satellite position at estimated transmit time
            satPos = get_sat_pos(t, ephSat)
            if not np.isnan(satPos).any():
                az, elev, slantRange = pm.ecef2aer(
                    satPos[0], satPos[1], satPos[2], p[0], p[1], p[2]
                )
                # Satellites with elevation larger than threshold
                if elev > elev_mask:
                    # Add satellite index to result
                    visSat = np.append(visSat, prn)
    return visSat


def get_doppler(ht, R, k, eph):
    """Calculate expected Doppler [Hz] for given time and position hypothesis.

    Inputs:
        ht - Time hypothesis (receiver) [s]
        R - Receiver position (ECEF) [m,m,m]
        k - Satellite index (PRN)
        eph - Ephemeris

    Output:
        D - Doppler shift frequency [Hz]

    Author: Jonas Beuchert
    """
    # Speed of light [m/s]
    c = 299792458
    # Crude transmit time estimate [s]
    t = ht - 76.5e-3
    # GPS time with respect to 1980 to time of week (TOW)
    tow = np.mod(t, 7 * 24 * 60 * 60)
    # Find column of ephemeris matrix that matches satellite index and time
    col = find_eph(eph, k, tow)
    if np.isnan(col):
        return np.NaN
    else:
        # Extract column
        eph = eph[:, col]
        for it in range(2):  # 2 iterations to refine transmit time estimate
            p = get_sat_pos(t, eph)  # Satellite position estimate [m,m,m]
            d = np.linalg.norm(R - p) / c  # Propagation delay estimate [s]
            t = ht - d  # Transmit time estimate [s]
        L1 = 1575.42e6  # GPS signal frequency [Hz]
        P, V = get_sat_pos_vel(t, eph)  # Satellite velocity [m/s,m/s,m/s]
        lambd = c / L1  # Wave length of transmitted signal
    # Doppler shift (cf. 'Cycle slip detection in single frequency GPS carrier
    # phase observations using expected Doppler shift')
    return (np.dot((R - P) / np.linalg.norm(R - P), V) / lambd)


def generate_ca_code(PRN):
    """Generate one of the GPS, EGNOS, or WAAS satellite C/A codes.

    Input:
        PRN         - PRN number of the sequence

    Output:
        CAcode      - Array containing the desired C/A code sequence (chips)

    Author: Jonas Beuchert
    """
    # Make the code shift array; the shift depends on the PRN number
    # The g2s vector holds the appropriate shift of the g2 code to generate
    # the C/A code (ex. for SV#19 - use a G2 shift of g2s(19) = 471)
    g2s = [
        5,
        6,
        7,
        8,
        17,
        18,
        139,
        140,
        141,
        251,
        252,
        254,
        255,
        256,
        257,
        258,
        469,
        470,
        471,
        472,
        473,
        474,
        509,
        512,
        513,
        514,
        515,
        516,
        859,
        860,
        861,
        862,  # End of shifts for GPS satellites
        145, 175, 52, 21, 237, 235, 886, 657, 634, 762, 355, 1012, 176, 603,
        130, 359, 595, 68, 386  # End of shifts for EGNOS and WAAS satellites
    ]  # For EGNOS and WAAS, subtract 87 from the PRN

    # Adjust EGNOS and WAAS PRNs
    if PRN >= 120:
        PRN = PRN - 87

    if PRN > len(g2s):
        raise Exception(
            "Provided PRN out of range. Only 1-32 and 120-139 supported.")

    # Pick right shift for the given PRN number
    g2shift = g2s[PRN - 1]

    # Generate G1 code

    # Initialize g1 output to speed up the function
    g1 = np.zeros(1023)
    # Load shift register
    reg = -1 * np.ones(10)

    # Generate all G1 signal chips based on the G1 feedback polynomial -----
    for i in range(1023):
        g1[i] = reg[9]
        saveBit = reg[2] * reg[9]
        reg[1:10] = reg[0:9]
        reg[0] = saveBit

    # Generate G2 code

    # Initialize g2 output to speed up the function
    g2 = np.zeros(1023)
    # Load shift register
    reg = -1 * np.ones(10)

    # Generate all G2 signal chips based on the G2 feedback polynomial
    for i in range(1023):
        g2[i] = reg[9]
        saveBit = reg[1] * reg[2] * reg[5] * reg[7] * reg[8] * reg[9]
        reg[1:10] = reg[0:9]
        reg[0] = saveBit

    # Shift G2 code
    # The idea: g2 = concatenate[ g2_right_part, g2_left_part ]
    g2 = np.concatenate((g2[1023 - g2shift: 1023], g2[0: 1023 - g2shift]))

    # Form single sample C/A code by multiplying G1 and G2
    return -(g1 * g2)


def generate_e1_code(prn, fs, pilot=False):
    """Generate and sample Galileo signal that is transmitted in E1 band.

    Inputs:
        prn - Index of satellite (1-50)
        fs - Sampling rate [Hz]
        pilot - Flag if data component E1B (pilot=False) or primary pilot
                component E1C (pilot=True) is returned, default=False.
    Output:
        replica - Binary sampled E1 Open Service data signal

    Author: Jonas Beuchert
    """
    # chip_rate = 1023000
    # Number of samples per code sequence
    n = fs * 4e-3

    # Number of chips per code sequence
    code_length = 4092.0

    # Distance in chips between two samples (increment)
    incr = code_length / n

    if not pilot:
        # Obtain E1B (data) code
        c = e1b(prn)
    else:
        # Obtain primary E1C (pilot) code
        c = e1c(prn)

    # Find indices of samples in E1-B / E1-C code
    idx = incr * np.arange(int(n))
    idx = np.floor(idx)
    idx = np.mod(idx, code_length).astype('int')

    # Sample E1-B code
    x = c[idx]

    e1_code = - 1.0 + 2.0 * x

    # Obtain sampled BOC(1,1)
    boc = boc11(incr, int(n))

    # Multiply both signals
    return e1_code * boc


def boc11(incr, n):
    """Generate and sample binary offset carrier (BOC) of Galileo satellite.

    Inputs:
        incr - Increment; difference in chips between two consecutive samples
        n - Number of samples per code sequence
    Output:
        boc - Sampled binary offset carrier

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/nco.py
    written by Peter Monta.
    """
    c = np.array([-1, 1])
    boc11_length = 2
    idx = incr * np.arange(n)
    idx = idx * 2
    idx = np.floor(idx).astype('int')
    idx = np.mod(idx, boc11_length)
    return c[idx]


e1b_codes = {}


def e1b(prn):
    """Generate unsampled E1B code of Galileo satellite.

    Input:
        prn - Index of satellite
    Output:
        y - E1B code

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/galileo/e1b.py
    written by Peter Monta.
    """
    import e1_strings as es
    if prn not in e1b_codes:
        s = es.e1b_strings[prn]
        n = 4092
        y = np.zeros(n)
        for i in range(n):
            nib = i // 4
            bit = 3 - (i % 4)
            y[i] = (int(s[nib], 16) >> bit) & 1
        e1b_codes[prn] = y
    return e1b_codes[prn]


e1c_codes = {}


def e1c(prn):
    """Generate unsampled E1C code of Galileo satellite.

    Neglect secondary code.

    Input:
        prn - Index of satellite
    Output:
        y - E1C code

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/galileo/e1c.py
    written by Peter Monta.
    """
    # secondary_code = np.array([0,0,1,1,1,0,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,0])
    # secondary_code = 1.0 - 2.0*secondary_code
    import e1_strings as es
    if prn not in e1c_codes:
        s = es.e1c_strings[prn]
        n = 4092
        y = np.zeros(n)
        for i in range(n):
            nib = i // 4
            bit = 3 - (i % 4)
            y[i] = (int(s[nib], 16) >> bit) & 1
        e1c_codes[prn] = y
    return e1c_codes[prn]


def generate_b1c_code(prn, fs, pilot=False):
    """Generate and sample BeiDou signal that is transmitted in B1C band.

    Inputs:
        prn - Index of satellite (1-63)
        fs - Sampling rate [Hz]
        pilot - Flag if data component (pilot=False) or primary pilot component
                (pilot=True) is returned, default=False.
    Output:
        s_b1c - B1C signal, length of 10230 chips at 1.023 MHz chip rate

    Author: Jonas Beuchert
    """
    # Number of samples per code sequence
    n = fs * 10.0e-3

    # Number of chips per code sequence
    code_length = 10230.0

    # Distance in chips between two samples (increment)
    incr = code_length / n

    if not pilot:
        # Obtain B1C_data code
        c = b1c_data(prn)
    else:
        # Obtain primary B1C_pilot code
        c = b1c_pilot(prn)

    # Find indices of samples in B1C_data code
    idx = incr * np.arange(int(n))
    idx = np.floor(idx)
    idx = np.mod(idx, code_length).astype('int')

    # Sample B1C_data code
    x = c[idx]

    b1c_code = - 1.0 + 2.0 * x

    # Obtain sampled BOC
    boc = boc11(incr, int(n))

    # Multiply both signals
    return b1c_code * boc


b1cd_codes = {}


def b1c_data(prn):
    """Generate unsampled BeiDou B1C_data signal.

    Input:
        prn - Index of satellite (1-63)
    Output:
        y - B1C_data, length of 10230 chips at 1.023 MHz chip rate

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/beidou/b1cd.py
    written by Peter Monta.
    """
    from sympy.ntheory import legendre_symbol

    if prn not in b1cd_codes:

        # chip_rate = 1023000
        code_length = 10230

        b1cd_params = {
           1: (2678,699),    2: (4802,694),    3: (958,7318),    4: (859,2127),
           5: (3843,715),    6: (2232,6682),   7: (124,7850),    8: (4352,5495),
           9: (1816,1162),  10: (1126,7682),  11: (1860,6792),  12: (4800,9973),
          13: (2267,6596),  14: (424,2092),   15: (4192,19),    16: (4333,10151),
          17: (2656,6297),  18: (4148,5766),  19: (243,2359),   20: (1330,7136),
          21: (1593,1706),  22: (1470,2128),  23: (882,6827),   24: (3202,693),
          25: (5095,9729),  26: (2546,1620),  27: (1733,6805),  28: (4795,534),
          29: (4577,712),   30: (1627,1929),  31: (3638,5355),  32: (2553,6139),
          33: (3646,6339),  34: (1087,1470),  35: (1843,6867),  36: (216,7851),
          37: (2245,1162),  38: (726,7659),   39: (1966,1156),  40: (670,2672),
          41: (4130,6043),  42: (53,2862),    43: (4830,180),   44: (182,2663),
          45: (2181,6940),  46: (2006,1645),  47: (1080,1582),  48: (2288,951),
          49: (2027,6878),  50: (271,7701),   51: (915,1823),   52: (497,2391),
          53: (139,2606),   54: (3693,822),   55: (2054,6403),  56: (4342,239),
          57: (3342,442),   58: (2592,6769),  59: (1007,2560),  60: (310,2502),
          61: (4203,5072),  62: (455,7268),   63: (4318,341),
        }

        N = 10243
        L = np.array([legendre_symbol(i, N) for i in range(N)])
        L[L == -1] = 0
        L[0] = 0

        w, p = b1cd_params[prn]
        W = np.array([L[k] ^ L[(k+w) % N] for k in range(N)])
        c = np.array([W[(n+p-1) % N] for n in range(code_length)])
        b1cd_codes[prn] = c

    return b1cd_codes[prn]


b1cp_codes = {}
b1cp_secondary_codes = {}


def b1c_pilot(prn, secondary=False):
    """Generate unsampled BeiDou B1C_pilot signal.

    Input:
        prn - Index of satellite (1-63)
        secondary - Flag if primary code (secondary=False) or secondary code
                    (secondary=True) is returned, default=False

    Output:
        y - Primary or secondary B1C_pilot, length of 10230 chips at
            1.023 MHz chip rate and length of 1800 at xxx chip rate,
            respectively.

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/beidou/b1cp.py
    written by Peter Monta.
    """
    from sympy.ntheory import legendre_symbol

    if not secondary:

        if prn not in b1cp_codes:

            # chip_rate = 1023000
            code_length = 10230

            b1cp_params = {
               1: (796,7575),     2: (156,2369),     3: (4198,5688),    4: (3941,539),
               5: (1374,2270),    6: (1338,7306),    7: (1833,6457),    8: (2521,6254),
               9: (3175,5644),   10: (168,7119),    11: (2715,1402),   12: (4408,5557),
              13: (3160,5764),   14: (2796,1073),   15: (459,7001),    16: (3594,5910),
              17: (4813,10060),  18: (586,2710),    19: (1428,1546),   20: (2371,6887),
              21: (2285,1883),   22: (3377,5613),   23: (4965,5062),   24: (3779,1038),
              25: (4547,10170),  26: (1646,6484),   27: (1430,1718),   28: (607,2535),
              29: (2118,1158),   30: (4709,526),    31: (1149,7331),   32: (3283,5844),
              33: (2473,6423),   34: (1006,6968),   35: (3670,1280),   36: (1817,1838),
              37: (771,1989),    38: (2173,6468),   39: (740,2091),    40: (1433,1581),
              41: (2458,1453),   42: (3459,6252),   43: (2155,7122),   44: (1205,7711),
              45: (413,7216),    46: (874,2113),    47: (2463,1095),   48: (1106,1628),
              49: (1590,1713),   50: (3873,6102),   51: (4026,6123),   52: (4272,6070),
              53: (3556,1115),   54: (128,8047),    55: (1200,6795),   56: (130,2575),
              57: (4494,53),     58: (1871,1729),   59: (3073,6388),   60: (4386,682),
              61: (4098,5565),   62: (1923,7160),   63: (1176,2277),
            }

            N = 10243
            L = np.array([legendre_symbol(i, N) for i in range(N)])
            L[L == -1] = 0
            L[0] = 0

            w, p = b1cp_params[prn]
            W = np.array([L[k] ^ L[(k+w) % N] for k in range(N)])
            c = np.array([W[(n+p-1) % N] for n in range(code_length)])
            b1cp_codes[prn] = c

        return b1cp_codes[prn]

    else:

        if prn not in b1cp_secondary_codes:

            b1cp_secondary_params = {
               1: (269,1889),    2: (1448,1268),   3: (1028,1593),   4: (1324,1186),
               5: (822,1239),    6: (5,1930),      7: (155,176),     8: (458,1696),
               9: (310,26),     10: (959,1344),   11: (1238,1271),  12: (1180,1182),
              13: (1288,1381),  14: (334,1604),   15: (885,1333),   16: (1362,1185),
              17: (181,31),     18: (1648,704),   19: (838,1190),   20: (313,1646),
              21: (750,1385),   22: (225,113),    23: (1477,860),   24: (309,1656),
              25: (108,1921),   26: (1457,1173),  27: (149,1928),   28: (322,57),
              29: (271,150),    30: (576,1214),   31: (1103,1148),  32: (450,1458),
              33: (399,1519),   34: (241,1635),   35: (1045,1257),  36: (164,1687),
              37: (513,1382),   38: (687,1514),   39: (422,1),      40: (303,1583),
              41: (324,1806),   42: (495,1664),   43: (725,1338),   44: (780,1111),
              45: (367,1706),   46: (882,1543),   47: (631,1813),   48: (37,228),
              49: (647,2871),   50: (1043,2884),  51: (24,1823),    52: (120,75),
              53: (134,11),     54: (136,63),     55: (158,1937),   56: (214,22),
              57: (335,1768),   58: (340,1526),   59: (661,1402),   60: (889,1445),
              61: (929,1680),   62: (1002,1290),  63: (1149,1245),
            }

            sec_N = 3607
            sec_L = np.array([legendre_symbol(i, sec_N) for i in range(sec_N)])
            sec_L[sec_L == -1] = 0
            sec_L[0] = 0

            sec_code_length = 1800

            w, p = b1cp_secondary_params[prn]
            W = np.array([sec_L[k] ^ sec_L[(k+w) % sec_N] for k in range(sec_N)])
            c = np.array([W[(n+p-1) % sec_N] for n in range(sec_code_length)])
            b1cp_secondary_codes[prn] = c

        return b1cp_secondary_codes[prn]


def generate_l1c_code(prn, fs, pilot=False):
    """Generate and sample GPS signal that is transmitted in L1C band.

    Inputs:
        prn - Index of satellite (1-210)
        fs - Sampling rate [Hz]
        pilot - Flag if data component (pilot=False) or primary pilot component
                (pilot=True) is returned, default=False.
    Output:
        s_l1c - L1C signal, length of 10230 chips at 1.023 MHz chip rate

    Author: Jonas Beuchert
    """
    # Number of samples per code sequence
    n = fs * 10.0e-3

    # Number of chips per code sequence
    code_length = 10230.0

    # Distance in chips between two samples (increment)
    incr = code_length / n

    if not pilot:
        # Obtain L1C_data code
        c = l1c_data(prn)
    else:
        # Obtain L1C_pilot code
        c = l1c_pilot(prn)

    # Find indices of samples in L1C_data code
    idx = incr * np.arange(int(n))
    idx = np.floor(idx)
    idx = np.mod(idx, code_length).astype('int')

    # Sample L1C_data code
    x = c[idx]

    l1c_code = - 1.0 + 2.0 * x

    # Obtain sampled BOC
    boc = boc11(incr, int(n))

    # Multiply both signals
    return l1c_code * boc


l1cd_codes = {}


def l1c_data(prn):
    """Generate unsampled GPS L1C_data signal.

    Input:
        prn - Index of satellite (1-210)
    Output:
        y - L1C_data, length of 10230 chips at 1.023 MHz chip rate

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/gps/l1cd.py
    written by Peter Monta.
    """
    from sympy.ntheory import legendre_symbol

    if prn not in l1cd_codes:

        # chip_rate = 1023000
        # code_length = 10230

        l1cd_params = {
            1: (5097,181),     2: (5110,359),    3: (5079,72),      4: (4403,1110),
            5: (4121,1480),    6: (5043,5034),   7: (5042,4622),    8: (5104,1),
            9: (4940,4547),   10: (5035,826),   11: (4372,6284),   12: (5064,4195),
           13: (5084,368),    14: (5048,1),     15: (4950,4796),   16: (5019,523),
           17: (5076,151),    18: (3736,713),   19: (4993,9850),   20: (5060,5734),
           21: (5061,34),     22: (5096,6142),  23: (4983,190),    24: (4783,644),
           25: (4991,467),    26: (4815,5384),  27: (4443,801),    28: (4769,594),
           29: (4879,4450),   30: (4894,9437),  31: (4985,4307),   32: (5056,5906),
           33: (4921,378),    34: (5036,9448),  35: (4812,9432),   36: (4838,5849),
           37: (4855,5547),   38: (4904,9546),  39: (4753,9132),   40: (4483,403),
           41: (4942,3766),   42: (4813,3),     43: (4957,684),    44: (4618,9711),
           45: (4669,333),    46: (4969,6124),  47: (5031,10216),  48: (5038,4251),
           49: (4740,9893),   50: (4073,9884),  51: (4843,4627),   52: (4979,4449),
           53: (4867,9798),   54: (4964,985),   55: (5025,4272),   56: (4579,126),
           57: (4390,10024),  58: (4763,434),   59: (4612,1029),   60: (4784,561),
           61: (3716,289),    62: (4703,638),   63: (4851,4353),
           64: (4955,9899),   65: (5018,4629),  66: (4642,669),    67: (4840,4378),
           68: (4961,4528),   69: (4263,9718),  70: (5011,5485),   71: (4922,6222),
           72: (4317,672),    73: (3636,1275),  74: (4884,6083),   75: (5041,5264),
           76: (4912,10167),  77: (4504,1085),  78: (4617,194),    79: (4633,5012),
           80: (4566,4938),   81: (4702,9356),  82: (4758,5057),   83: (4860,866),
           84: (3962,2),      85: (4882,204),   86: (4467,9808),   87: (4730,4365),
           88: (4910,162),    89: (4684,367),   90: (4908,201),    91: (4759,18),
           92: (4880,251),    93: (4095,10167), 94: (4971,21),     95: (4873,685),
           96: (4561,92),     97: (4588,1057),  98: (4773,3),      99: (4997,5756),
          100: (4583,14),    101: (4900,9979), 102: (4574,9569),  103: (4629,515),
          104: (4676,753),   105: (4181,1181), 106: (5057,9442),  107: (4944,669),
          108: (4401,4834),  109: (4586,541),  110: (4699,9933),  111: (3676,6683),
          112: (4387,4828),  113: (4866,9710), 114: (4926,10170), 115: (4657,9629),
          116: (4477,260),   117: (4359,86),   118: (4673,5544),  119: (4258,923),
          120: (4447,257),   121: (4570,507),  122: (4486,4572),  123: (4362,4491),
          124: (4481,341),   125: (4322,130),  126: (4668,79),    127: (3967,1142),
          128: (4374,448),   129: (4553,875),  130: (4641,555),   131: (4215,1272),
          132: (3853,5198),  133: (4787,9529), 134: (4266,4459),  135: (4199,10019),
          136: (4545,9353),  137: (4208,9780), 138: (4485,375),   139: (3714,503),
          140: (4407,4507),  141: (4182,875),  142: (4203,1246),  143: (3788,1),
          144: (4471,4534),  145: (4691,8),    146: (4281,9549),  147: (4410,6240),
          148: (3953,22),    149: (3465,5652), 150: (4801,10069), 151: (4278,4796),
          152: (4546,4980),  153: (3779,27),   154: (4115,90),    155: (4193,9788),
          156: (3372,715),   157: (3786,9720), 158: (3491,301),   159: (3812,5450),
          160: (3594,5215),  161: (4028,13),   162: (3652,1147),  163: (4224,4855),
          164: (4334,1190),  165: (3245,1267), 166: (3921,1302),  167: (3840,1),
          168: (3514,5007),  169: (2922,549),  170: (4227,368),   171: (3376,6300),
          172: (3560,5658),  173: (4989,4302), 174: (4756,851),   175: (4624,4353),
          176: (4446,9618),  177: (4174,9652), 178: (4551,1232),  179: (3972,109),
          180: (4399,10174), 181: (4562,6178), 182: (3133,1851),  183: (4157,1299),
          184: (5053,325),   185: (4536,10206),186: (5067,9968),  187: (3905,10191),
          188: (3721,5438),  189: (3787,10080),190: (4674,219),   191: (3436,758),
          192: (2673,2140),  193: (4834,9753), 194: (4456,4799),  195: (4056,10126),
          196: (3804,241),   197: (3672,1245), 198: (4205,1274),  199: (3348,1456),
          200: (4152,9967),  201: (3883,235),  202: (3473,512),   203: (3669,1078),
          204: (3455,1078),  205: (2318,953),  206: (2945,5647),  207: (2947,669),
          208: (3220,1311),  209: (4052,5827), 210: (2953,15),
          }

        N = 10243
        L = np.array([legendre_symbol(i, N) for i in range(N)])
        L[L == -1] = 0
        L[0] = 0

        w, p = l1cd_params[prn]
        W = np.array([L[k] ^ L[(k+w) % N] for k in range(N)])
        expansion = np.array([0, 1, 1, 0, 1, 0, 0])
        c = np.concatenate((W[0:p-1], expansion, W[p-1:N]))
        l1cd_codes[prn] = c

    return l1cd_codes[prn]


l1cp_codes = {}
l1cp_secondary_codes = {}


def l1c_pilot(prn, secondary=False):
    """Generate unsampled GPS L1C_pilot signal.

    Input:
        prn - Index of satellite (1-210)
        secondary - Flag if primary code (secondary=False) or secondary code
                    (secondary=True) is returned, default=False

    Output:
        y - Primary or secondary B1C_pilot, length of 10230 chips at
            1.023 MHz chip rate and length of 1800 at xxx chip rate,
            respectively.

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/gps/l1cp.py
    written by Peter Monta.
    """
    from sympy.ntheory import legendre_symbol

    if not secondary:

        if prn not in l1cp_codes:

            # chip_rate = 1023000
            # code_length = 10230

            l1cp_params = {
                1: (5111,412),    2: (5109,161),     3: (5108,1),      4: (5106,303),
                5: (5103,207),    6: (5101,4971),    7: (5100,4496),   8: (5098,5),
                9: (5095,4557),  10: (5094,485),    11: (5093,253),   12: (5091,4676),
               13: (5090,1),     14: (5081,66),     15: (5080,4485),  16: (5069,282),
               17: (5068,193),   18: (5054,5211),   19: (5044,729),   20: (5027,4848),
               21: (5026,982),   22: (5014,5955),   23: (5004,9805),  24: (4980,670),
               25: (4915,464),   26: (4909,29),     27: (4893,429),   28: (4885,394),
               29: (4832,616),   30: (4824,9457),   31: (4591,4429),  32: (3706,4771),
               33: (5092,365),   34: (4986,9705),   35: (4965,9489),  36: (4920,4193),
               37: (4917,9947),  38: (4858,824),    39: (4847,864),   40: (4790,347),
               41: (4770,677),   42: (4318,6544),   43: (4126,6312),  44: (3961,9804),
               45: (3790,278),   46: (4911,9461),   47: (4881,444),   48: (4827,4839),
               49: (4795,4144),  50: (4789,9875),   51: (4725,197),   52: (4675,1156),
               53: (4539,4674),  54: (4535,10035),  55: (4458,4504),  56: (4197,5),
               57: (4096,9937),  58: (3484,430),    59: (3481,5),     60: (3393,355),
               61: (3175,909),   62: (2360,1622),   63: (1852,6284),
               64: (5065,9429),  65: (5063,77),     66: (5055,932),   67: (5012,5973),
               68: (4981,377),   69: (4952,10000),  70: (4934,951),   71: (4932,6212),
               72: (4786,686),   73: (4762,9352),   74: (4640,5999),  75: (4601,9912),
               76: (4563,9620),  77: (4388,635),    78: (3820,4951),  79: (3687,5453),
               80: (5052,4658),  81: (5051,4800),   82: (5047,59),    83: (5039,318),
               84: (5015,571),   85: (5005,565),    86: (4984,9947),  87: (4975,4654),
               88: (4974,148),   89: (4972,3929),   90: (4962,293),   91: (4913,178),
               92: (4907,10142), 93: (4903,9683),   94: (4833,137),   95: (4778,565),
               96: (4721,35),    97: (4661,5949),   98: (4660,2),     99: (4655,5982),
              100: (4623,825),  101: (4590,9614),  102: (4548,9790), 103: (4461,5613),
              104: (4442,764),  105: (4347,660),   106: (4259,4870), 107: (4256,4950),
              108: (4166,4881), 109: (4155,1151),  110: (4109,9977), 111: (4100,5122),
              112: (4023,10074),113: (3998,4832),  114: (3979,77),   115: (3903,4698),
              116: (3568,1002), 117: (5088,5549),  118: (5050,9606), 119: (5020,9228),
              120: (4990,604),  121: (4982,4678),  122: (4966,4854), 123: (4949,4122),
              124: (4947,9471), 125: (4937,5026),  126: (4935,272),  127: (4906,1027),
              128: (4901,317),  129: (4872,691),   130: (4865,509),  131: (4863,9708),
              132: (4818,5033), 133: (4785,9938),  134: (4781,4314), 135: (4776,10140),
              136: (4775,4790), 137: (4754,9823),  138: (4696,6093), 139: (4690,469),
              140: (4658,1215), 141: (4607,799),   142: (4599,756),  143: (4596,9994),
              144: (4530,4843), 145: (4524,5271),  146: (4451,9661), 147: (4441,6255),
              148: (4396,5203), 149: (4340,203),   150: (4335,10070),151: (4296,30),
              152: (4267,103),  153: (4168,5692),  154: (4149,32),   155: (4097,9826),
              156: (4061,76),   157: (3989,59),    158: (3966,6831), 159: (3789,958),
              160: (3775,1471), 161: (3622,10070), 162: (3523,553),  163: (3515,5487),
              164: (3492,55),   165: (3345,208),   166: (3235,645),  167: (3169,5268),
              168: (3157,1873), 169: (3082,427),   170: (3072,367),  171: (3032,1404),
              172: (3030,5652), 173: (4582,5),     174: (4595,368),  175: (4068,451),
              176: (4871,9595), 177: (4514,1030),  178: (4439,1324), 179: (4122,692),
              180: (4948,9819), 181: (4774,4520),  182: (3923,9911), 183: (3411,278),
              184: (4745,642),  185: (4195,6330),  186: (4897,5508), 187: (3047,1872),
              188: (4185,5445), 189: (4354,10131), 190: (5077,422),  191: (4042,4918),
              192: (2111,787),  193: (4311,9864),  194: (5024,9753), 195: (4352,9859),
              196: (4678,328),  197: (5034,1),     198: (5085,4733), 199: (3646,164),
              200: (4868,135),  201: (3668,174),   202: (4211,132),  203: (2883,538),
              204: (2850,176),  205: (2815,198),   206: (2542,595),  207: (2492,574),
              208: (2376,321),  209: (2036,596),   210: (1920,491),
              }

            N = 10223
            L = np.array([legendre_symbol(i, N) for i in range(N)])
            L[L == -1] = 0
            L[0] = 0

            w, p = l1cp_params[prn]
            W = np.array([L[k] ^ L[(k+w) % N] for k in range(N)])
            expansion = np.array([0, 1, 1, 0, 1, 0, 0])
            c = np.concatenate((W[0:p-1], expansion, W[p-1:N]))
            l1cp_codes[prn] = c

        return l1cp_codes[prn]

    else:

        if prn not in l1cp_secondary_codes:

            l1cp_secondary_params = {
                1: (0o5111,0o3266),   2: (0o5421,0o2040),   3: (0o5501,0o1527),   4: (0o5403,0o3307),
                5: (0o6417,0o3756),   6: (0o6141,0o3026),   7: (0o6351,0o0562),   8: (0o6501,0o0420),
                9: (0o6205,0o3415),  10: (0o6235,0o0337),  11: (0o7751,0o0265),  12: (0o6623,0o1230),
                13: (0o6733,0o2204),  14: (0o7627,0o1440),  15: (0o5667,0o2412),  16: (0o5051,0o3516),
                17: (0o7665,0o2761),  18: (0o6325,0o3750),  19: (0o4365,0o2701),  20: (0o4745,0o1206),
                21: (0o7633,0o1544),  22: (0o6747,0o1774),  23: (0o4475,0o0546),  24: (0o4225,0o2213),
                25: (0o7063,0o3707),  26: (0o4423,0o2051),  27: (0o6651,0o3650),  28: (0o4161,0o1777),
                29: (0o7237,0o3203),  30: (0o4473,0o1762),  31: (0o5477,0o2100),  32: (0o6163,0o0571),
                33: (0o7223,0o3710),  34: (0o6323,0o3535),  35: (0o7125,0o3110),  36: (0o7035,0o1426),
                37: (0o4341,0o0255),  38: (0o4353,0o0321),  39: (0o4107,0o3124),  40: (0o5735,0o0572),
                41: (0o6741,0o1736),  42: (0o7071,0o3306),  43: (0o4563,0o1307),  44: (0o5755,0o3763),
                45: (0o6127,0o1604),  46: (0o4671,0o1021),  47: (0o4511,0o2624),  48: (0o4533,0o0406),
                49: (0o5357,0o0114),  50: (0o5607,0o0077),  51: (0o6673,0o3477),  52: (0o6153,0o1000),
                53: (0o7565,0o3460),  54: (0o7107,0o2607),  55: (0o6211,0o2057),  56: (0o4321,0o3467),
                57: (0o7201,0o0706),  58: (0o4451,0o2032),  59: (0o5411,0o1464),  60: (0o5141,0o0520),
                61: (0o7041,0o1766),  62: (0o6637,0o3270),  63: (0o4577,0o0341),
                64: (0o5111,0o1740,0o3035),  65: (0o5111,0o3664,0o1557),  66: (0o5111,0o1427,0o0237),  67: (0o5111,0o2627,0o2527),
                68: (0o5111,0o0701,0o3307),  69: (0o5111,0o3460,0o1402),  70: (0o5111,0o1373,0o1225),  71: (0o5111,0o2540,0o0607),
                72: (0o5111,0o2004,0o0351),  73: (0o5111,0o2274,0o3724),  74: (0o5111,0o1340,0o1675),  75: (0o5111,0o0602,0o2625),
                76: (0o5111,0o2502,0o1030),  77: (0o5111,0o0327,0o1443),  78: (0o5111,0o2600,0o3277),  79: (0o5111,0o0464,0o1132),
                80: (0o5111,0o3674,0o0572),  81: (0o5111,0o3040,0o1241),  82: (0o5111,0o1153,0o0535),  83: (0o5111,0o0747,0o1366),
                84: (0o5111,0o1770,0o0041),  85: (0o5111,0o3772,0o0561),  86: (0o5111,0o1731,0o0122),  87: (0o5111,0o1672,0o1205),
                88: (0o5111,0o1333,0o3753),  89: (0o5111,0o2705,0o2543),  90: (0o5111,0o2713,0o3031),  91: (0o5111,0o3562,0o2260),
                92: (0o5111,0o3245,0o3773),  93: (0o5111,0o3770,0o3156),  94: (0o5111,0o3202,0o2215),  95: (0o5111,0o3521,0o0146),
                96: (0o5111,0o3250,0o2413),  97: (0o5111,0o2117,0o2564),  98: (0o5111,0o0530,0o3310),  99: (0o5111,0o3021,0o2267),
               100: (0o5421,0o2511,0o3120), 101: (0o5421,0o1562,0o0064), 102: (0o5421,0o1067,0o1042), 103: (0o5421,0o0424,0o0476),
               104: (0o5421,0o3402,0o1020), 105: (0o5421,0o1326,0o0431), 106: (0o5421,0o2142,0o0216), 107: (0o5421,0o0733,0o2736),
               108: (0o5421,0o0504,0o2527), 109: (0o5421,0o1611,0o2431), 110: (0o5421,0o2724,0o1013), 111: (0o5421,0o0753,0o0524),
               112: (0o5421,0o3724,0o0726), 113: (0o5421,0o2652,0o1042), 114: (0o5421,0o1743,0o3362), 115: (0o5421,0o0013,0o1364),
               116: (0o5421,0o3464,0o3354), 117: (0o5421,0o2300,0o0623), 118: (0o5421,0o1334,0o0145), 119: (0o5421,0o2175,0o0214),
               120: (0o5421,0o2564,0o0223), 121: (0o5421,0o3075,0o0151), 122: (0o5421,0o3455,0o2405), 123: (0o5421,0o3627,0o2522),
               124: (0o5421,0o0617,0o3235), 125: (0o5421,0o1324,0o0452), 126: (0o5421,0o3506,0o2617), 127: (0o5421,0o2231,0o1300),
               128: (0o5421,0o1110,0o1430), 129: (0o5421,0o1271,0o0773), 130: (0o5421,0o3740,0o0772), 131: (0o5421,0o3652,0o3561),
               132: (0o5421,0o1644,0o0607), 133: (0o5421,0o3635,0o0420), 134: (0o5421,0o3436,0o0527), 135: (0o5421,0o3076,0o3770),
               136: (0o5421,0o0434,0o2536), 137: (0o5421,0o3340,0o2233), 138: (0o5421,0o0054,0o3366), 139: (0o5403,0o2446,0o3766),
               140: (0o5403,0o0025,0o3554), 141: (0o5403,0o0150,0o2060), 142: (0o5403,0o2746,0o2070), 143: (0o5403,0o2723,0o0713),
               144: (0o5403,0o2601,0o3366), 145: (0o5403,0o3440,0o3247), 146: (0o5403,0o1312,0o2776), 147: (0o5403,0o0544,0o1244),
               148: (0o5403,0o2062,0o2102), 149: (0o5403,0o0176,0o1712), 150: (0o5403,0o3616,0o1245), 151: (0o5403,0o1740,0o3344),
               152: (0o5403,0o3777,0o1277), 153: (0o5403,0o0432,0o0165), 154: (0o5403,0o2466,0o2131), 155: (0o5403,0o1667,0o3623),
               156: (0o5403,0o3601,0o0141), 157: (0o5403,0o2706,0o0421), 158: (0o5403,0o2022,0o3032), 159: (0o5403,0o1363,0o2065),
               160: (0o5403,0o2331,0o3024), 161: (0o5403,0o3556,0o2663), 162: (0o5403,0o2205,0o2274), 163: (0o5403,0o3734,0o2114),
               164: (0o5403,0o2115,0o1664), 165: (0o5403,0o0010,0o0413), 166: (0o5403,0o2140,0o1512), 167: (0o5403,0o3136,0o0135),
               168: (0o5403,0o0272,0o2737), 169: (0o5403,0o3264,0o1015), 170: (0o5403,0o2017,0o1075), 171: (0o5403,0o2505,0o1255),
               172: (0o5403,0o3532,0o3473), 173: (0o5403,0o0647,0o2716), 174: (0o5403,0o1542,0o0101), 175: (0o5403,0o2154,0o1105),
               176: (0o5403,0o3734,0o1407), 177: (0o5403,0o2621,0o3407), 178: (0o5403,0o2711,0o1046), 179: (0o5403,0o0217,0o3237),
               180: (0o5403,0o3503,0o0154), 181: (0o5403,0o3457,0o3010), 182: (0o5403,0o3750,0o2245), 183: (0o5403,0o2525,0o2051),
               184: (0o5403,0o0113,0o2144), 185: (0o5403,0o0265,0o1743), 186: (0o5403,0o1711,0o2511), 187: (0o5403,0o0552,0o3410),
               188: (0o5403,0o0675,0o1414), 189: (0o5403,0o1706,0o1275), 190: (0o5403,0o3513,0o2257), 191: (0o5403,0o1135,0o2331),
               192: (0o5403,0o0566,0o0276), 193: (0o5403,0o0500,0o3261), 194: (0o5403,0o0254,0o1760), 195: (0o5403,0o3445,0o0430),
               196: (0o5403,0o2542,0o3477), 197: (0o5403,0o1257,0o1676), 198: (0o6501,0o0211,0o1636), 199: (0o6501,0o0534,0o2411),
               200: (0o6501,0o1420,0o1473), 201: (0o6501,0o3401,0o2266), 202: (0o6501,0o0714,0o2104), 203: (0o6501,0o0613,0o2070),
               204: (0o6501,0o2475,0o1766), 205: (0o6501,0o2572,0o0711), 206: (0o6501,0o3265,0o2533), 207: (0o6501,0o1250,0o0353),
               208: (0o6501,0o1711,0o1744), 209: (0o6501,0o2704,0o0053), 210: (0o6501,0o0135,0o2222),
               }

            sec_code_length = 1800

            def int2list(x, n):
                y = []
                for i in range(n):
                    y.append((x >> i) & 1)
                return y

            def xorprod(a, b):
                t = 0
                for x, y in zip(a, b):
                    t = t ^ (x*y)
                return t

            def s_shift(x, p):
                return [xorprod(x, p)] + x[0:-1]

            p, init = l1cp_secondary_params[prn]
            p = int2list(p//2, 11)
            x = int2list(init, 11)
            c = np.zeros(sec_code_length)
            for i in range(sec_code_length):
                c[i] = x[10]
                x = s_shift(x, p)
            l1cp_secondary_codes[prn] = c

        return l1cp_secondary_codes[prn]


glonass_l1_code = {}


def generate_ca_code_glonass():
    """Generate GLONASS signal that is transmitted in L1 band.

    Inputs:
        prn - Index of satellite (1-210)
        fs - Sampling rate [Hz]
        pilot - Flag if data component (pilot=False) or primary pilot component
                (pilot=True) is returned, default=False.
    Output:
        glonass_l1_code - L1 C/A code, length of 511 chips at 0.511 MHz chip rate

    Adapted by Jonas Beuchert from
    https://github.com/pmonta/GNSS-DSP-tools/blob/master/gnsstools/glonass/ca.py
    written by Peter Monta.
    """
    global glonass_l1_code
    # Check if C/A code must be generated
    if len(glonass_l1_code) == 0:

        # chip_rate = 511000
        code_length = 511

        x = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        glonass_l1_code = np.zeros(code_length)
        for i in range(code_length):
            glonass_l1_code[i] = x[6]
            x = [x[8] ^ x[4]] + x[0:8]

    return glonass_l1_code


def rinexe(ephemerisfile, system=None):
    """Read a RINEX Navigation Message file and reformat the data.

    Reformat the data into a 2D NumPy array with 21 rows and a column for each
    satellite.
    Units are either seconds, meters, or radians

    Typical calls: rinexe("brdc1310.20n")
                   rinexe("BRDC00IGS_R_20203410000_01D_MN.rnx", system='S')

    Inputs:
        ephemerisfile - Path to RINEX Navigation Message file
        system - Character of system to consider, only in RINEX version 3.
               'G' - GPS
               'S' - SBAS
               'R' - GLONASS
               'E' - Galileo
               'C' - BeiDou
               'J' - QZSS
               'I' - NavIC
               None - Only one GNSS is assumed to be present (default)

    Author: Jonas Beuchert
    """
    with open(ephemerisfile, "r") as fide:
        try:
            line = fide.readline()
            version = int(line[5])
        except ValueError:
            raise ValueError('Could not find RINEX version in first line of file.')
        head_lines = 1
        answer = -1
        leap_seconds = 18  # Use default leap seconds if they are not in header
        while answer < 0:  # Skip header
            head_lines = head_lines + 1
            line = fide.readline()
            if line.find("LEAP SECONDS") >= 0 and (system == 'S' or system == 'R'):
                # Read leap seconds for SBAS or GLONASS
                leap_seconds = int(line[4:6])
                leap_seconds = np.timedelta64(leap_seconds, 's')
            answer = line.find("END OF HEADER")
        line = "init"
        if version == 2:
            noeph = -1
            while not line == "":
                noeph = noeph + 1
                line = fide.readline()
            # if not sbas:
            noeph = int(noeph / 8)
            # else:
            # noeph = int(noeph / 4)
        elif version == 3:
            noeph = 0
            while not line == "":
                line = fide.readline()
                if not line == "" and not line[0] == " " and (
                        system is None or line[0] == system):
                    noeph = noeph + 1
        else:
            raise ValueError("Found unsupported RINEX version, use 2 or 3.xx.")
        fide.seek(0)
        for i in range(head_lines):
            line = fide.readline()

        # Set aside memory for the input
        svprn = np.zeros(noeph)
        # weekno = np.zeros(noeph)
        # t0c = np.zeros(noeph)
        # tgd = np.zeros(noeph)
        # aodc = np.zeros(noeph)
        toe = np.zeros(noeph)
        af2 = np.zeros(noeph)
        af1 = np.zeros(noeph)
        af0 = np.zeros(noeph)
        # aode = np.zeros(noeph)
        deltan = np.zeros(noeph)
        M0 = np.zeros(noeph)
        ecc = np.zeros(noeph)
        roota = np.zeros(noeph)
        toe = np.zeros(noeph)
        cic = np.zeros(noeph)
        crc = np.zeros(noeph)
        cis = np.zeros(noeph)
        crs = np.zeros(noeph)
        cuc = np.zeros(noeph)
        cus = np.zeros(noeph)
        Omega0 = np.zeros(noeph)
        omega = np.zeros(noeph)
        i0 = np.zeros(noeph)
        Omegadot = np.zeros(noeph)
        idot = np.zeros(noeph)
        # tom = np.zeros(noeph)
        # accuracy = np.zeros(noeph)
        # health = np.zeros(noeph)
        # fit = np.zeros(noeph)

        if not system == 'S' and not system == 'R':
            if version == 2:
                for i in range(noeph):
                    line = fide.readline()
                    svprn[i] = int(line[0:2].replace("D", "E"))
                    # year = line[2:6]
                    # month = line[6:9]
                    # day = line[9:12]
                    # hour = line[12:15]
                    # minute = line[15:18]
                    # second = line[18:22]
                    af0[i] = float(line[22:41].replace("D", "E"))
                    af1[i] = float(line[41:60].replace("D", "E"))
                    af2[i] = float(line[60:79].replace("D", "E"))
                    line = fide.readline()
                    # IODE = line[3:22]
                    crs[i] = float(line[22:41].replace("D", "E"))
                    deltan[i] = float(line[41:60].replace("D", "E"))
                    M0[i] = float(line[60:79].replace("D", "E"))
                    line = fide.readline()
                    cuc[i] = float(line[3:22].replace("D", "E"))
                    ecc[i] = float(line[22:41].replace("D", "E"))
                    cus[i] = float(line[41:60].replace("D", "E"))
                    roota[i] = float(line[60:79].replace("D", "E"))
                    line = fide.readline()
                    toe[i] = float(line[3:22].replace("D", "E"))
                    cic[i] = float(line[22:41].replace("D", "E"))
                    Omega0[i] = float(line[41:60].replace("D", "E"))
                    cis[i] = float(line[60:79].replace("D", "E"))
                    line = fide.readline()
                    i0[i] = float(line[3:22].replace("D", "E"))
                    crc[i] = float(line[22:41].replace("D", "E"))
                    omega[i] = float(line[41:60].replace("D", "E"))
                    Omegadot[i] = float(line[60:79].replace("D", "E"))
                    line = fide.readline()
                    idot[i] = float(line[3:22].replace("D", "E"))
                    # codes = float(line[22:41].replace("D", "E"))
                    # weekno = float(line[41:60].replace("D", "E"))
                    # L2flag = float(line[60:79].replace("D", "E"))
                    line = fide.readline()
                    # svaccur = float(line[3:22].replace("D", "E"))
                    # svhealth = float(line[22:41].replace("D", "E"))
                    # tgd[i] = float(line[41:60].replace("D", "E"))
                    # iodc = line[60:79]
                    line = fide.readline()
                    # tom[i] = float(line[3:22].replace("D", "E"))
                    # spare = line[22:41]
                    # spare = line[41:60]
                    # spare = line[60:79]
            elif version == 3:
                for i in range(noeph):
                    if system is not None:
                        # Multiple systems might be present
                        # Skip lines until desired one is found
                        line = "init"
                        while not line[0] == system:
                            line = fide.readline()
                    else:
                        line = fide.readline()
                    svprn[i] = int(line[1:3])
                    af0[i] = float(line[23:42])
                    af1[i] = float(line[42:61])
                    af2[i] = float(line[61:80])
                    line = fide.readline()
                    crs[i] = float(line[23:42])
                    deltan[i] = float(line[42:61])
                    M0[i] = float(line[61:80])
                    line = fide.readline()
                    cuc[i] = float(line[4:23])
                    ecc[i] = float(line[23:42])
                    cus[i] = float(line[42:61])
                    roota[i] = float(line[61:80])
                    line = fide.readline()
                    toe[i] = float(line[4:23])
                    cic[i] = float(line[23:42])
                    Omega0[i] = float(line[42:61])
                    cis[i] = float(line[61:80])
                    line = fide.readline()
                    i0[i] = float(line[4:23])
                    crc[i] = float(line[23:42])
                    omega[i] = float(line[42:61])
                    Omegadot[i] = float(line[61:80])
                    line = fide.readline()
                    idot[i] = float(line[4:23])
                    line = fide.readline()
                    # tgd[i] = float(line[42:61])
                    line = fide.readline()
                    # tom[i] = float(line[4:23])
            else:
                raise ValueError(
                    "Found unsupported RINEX version, use 2 or 3.xx.")
        else:  # SBAS / GLONASS navigation message format
            if version == 2:
                raise ValueError(
                    "RINEX version 2 not supported for SBAS and GLONASS.")
            elif version == 3:
                # Set aside memory for the input
                pos_x = np.zeros(noeph)
                vel_x = np.zeros(noeph)
                acc_x = np.zeros(noeph)
                pos_y = np.zeros(noeph)
                vel_y = np.zeros(noeph)
                acc_y = np.zeros(noeph)
                pos_z = np.zeros(noeph)
                vel_z = np.zeros(noeph)
                acc_z = np.zeros(noeph)

                for i in range(noeph):
                    # Multiple systems might be present
                    # Skip lines until desired one is found
                    line = "init"
                    while not line[0] == system:
                        line = fide.readline()
                    if line[0] == 'S':
                        # Satellite number
                        svprn[i] = 100 + int(line[1:3])
                    # Time of Ephemeris (sec of BDT week)
                    year = line[4:8]
                    month = line[9:11]
                    day = line[12:14]
                    hour = line[15:17]
                    minute = line[18:20]
                    second = line[21:23]
                    time_utc = np.datetime64(year + '-'
                                            + month + '-'
                                            + day + 'T'
                                            + hour + ':'
                                            + minute + ':'
                                            + second)
                    time_bds = (
                        time_utc - np.datetime64('1980-01-06') + leap_seconds
                        ) / np.timedelta64(1, 's')
                    # time_bds = utc_2_gps_time(time_utc)
                    toe[i] = np.mod(time_bds, 7 * 24 * 60 * 60)
                    # SV clock bias [s] (aGf0)
                    af0[i] = float(line[23:42])
                    # Transmission time in GPS seconds of week
                    # tom[i] = float(line[42:61])
                    line = fide.readline()
                    pos_x[i] = float(line[4:23])
                    vel_x[i] = float(line[23:42])
                    acc_x[i] = float(line[42:61])
                    line = fide.readline()
                    pos_y[i] = float(line[4:23])
                    vel_y[i] = float(line[23:42])
                    acc_y[i] = float(line[42:61])
                    line = fide.readline()
                    pos_z[i] = float(line[4:23])
                    vel_z[i] = float(line[23:42])
                    acc_z[i] = float(line[42:61])
            else:
                raise ValueError(
                    "Found unsupported RINEX version, use 2 or 3.xx.")

    #  Description of variable eph
    if not system == 'S' and not system == 'R':
        return np.array(
            [
                svprn,
                af2,
                M0,
                roota,
                deltan,
                ecc,
                omega,
                cuc,
                cus,
                crc,
                crs,
                i0,
                idot,
                cic,
                cis,
                Omega0,
                Omegadot,
                toe,
                af0,
                af1,
                toe
            ]
        )
    else:  # SBAS / GLONASS
        return np.array(
            [
                svprn,
                af2,
                np.empty(noeph) * np.nan,
                pos_x,
                vel_x,
                acc_x,
                pos_y,
                vel_y,
                acc_y,
                pos_z,
                vel_z,
                acc_z,
                np.empty(noeph) * np.nan,
                np.empty(noeph) * np.nan,
                np.empty(noeph) * np.nan,
                np.empty(noeph) * np.nan,
                np.empty(noeph) * np.nan,
                toe,
                af0,
                af1,
                toe
            ]
        )


def gps_ionosphere_parameters_from_rinex(rinexfile):
    """Read ionospheric correction parameters and leap sec. for GPS from RINEX.

    Input:
        rinexfile - Path to RINEX Navigation Message file

    Outputs:
        alpha - Coefficients of a cubic equation representing the amplitude of
                the vertical delay (4 coefficients, numpy array)
        beta - Coefficients of a cubic equation representing the period of the
               model (4 coefficients, numpy array)
        leap_seconds - GPS leap seconds w.r.t. to UTC

    Author: Jonas Beuchert
    """
    # Open file in read mode
    file_id = open(rinexfile, "r")
    # Initialize results
    alpha = np.full(4, np.nan)
    beta = np.full(4, np.nan)
    leap_seconds = np.nan
    # Search in header for desired parameters
    end_of_header = False
    while not end_of_header:
        # Read single line
        line = file_id.readline()
        # Search line for key words
        if line.find("ION ALPHA") >= 0:
            # Alpha parameters, RINEX 2
            for idx in np.arange(4):
                start_char = 3 + idx * 12
                end_char = 3 + (idx+1) * 12 - 1
                alpha[idx] = float(line[start_char:end_char].replace("D", "E"))
        elif line.find("ION BETA") >= 0:
            # Beta parameters, RINEX 2
            for idx in np.arange(4):
                start_char = 3 + idx * 12
                end_char = 3 + (idx+1) * 12 - 1
                beta[idx] = float(line[start_char:end_char].replace("D", "E"))
        elif line.find("GPSA") >= 0:
            # Alpha parameters, RINEX 3
            for idx in np.arange(4):
                start_char = 6 + idx * 12
                end_char = 6 + (idx+1) * 12 - 1
                alpha[idx] = float(line[start_char:end_char])
        elif line.find("GPSB") >= 0:
            # Beta parameters, RINEX 3
            for idx in np.arange(4):
                start_char = 6 + idx * 12
                end_char = 6 + (idx+1) * 12 - 1
                beta[idx] = float(line[start_char:end_char])
        elif line.find("LEAP SECONDS") >= 0:
            # Leap seconds
            start_char = 4
            end_char = 6
            leap_seconds = float(line[start_char:end_char])
        # Check if end of header or end of file has been reached
        end_of_header = line.find("END OF HEADER") >= 0 or line == ""
    # Close RINEX file
    file_id.close()
    # Return tuple
    return alpha, beta, leap_seconds


def read_sp3(file_name):
    """Read final precise orbits from SP3 file.

    Requires to install package GNSSpy first, e.g., by executing
    'python setup.py install' in the directory '../3rd_party/gnsspy'.

    Input:
        file_name - Path to SP3 file
    Output:
        sp3 - Supporting points of orbits as pandas.DataFrame

    Author: Jonas Beuchert
    Based on https://github.com/GNSSpy-Project/gnsspy/blob/fce079af37d585dc757c56539a98cc0dfe66f9de/gnsspy/position/interpolation.py
    """
    import pandas as pd
    from gnsspy.io import readFile

    sp3 = readFile.read_sp3File(file_name)
    sp3 = sp3.dropna(subset=["deltaT"])
    sp3 = sp3.reorder_levels(["Epoch", "SV"])
    return sp3


def interpolate_code_phase(corr, mode="quadratic"):
    """Interpolate code phase value based on correlogram.

    Necessary to obtain fractional code phases with a resolution that is not
    limited by the sampling frequency.

    Input:
        corr - 1D correlogram, correlation over code phase for one satellite
        mode - [Optional] type of interpolation
               'none' - No interpolation
               'linear' - Linear interpolation
               'quadratic' - [Default] quadratic interpol. based on 3 points
               'quadratic5' - Quadratic interpolation based on 5 points
    Output:
        codePhase - Interpolated code phase (in samples)

    Author: Jonas Beuchert
    """
    maxInd = np.argmax(corr)
    if mode == "none":
        return maxInd + 1.0
    maxVal = corr[maxInd]
    leftInd = maxInd - 1
    rightInd = np.mod(maxInd + 1, corr.shape[0])
    leftVal = corr[leftInd]
    rightVal = corr[rightInd]
    if mode == "linear":
        return ((maxInd - 1) * leftVal + maxInd * maxVal + (maxInd + 1) *
                rightVal) / (leftVal + maxVal + rightVal) + 1.0
    if mode == "quadratic":
        # Algorithm from Chapter 8.12 of
        # Tsui, James Bao-Yen. Fundamentals of global positioning system
        # receivers: a software approach. Vol. 173. John Wiley & Sons, 2005.
        # http://twanclik.free.fr/electricity/electronic/pdfdone7/Fundamentals%20of%20Global%20Positioning%20System%20Receivers.pdf
        x1 = -1.0
        x2 = 0.0
        x3 = 1.0
        y1 = leftVal
        y2 = maxVal
        y3 = rightVal
        Y = np.array([y1, y2, y3])
        X = np.array([[x1**2, x1, 1.0],
                      [x2**2, x2, 1.0],
                      [x3**2, x3, 1.0]])
        A = np.linalg.lstsq(X, Y, rcond=None)[0]
        a = A[0]
        b = A[1]
        # c = A[2]
        x = -b / 2.0 / a
        return maxInd + x + 1.0
    if mode == "quadratic5":
        leftLeftInd = maxInd - 2
        rightRightInd = np.mod(maxInd + 2, corr.shape[0])
        leftLeftVal = corr[leftLeftInd]
        rightRightVal = corr[rightRightInd]
        # Algorithm from Chapter 8.12 of
        # Tsui, James Bao-Yen. Fundamentals of global positioning system
        # receivers: a software approach. Vol. 173. John Wiley & Sons, 2005.
        # http://twanclik.free.fr/electricity/electronic/pdfdone7/Fundamentals%20of%20Global%20Positioning%20System%20Receivers.pdf
        x1 = -2.0
        x2 = -1.0
        x3 = 0.0
        x4 = 1.0
        x5 = 2.0
        y1 = leftLeftVal
        y2 = leftVal
        y3 = maxVal
        y4 = rightVal
        y5 = rightRightVal
        Y = np.array([y1, y2, y3, y4, y5])
        X = np.array([[x1**2, x1, 1.0],
                      [x2**2, x2, 1.0],
                      [x3**2, x3, 1.0],
                      [x4**2, x4, 1.0],
                      [x5**2, x5, 1.0]])
        A = np.linalg.lstsq(X, Y, rcond=None)[0]
        a = A[0]
        b = A[1]
        # c = A[2]
        x = -b / 2.0 / a
        return maxInd + x + 1.0


def gps_time_2_utc(gps_time_sec, leapSeconds=None):
    """Convert time from seconds since GPS start into UTC.

    18 leap seconds, i.e., -18 s, for all dates after 2016-12-31.

    Inputs:
        gps_time_sec - Time in seconds since GPS reference date & time [s]
        leapSeconds - GPS leap seconds w.r.t. UTC; if None, then leap seconds
                      are calculated from date; default=None

    Output:
        utc - UTC (datetime format)

    Author: Jonas Beuchert
    """
    if leapSeconds is None or not np.isfinite(leapSeconds):
        stepDates = np.array(
            [46828800.0, 78364801.0, 109900802.0, 173059203.0,
             252028804.0, 315187205.0, 346723206.0, 393984007.0,
             425520008.0, 457056009.0, 504489610.0, 551750411.0,
             599184012.0, 820108813.0, 914803214.0, 1025136015.0,
             1119744016.0, 1167177617]
            )
        leapSeconds = 0
        while leapSeconds < 18 and gps_time_sec > stepDates[leapSeconds]:
            leapSeconds = leapSeconds + 1
    referenceDate = np.datetime64('1980-01-06')  # GPS reference date
    return (np.timedelta64(int((gps_time_sec - leapSeconds) * 1e9), 'ns')
            + referenceDate)


def utc_2_gps_time(utc, leapSeconds=None):
    """Convert time from UTC to seconds since GPS start.

    18 leap seconds, i.e., +18 s, for all dates after 2016-12-31.

    Inputs:
        utc - UTC (numpy.datetime64)
        leapSeconds - GPS leap seconds w.r.t. UTC; if None, then leap seconds
                      are calculated from date; default=None

    Output:
        gps_time - Time in seconds since GPS reference date & time [s]

    Author: Jonas Beuchert
    """
    if leapSeconds is None or not np.isfinite(leapSeconds):
        stepDates = np.array([
            np.datetime64('1981-07-01'),
            np.datetime64('1982-07-01'),
            np.datetime64('1983-07-01'),
            np.datetime64('1985-07-01'),
            np.datetime64('1988-01-01'),
            np.datetime64('1990-01-01'),
            np.datetime64('1991-01-01'),
            np.datetime64('1992-07-01'),
            np.datetime64('1993-07-01'),
            np.datetime64('1994-07-01'),
            np.datetime64('1996-01-01'),
            np.datetime64('1997-07-01'),
            np.datetime64('1999-01-01'),
            np.datetime64('2006-01-01'),
            np.datetime64('2009-01-01'),
            np.datetime64('2012-07-01'),
            np.datetime64('2015-07-01'),
            np.datetime64('2016-12-31')])
        leapSeconds = 0
        while leapSeconds < 18 and utc > stepDates[leapSeconds]:
            leapSeconds = leapSeconds + 1
    referenceDate = np.datetime64('1980-01-06')  # GPS reference date
    leapSeconds = np.timedelta64(leapSeconds, 's')
    return (utc - referenceDate + leapSeconds) / np.timedelta64(1, 's')


def gps_time_2_beidou_time(gps_time_sec):
    """Convert time from seconds since GPS start into BeiDou time (BDT).

    Input:
        gps_time_sec - Time in seconds since GPS reference date & time [s]

    Output:
        bdt - BeiDou time [s] (time in seconds since BeiDou time reference)

    Author: Jonas Beuchert
    """
    return gps_time_sec - 820108814.0


def predict_pseudoranges(sats, eph, coarse_time, rec_pos, common_bias,
                         trop=False):
    """Predict pseudoranges to satellites for given time and receiver position.

    Inputs:
      sats - Indices of satellites (PRNs)
      eph - Ephemeris as matrix
      coarse_time - Coarse GPS time [s]
      rec_pos - Receiver position in ECEF XYZ coordinates [m,m,m]
      common_bias - Common bias in all pseudoranges [m]
      trop - [Optional] flag indicating if troposheric correction is applied;
             default = True

    Output:
      predictedPR - Predicted pseudoranges [m]

    Author: Jonas Beuchert
    Algorithm from Chapter 4.4.2 of
    F. van Diggelen, A-GPS: Assisted GPS, GNSS, and SBAS, 2009.
    """
    # Speed of light [m/s]
    c = 299792458.0
    # Number of satellites
    nSats = sats.shape[0]

    # GPS time since 1980 to time of week (TOW) [s]
    coarseTimeTOW = np.mod(coarse_time, 7 * 24 * 60 * 60)

    # Identify matching columns in ephemeris matrix, closest column in time
    # for each satellite
    if nSats < eph.shape[1]:
        col = np.array([find_eph(eph, s_i, coarseTimeTOW) for s_i in sats])
        if col.size == 0:
            raise IndexError("Cannot find satellite in navigation data.")
        # Extract matching columns
        eph = eph[:, col]

    # Find satellite positions at coarse transmission time
    txGPS = coarseTimeTOW - get_sat_clk_corr(coarseTimeTOW, sats, eph)
    satPosCoarse = get_sat_pos(txGPS, eph)

    # Find closest one (alternatively, find highest)
    distancesCoarse = np.sqrt(np.sum((rec_pos - satPosCoarse)**2, axis=-1))
    satByDistance = np.argsort(distancesCoarse)

    # Assign integer ms-part of distances
    Ns = np.zeros(nSats)
    # Time equivalent to distance [ms]
    distancesCoarse = distancesCoarse / c / 1e-3
    # Index of 1st satellite (reference satellite)
    N0Inx = satByDistance[0]
    # Initial guess
    Ns[N0Inx] = np.floor(distancesCoarse[N0Inx])
    # Time error [ms]
    deltaT = eph[18] * 1e3
    # Update considering time error
    for i in range(1, nSats):
        k = satByDistance[i]
        Ns[k] = np.round(Ns[N0Inx] + (distancesCoarse[k] - deltaT[k])
                         - (distancesCoarse[N0Inx] - deltaT[N0Inx]))

    # Find integer ms-part difference to reference satellite
    Ks = Ns - Ns[N0Inx]

    # Correct for satellite clock error
    tCorr = np.empty(nSats)
    for i in range(nSats):
        k = np.array([sats[i]])
        tCorr[i] = get_sat_clk_corr(coarseTimeTOW - Ks[i] * 1e-3, k,
                                    eph[:, i, np.newaxis])
    txGPS = coarseTimeTOW - Ks * 1e-3 - tCorr

    # Get satellite position at corrected transmission time
    satPos = get_sat_pos(txGPS, eph)

    # Calculate rough propagation delay
    travelTime = np.linalg.norm(satPos - rec_pos, axis=1) / c

    # Initialize array
    rotSatPos = np.empty((nSats, 3))

    for i in range(nSats):
        k = satByDistance[i]

        # Rotate satellite ECEF coordinates due to earth rotation during signal
        # travel time
        OmegaEdot = 7.292115147e-5  # Earth's angular velocity [rad/s]
        omegaTau = OmegaEdot * travelTime[k]  # Angle [rad]
        R3 = np.array([[np.cos(omegaTau), np.sin(omegaTau), 0.0],
                       [-np.sin(omegaTau), np.cos(omegaTau), 0.0],
                       [0.0, 0.0, 1.0]])  # Rotation matrix
        rotSatPos[k] = R3 @ satPos[k]  # Apply rotation

    if trop:

        # Initialize array
        trop = np.empty(nSats)

        for i in range(nSats):
            k = satByDistance[i]

            # Transform into topocentric coordinate system
            az, el, rng = topocent(rec_pos, rotSatPos[k])
            # Elevation of satellite w.r.t. receiver [deg]
            # Tropospheric correction
            trop[k] = tropo(np.sin(el*np.pi/180.0), 0.0, 1013.0, 293.0, 50.0,
                            0.0, 0.0, 0.0)

    else:

        trop = 0.0

    # Correct for common bias, satellite clock offset, tropospheric delay
    return (np.linalg.norm(rotSatPos - rec_pos, axis=1) + common_bias - tCorr*c
            + trop)  # [m]


def get_code_phase(ht, hp, hb, prn, eph, code_duration=1e-3, corr=True,
                   corr_type=0, alpha=np.array([0.1676e-07, 0.1490e-07,
                                                -0.1192e-06, -0.5960e-07]),
                   beta=np.array([0.1085e+06, 0.3277e+05, -0.2621e+06,
                                  -0.6554e+05])):
    """Calculate expected code phase [s] for given time and position.

    Precision comparable to predict_pseudoranges(...) but faster.

    Inputs:
      ht - time hypothesis (received time) [s]
      hp - position hypothesis [m,m,m]
      hb - common bias hypothesis [m]
      prn - satellite index
      eph - Ephemeris as matrix
      code_duration - Duration of the code [s], 1e-3 for GPS' C/A code, 4e-3
                      for Galileo's E1BC code, default=1e-3
      corr - Switch for atmospheric correction, default=True
      corr_type - Type of atmospheric correction, default=0
                  0 - Tropospheric correction according to Goad et al. using
                      default parameters
                  1 - Tropospheric correction according to Hopfield using
                      default parameters, ionospheric correction according to
                      Klobuchar
      alpha - Parameters from navigation message for ionospheric correction if
              corr_type=1
      beta - Parameters from navigation message for ionospheric correction if
             corr_type=1

    Output:
      phi - code phase [s]; add code_duration - phi if using with DPE

    Author: Jonas Beuchert
    Inspired by
    Bissig, Pascal, et al. “Fast and Robust GPS Fix Using One Millisecond of
    Data.” Proceedings of the 16th ACM/IEEE International Conference on
    Information Processing in Sensor Networks, 2017, pp. 223–233.
    https://tik-old.ee.ethz.ch/file/f65e5d021e6daee3344591d433b49e83/paper.pdf
    """
    # Speed of light [m/s]
    c = 299792458.0
    # Receiver position in geodetic coordinates
    lat, lon, h = pm.ecef2geodetic(hp[0], hp[1], hp[2])
    # Crude transmit time estimate [s]
    t = ht - 76.5e-3
    # GPS time with respect to 1980 to time of week (TOW)
    tow = np.mod(t, 7 * 24 * 60 * 60)
    if prn.shape[0] < eph.shape[1]:
        # Find column of ephemeris matrix that matches satellite index and time
        col = np.array([find_eph(eph, prn_i, tow) for prn_i in prn])
        # Extract matching columns
        eph = eph[:, col]
    # 2 iterations to refine transmit time estimate
    for it in range(2):
        # Satellite position estimate [m,m,m]
        p = get_sat_pos(t, eph)
        # Propagation delay estimate [s]
        d = np.linalg.norm(hp - p, axis=1) / c
        # Apply common bias
        d = d + hb / c
        # Transmit time estimate [s]
        t = ht - d
    # Satellite clock error [s]
    tCorr = get_sat_clk_corr(ht, prn, eph)
    # Apply satellite clock error to transmission delay
    d = d - tCorr
    if corr:
        if corr_type == 0:
            # Satellite elevation
            az, elev, dist = pm.ecef2aer(p[:, 0], p[:, 1], p[:, 2], lat, lon,
                                         h)
            ddr = np.array([tropo(np.sin(elev_i * np.pi / 180), 0.0, 1013.0,
                                  293.0, 50.0, 0.0, 0.0, 0.0) for elev_i
                            in elev])
            # Tropospheric correction
            tropoDelay = ddr / c
            # Apply tropospheric correction
            d = d + tropoDelay
        else:
            # Ionospheric error non-iterative
            iono_T = ionospheric_klobuchar(hp, p,
                                           np.mod(t[0], 7 * 24 * 60 * 60),
                                           alpha, beta)  # [s]
            # Tropospheric time
            trop_T_equiv = tropospheric_hopfield(hp, p) / c
            d = d + iono_T + trop_T_equiv
    # Code phase [s]
    return np.mod(d, code_duration)


def acquisition(longsignal, IF, Fs, freq_step=500,
                ms_to_process=1, prn_list=np.arange(1, 33),
                expected_doppler=0.0, max_doppler_err=5000.0,
                code_phase_interp='quadratic', fine_freq=True, gnss='gps',
                channel='combined', channel_coherent=False, l1c=False,
                ms_coherent_integration=None, snr_threshold=18.0,
                database=None):
    """Perform signal acquisition using parallel code phase search.

    Secondary codes are ignored.

    Inputs:
        longsignal - Binary GPS signal
        IF - Intermediate frequency [Hz]
        Fs - Sampling frequency [Hz]
        freq_step - Width of frequency bins for coarse acquisition [Hz],
                    choose approximately 1 kHz / ms_to_process [default=500]
        freq_min - Minimum Doppler frequency [Hz], choose dependent on maximum
                   expected receiver velocity [default=-5000]
        ms_to_process - Number of milliseconds to use [default=1]
        prn_list - Indices of satellites to use (PRNs), i.e., satellites that
                   are expected to be visible [default=1:32]
        expected_doppler - Expected Doppler shifts of satellites, which are
                           expected to be visible [Hz] [default=[0, ... 0]]
        max_doppler_err - Maximum expected absolute deviation of true Doppler
                          shifts from expected ones [Hz] [default=5000]
        code_phase_interpolation - Type of code-phase interpolation ('none',
                                   'linear', 'quadratic', 'quadratic5')
        fine_freq - Enable fine frequency calculation [default=True], not
                    tested for Galileo yet, not present for BeiDou and GPS L1C
        gnss - Type of navigation satellite system, 'gps', 'sbas', 'galileo',
               or 'beidou' [default='gps']
        channel - Signal channels to use, 'data', 'pilot', or 'combined',
                  Galileo and BeiDou only [default='combined']
        channel_coherent - Coherent or non-coherent acqusition of channels if
                           channel='combined' [default=False]
        l1c - Use GPS L1C signal instead of GPS L1 C/A codes [default=False]
        ms_coherent_integration - Integration time for single coherent
                                  integration [ms], less or equal than smallest
                                  code duration, if None, then code duration
                                  is used [default=None]
        snr_threshold - Minimum signal-to-noise ratio (SNR) to acquire a
                        satellite [dB] [default=18.0]
        database - Database with pre-sampled satellite code replicas; object of
                   type CodeDB; if present, then replicas are be taken from
                   database instead of created online [default=None]

    Outputs:
        acquired_sv - PRNs of acquired satellites
        acquired_snr - Signal-to-noise ratio of all acquired satellites
        acquired_doppler - Coarse Doppler shift of all acquired satellites [Hz]
        acquired_codedelay - C/A code delay of all acquired satellites
                             [number of samples]
        acquired_fine_freq - Fine carrier wave frequency of all acquired
                             satellites [Hz]
        results_doppler - Coarse or fine Doppler shift of all satellites [Hz]
        results_code_phase - C/A code phase of all satellites [num. of samples]
        results_peak_metric - Signal-to-noise ratio of all satellites

    Author: Jonas Beuchert
    """
    if gnss == 'gps':
        if not l1c:  # L1
            n_prn = 32
            code_duration = 1e-3
        else:  # L1C
            n_prn = 210
            code_duration = 10e-3
            fine_freq = False
    elif gnss == 'sbas':
        n_prn = 138
        code_duration = 1e-3
    elif gnss == 'galileo':
        n_prn = 50
        code_duration = 4e-3
    elif gnss == 'beidou' or gnss == 'bds':
        gnss = 'beidou'
        n_prn = 63
        code_duration = 10e-3
        fine_freq = False
    else:
        raise Exception(
            "Chosen GNSS not supported, select 'gps', 'sbas', 'galileo', or "
            + "'beidou'.")
    # Set number of signal channels
    if (gnss == 'gps' and not l1c) or gnss == 'sbas' \
       or channel == 'data' or channel == 'pilot':
        n_channels = 1
    elif channel == 'combined':
        n_channels = 2
    else:
        raise Exception(
            "Chosen signal channel not supported, select 'data', 'pilot', or "
            + "'combined'.")
    # Check if scalar is passed as expected Doppler
    if not hasattr(expected_doppler, "__len__"):
        expected_doppler = expected_doppler * np.ones(prn_list.shape)
    if prn_list.shape[0] is not expected_doppler.shape[0]:
        raise Exception(
            "prn_list and expected_doppler do not have the same shape.")
    # Number of code sequences in ms_to_process
    n_codes = int(np.ceil(ms_to_process / code_duration * 1e-3))
    # Maximum number of samples to read
    max_samp = longsignal.shape[0]
    # Samples per C/A code sequence
    sample = int(np.ceil(Fs * code_duration))
    sampleindex = np.arange(1, sample + 1)
    # C/A code frequency
    codeFreqBasis = 1.023e6
    # Length of C/A code sequence
    codelength = codeFreqBasis * code_duration
    # Check if integration interval other than code duration shall be used
    if ms_coherent_integration is not None \
       and ms_coherent_integration < code_duration / 1e-3:
        # Number of samples per integration interval
        samples_per_integration = int(np.round(ms_coherent_integration
                                               * 1e-3 * Fs))
        idx = 0
        extended_signal = np.empty(0)
        while idx * samples_per_integration < max_samp:
            # Extract signal chunk
            rawsignal = longsignal[idx * samples_per_integration:
                                   np.min([(idx + 1) * samples_per_integration,
                                           max_samp])]
            # Zero-pad signal chunk to same length as code
            extended_signal = np.concatenate((extended_signal, rawsignal,
                                              np.zeros(sample)))
            idx = idx + 1
        longsignal = extended_signal
        # Number of code sequences
        n_codes = idx
        # Maximum number of samples to read
        max_samp = longsignal.shape[0]
    # Initialization
    acquired_sv = np.empty(0, dtype=int)
    acquired_snr = np.empty(0)
    acquired_doppler = np.empty(0)
    acquired_codedelay = np.empty(0, dtype=int)
    acquired_fine_freq = np.empty(0)
    results_doppler = np.full(n_prn, np.nan)
    results_code_phase = np.full(n_prn, np.nan)
    results_peak_metric = np.full(n_prn, np.nan)
    # Minimum Doppler frequencies to start search
    freq_min = expected_doppler - max_doppler_err
    # Number of frequency bins
    freqNum = 2 * int(np.abs(max_doppler_err) / freq_step) + 1

    # Generate carrier wave replica
    carrier = np.empty((prn_list.shape[0], freqNum, sample), dtype=complex)
    for prn_idx in range(prn_list.shape[0]):
        for freqband in range(freqNum):
            dopplershift = freq_min[prn_idx] + freq_step * freqband
            carrier[prn_idx, freqband] = np.exp(
                1j * 2.0 * np.pi * (IF + dopplershift) * sampleindex / Fs)

    # Loop over all satellites that are expected to be visible
    for prn_idx in range(prn_list.shape[0]):
        svindex = prn_list[prn_idx]
        # Initialize correlogram
        correlation = np.zeros((freqNum, sample))
        # Iterate over channels
        for channel_idx in range(n_channels):
            if (gnss == 'gps' and not l1c) or gnss == 'sbas':
                if database is None:
                    # Generate C/A code replica
                    ocode = generate_ca_code(svindex)
                    ocode = np.concatenate((ocode, ocode))
                    scode = ocode[np.ceil(sampleindex * codeFreqBasis / Fs
                                          ).astype(int) - 1]
                    replica = scode
                else:
                    # Get C/A code replica from database
                    replica = database.query_db(gnss, svindex)
            elif (gnss == 'gps' and l1c) or gnss == 'galileo' \
                    or gnss == 'beidou':
                # Generate E1 / B1C / L1C code replica
                if not channel == 'combined' or not channel_coherent:
                    # Non-coherent acqusition of channels
                    # -> Either data or pilot signal
                    if channel_idx == 0 and channel != 'pilot':
                        # Acqusition for data channel E1B / B1C_data / L1C_d
                        pilot = False
                    else:
                        # Acqusition for pilot channel E1C / B1C_pilot / L1C_p
                        pilot = True
                    if database is None:
                        # Generate code replica
                        if gnss == 'galileo':
                            # E1 code
                            replica = generate_e1_code(svindex, Fs, pilot=pilot)
                        elif gnss == 'beidou':
                            # B1C code
                            replica = generate_b1c_code(svindex, Fs, pilot=pilot)
                        elif gnss == 'gps' and l1c:
                            # L1C code
                            replica = generate_l1c_code(svindex, Fs, pilot=pilot)
                    else:
                        # Get code replica from database
                        replica = database.query_db(gnss, svindex, pilot)
                else:
                    # Coherent acqusition of channels
                    # -> Combine both channels
                    if channel_idx == 0:
                        if database is None:
                            if gnss == 'galileo':
                                replica_data = generate_e1_code(svindex, Fs,
                                                                pilot=False)
                                replica_pilot = generate_e1_code(svindex, Fs,
                                                                 pilot=True)
                            elif gnss == 'beidou':
                                replica_data = generate_b1c_code(svindex, Fs,
                                                                 pilot=False)
                                replica_pilot = generate_b1c_code(svindex, Fs,
                                                                  pilot=True)
                            elif gnss == 'gps' and l1c:
                                replica_data = generate_l1c_code(svindex, Fs,
                                                                 pilot=False)
                                replica_pilot = generate_l1c_code(svindex, Fs,
                                                                  pilot=True)
                        else:
                            replica_data = database.query_db(gnss, svindex,
                                                             pilot=False)
                            replica_pilot = database.query_db(gnss, svindex,
                                                              pilot=True)
                        # Add data signal to pilot signal
                        replica = replica_data + replica_pilot
                    else:
                        # Subtract data signal from pilot signal
                        replica = - replica_data + replica_pilot
            # Correlation in frequency domain
            temp3 = fft_lib.fft(replica)
            for idx in range(n_codes):  # Process each code sequence
                # Extract signal chunk
                rawsignal = longsignal[idx * sample:
                                       np.min([(idx + 1) * sample, max_samp])]
                # Zero padding to adjust for code sequence length
                rawsignal = np.concatenate((rawsignal, np.zeros(
                    sampleindex.shape[0] - rawsignal.shape[0])))
                for freqband in range(freqNum):
                    temp1 = rawsignal \
                        * carrier[prn_idx, freqband]
                    temp2 = np.conj(fft_lib.fft(temp1))
                    correlation[freqband] = correlation[freqband] \
                        + np.abs(fft_lib.ifft(temp3 * temp2))**2
            if channel == 'combined' and channel_coherent:
                correlation_0 = correlation
                correlation = np.zeros((freqNum, sample))
        # Normalize
        correlation = correlation * ms_to_process * code_duration / 4e-3
        if not channel == 'combined' or not channel_coherent:
            # Normalize
            correlation = correlation / n_channels
        else:
            # Take max
            correlation = np.maximum(np.abs(correlation_0),
                                     np.abs(correlation))
        # Find peak
        fbin = np.argmax(np.max(np.abs(correlation), axis=1))
        codePhase = np.argmax(np.max(np.abs(correlation), axis=0))
        peak = correlation[fbin, codePhase]
        # Doppler shift
        Doppler = freq_min[prn_idx] + freq_step * fbin
        # Signal-to-noise ration (SNR)
        codechipshift = np.ceil(Fs / codeFreqBasis)
        ind_snr = np.concatenate((np.arange(codePhase - codechipshift),
                                  np.arange(codePhase + codechipshift - 1,
                                            sample)))
        corr_snr = correlation[fbin, ind_snr.astype(int)]
        # import matplotlib.pyplot as plt
        # plt.plot(correlation[fbin])
        SNR = 10.0 * np.log10(peak**2 /
                              (np.sum(corr_snr**2) / corr_snr.shape[0]))
        # SNR = 10.0 * np.log10(peak /
        #                       np.mean(corr_snr))
        # # SNR = peak / np.max(corr_snr)
        # plt.title(gnss + " PRN " + str(svindex) + ", SNR = " + str(np.round(SNR,1)))
        # plt.show()

        # Acquisition threshold
        if SNR >= snr_threshold:
            acquired_sv = np.append(acquired_sv, svindex)
            acquired_snr = np.append(acquired_snr, SNR)
            acquired_doppler = np.append(acquired_doppler, Doppler)
            acquired_codedelay = np.append(acquired_codedelay, codePhase)
        results_peak_metric[svindex - 1] = SNR
        codePhase = interpolate_code_phase(correlation[fbin],
                                           mode=code_phase_interp)
        results_code_phase[svindex - 1] = sample - codePhase + 1.0
        results_doppler[svindex - 1] = Doppler

    # Fine frequency calculation
    if fine_freq:
        # Number of ms to perform FFT
        acq_L = 10
        longSignalIndex = np.mod(np.arange(1, sample * (acq_L + int(
            code_duration / 1e-3))), sample)
        longSignalIndex[longSignalIndex == 0] = sample
        if longSignalIndex.shape[0] > rawsignal.shape[0]:
            longsignal = np.concatenate((longsignal, np.zeros(
                longSignalIndex.shape[0] - rawsignal.shape[0])))
        longrawsignal = longsignal[longSignalIndex - 1]

        for svindex in range(acquired_sv.shape[0]):
            if (gnss == 'gps' and not l1c) or gnss == 'sbas':
                caCode = generate_ca_code(acquired_sv[svindex])
                codeValueIndex = np.floor((1.0 / Fs *
                                           (np.arange(acq_L * sample) + 1.0))
                                          / (1.0 / codeFreqBasis))
                longCaCode = caCode[np.mod(codeValueIndex, codelength).astype(
                    int)]
            elif gnss == 'galileo':
                caCode = generate_e1_code(acquired_sv[svindex], Fs)
                codeValueIndex = np.floor((1.0 / Fs *
                                           (np.arange(acq_L * sample) + 1.0))
                                          / (1.0 / codeFreqBasis))
                longCaCode = np.tile(caCode, acq_L)

            CarrSignal = longrawsignal[
                (sample - acquired_codedelay[svindex] - 1):
                (sample - acquired_codedelay[svindex])
                + acq_L * sample - 1] * longCaCode

            fftlength = CarrSignal.shape[0] * 20
            fftSignal = np.abs(fft_lib.fft(CarrSignal, fftlength))
            # Find acquired satellite in original RPN list
            prn_idx = np.where(prn_list == acquired_sv[svindex])[0][0]
            # Get frequency index range for this satellite
            maxDoppler = -freq_min[prn_idx]  # [Hz]
            minFreq = IF - maxDoppler
            maxFreq = IF + maxDoppler
            minIndex = np.ceil(minFreq / Fs * fftlength).astype(int)
            minIndex = np.max([1, minIndex])
            maxIndex = np.ceil(maxFreq / Fs * fftlength).astype(int)
            maxIndex = np.min([fftlength, maxIndex])
            FreqPeakIndex = np.argmax(fftSignal[minIndex-1:maxIndex])
            FreqPeakIndex = FreqPeakIndex + minIndex
            fineDoppler = FreqPeakIndex * Fs / fftlength

            acquired_fine_freq = np.append(acquired_fine_freq, fineDoppler)

            results_doppler[acquired_sv[svindex] - 1] = fineDoppler - IF

    return acquired_sv, acquired_snr, acquired_doppler, acquired_codedelay,\
        acquired_fine_freq, results_doppler, results_code_phase,\
        results_peak_metric


def acquisition_simplified(signals, utc, pos_geo, rinex_file=None, eph=None,
                           system_identifier='G', elev_mask=15,
                           intermediate_frequency=4092000.0,
                           frequency_bins=np.array([0])):
    """Satellite acquisition for snapper with parallel code phase search (PCPS).

    Sampling frequency and snapshot duration fixed to snapper parameters.
    Includes prediction of set of visible satellites.
    Acquisition of all potentially visible satellites of one satellite system.
    Can process a batch of snapshots.
    Non-coherent integration over time and across satellite signal channels.
    Quadratic code-phase interpolation based on three points.
    Core computations in single precision.
    GPS and SBAS: L1 C/A signal
    Galileo: E1 signal with data and primary pilot channel
    BeiDou: B1C signal with data and primary pilot channel
    Reads pre-sampled satellite signal replicas from 'codes_X.npy'.

    Inputs:
        signals - Binary signal snapshots {-1,+1}, Nx49104 NumPy array
        utc - Time stamps of snapshots in UTC, NumPy array of numpy.datetime64
        pos_geo - Initial geodetic position (latitude [deg], longitude [deg],
                  height [m]), NumPy array
        rinex_file - Path to RINEX navigation file, default=None
        eph - Navigation data for desired time interval and satellite system,
              2D numpy array with 21 rows, default=None, either 'rinex_file' or
              'eph' must be provided, 'eph' is recommended
        system_identifier - 'G' for GPS, 'S' for SBAS, 'E' for Galileo, or 'C'
                            for BeiDou, default='G'
        elev_mask - Satellite elevation threshold [deg], default=15
        intermediate_frequency - (Offset corrected) intermediate frequency,
                                 default=4092000.0
        frequency_bins - Centres of acquisition frequency bins relative to
                         intermediate frequency for PCPS, 1D NumPy array,
                         default=np.array([0])

    Outputs:
        snapshot_idx_vec - Index of snapshot to which the following results
                           belong, 1D NumPy array
        prn_vec - PRN of satellite to which the following results belong, all
                  potentially visible satellites are included, 1D NumPy array
        code_phase_vec - Code phase estimates [ms] of all potentially visible
                         satellites in the convention that is used in
                         coarse_time_navigation.py, 1D NumPy array
        snr_vec - Something like the signal-to-noise ratio [dB] that can be used
                  by the classifier in bayes_classifier_snr.npy to assess
                  satellite reliability, 1D NumPy array
        eph_idx_vec - Column indices of the potentially visble satellites in
                      the navigation data matrix, 1D NumPy array
        frequency_vec - Carrier frequency estimates [Hz] w.r.t. intermediate
                        frequency for all potentially visible satellites, 1D
                        NumPy array
        frequency_error_vec - Differences between estimated carrier frequencies
                              and predicted Doppler shifts [Hz], 1D Numpy array

    Author: Jonas Beuchert
    """
    # Remove signal mean to avoid DC artefacts in the frequency domain
    signals = signals.astype(np.float32)
    signals = signals - np.mean(signals, axis=-1, keepdims=True)

    # Sampling frequency
    sampling_frequency = 4092000.0

    # Snapshot duration (12 ms)
    snapshot_duration = 12e-3

    # Check some inputs
    if not isinstance(signals, np.ndarray) or signals.ndim != 2:
        raise Exception(
            "'signals' must be a 2D NumPy array.")
    if signals.shape[1] != int(sampling_frequency*snapshot_duration):
        raise Exception(
            "The second axis of 'signals' must have a length of {}.".format(
                int(sampling_frequency*snapshot_duration)
            ))
    if not isinstance(utc, np.ndarray) or utc.ndim != 1:
        raise Exception(
            "'utc' must be a 1D NumPy array.")
    if not isinstance(pos_geo, np.ndarray) or pos_geo.ndim != 1 \
         or pos_geo.shape[0] != 3:
        raise Exception(
            "'pos_geo' must be a 1D NumPy array with three elements.")
    if not isinstance(frequency_bins, np.ndarray) or pos_geo.ndim != 1:
        raise Exception(
            "'frequency_bins' must be a 1D NumPy array.")
    if signals.shape[0] != utc.shape[0]:
        raise Exception(
            "The first dimensions of 'signals' and 'utc' must have the same " \
            "size, but 'signals' has {} elements and 'utc' has {} elements.".format(
                signals.shape[0], utc.shape[0])
            )
    if rinex_file is None and eph is None:
        raise Exception(
            "Either 'eph' or 'rinex_file' must be provided, but both are 'None'."
            )
    if eph is not None and (not isinstance(eph, np.ndarray) or eph.ndim != 2):
        raise Exception(
            "'eph' must be a 2D NumPy array."
            )
    if eph is not None and eph.shape[0] != 21:
        raise Exception(
            "'eph' must have 21 rows, i.e., its first dimension must have size 21."
            )

    # Convert geodetic coordinates to ECEF (Cartesian XYZ)
    pos_ecef = np.empty(3)
    pos_ecef[0], pos_ecef[1], pos_ecef[2] = pm.geodetic2ecef(
        pos_geo[0], pos_geo[1], pos_geo[2]
        )

    if eph is None:
        # Read navigation data file
        try:
            eph = rinexe(rinex_file, system_identifier)
        except:
            raise Exception(
                "Could not read RINEX navigation data file.")
    # Check which PRNs are present in navigation data file
    prn = np.unique(eph[0]).astype(int)
    if prn.shape[0] == 0:
        raise Exception(
            "Could not find any satellites of the selected system in RINEX navigation data file.")

    # Set satellite signal code period depending on system
    if system_identifier == 'G' or system_identifier == 'S':
        code_period = 1e-3  # C/A codes have a period of 1 ms
    elif system_identifier == 'E':
        code_period = 4e-3  # E1 codes have a period of 4 ms
    elif system_identifier == 'C':
        code_period = 10e-3  # B1C codes have a period of 10 ms
    else:
        raise Exception(
            "Chosen GNSS not supported. Select 'G' for GPS, 'S' for SBAS, 'E' "
            + "for Galileo, or 'C' for BeiDou as 'system_identifier'.")

    # Convert UTC to GPS time
    reference_date = np.datetime64('1980-01-06')  # GPS reference date
    leap_seconds = np.timedelta64(18, 's')  # Hardcoded 18 leap seconds
    time = (utc - reference_date + leap_seconds) / np.timedelta64(1, 's')

    if system_identifier == 'C':
        # Convert GPS time to BeiDou time, but keep the GPS week number
        time = time - 14.0  # - 820108814.0 (this would change to BeiDou weeks)

    # Absolute system time to time of week (TOW)
    tow = np.mod(time, 7 * 24 * 60 * 60)

    # Vectorize everything: one row for one satellite at one point in time
    prn_vec = np.tile(prn, tow.shape[0])
    tow_vec = np.repeat(tow, prn.shape[0])
    # Remember which snapshot belongs to which row
    snapshot_idx_vec = np.repeat(np.arange(tow.shape[0]), prn.shape[0])

    # Find column for each satellite in ephemerides array
    # Initialize array to store column indices
    eph_idx_vec = np.empty(tow.shape[0] * prn.shape[0], dtype=int)
    # Time differences between ephemerides timestamps and snapshot timestamps
    if eph[20, -1] > 7 * 24 * 60 * 60:
        # eph[20] holds absolute GPS time
        differences = eph[20] - time.reshape(-1, 1)
        # Convert to time of week (TOW) [s]
        eph[20] = np.mod(eph[20], 7 * 24 * 60 * 60)
    else:
        # eph[20] holds time of week (TOW)
        differences = eph[20] - tow.reshape(-1, 1)
    # Ephemerides timestamp should be smaller than snapshot timestamp
    # So, ignore all rows with larger timestamp
    differences[differences > 0] = -np.inf
    # Iterate over all PRNs
    for sat_idx, sat_id in enumerate(prn):
        # Get column indices of this PRN
        eph_idx_sat = np.where(eph[0] == sat_id)[0]
        # Get time differences for this PRN
        differences_sat = differences[:, eph_idx_sat]
        # Find timestamps closest to zero
        eph_idx = eph_idx_sat[np.argmax(differences_sat, axis=-1)]
        # Store indices for this PRN
        eph_idx_vec[sat_idx::prn.shape[0]] = eph_idx

    # Crude transmit time estimate [s]
    transmit_time_vec = tow_vec - 76.5e-3

    # Get satellite position at estimated transmit time
    sat_pos_vec, sat_vel_vec = get_sat_pos_vel(transmit_time_vec,
                                               eph[:, eph_idx_vec])

    # Convert to elevation above horizon in degrees
    _, elev_vec, _ = pm.ecef2aer(
        sat_pos_vec[:, 0], sat_pos_vec[:, 1], sat_pos_vec[:, 2],
        pos_geo[0], pos_geo[1], pos_geo[2]
        )

    # Predict visible satellites
    # Satellites with elevation larger than threshold
    vis_sat_idx = (elev_vec > elev_mask)
    prn_vec = prn_vec[vis_sat_idx]
    snapshot_idx_vec = snapshot_idx_vec[vis_sat_idx]
    sat_pos_vec = sat_pos_vec[vis_sat_idx]
    sat_vel_vec = sat_vel_vec[vis_sat_idx]

    # Estimate Doppler shifts
    c = 299792458.0  # Speed of light [m/s]
    L1 = 1575.42e6  # GPS signal frequency [Hz]
    wave_length = c / L1  # Wave length of transmitted signal
    # Doppler shift (cf. 'Cycle slip detection in single frequency GPS carrier
    # phase observations using expected Doppler shift')
    doppler_vec = (((pos_ecef - sat_pos_vec) / np.linalg.norm(
        pos_ecef - sat_pos_vec, axis=-1, keepdims=True
        )) * sat_vel_vec).sum(1) / wave_length

    # Use single precision
    doppler_vec = doppler_vec.astype(np.float32)

    # Account for search along frequency axis
    n_bins = frequency_bins.shape[0]
    frequency_bins = np.tile(frequency_bins, doppler_vec.shape[0]).astype(np.float32)
    doppler_vec = np.repeat(doppler_vec, n_bins)
    snapshot_idx_vec_f = np.repeat(snapshot_idx_vec, n_bins)
    doppler_vec += frequency_bins

    # Samples per C/A code sequence
    sample = int(sampling_frequency * code_period)
    sample_idx = np.arange(1, sample + 1)

    if np.isscalar(intermediate_frequency):
        intermediate_frequency_f = intermediate_frequency
    else:
        intermediate_frequency_f = intermediate_frequency[snapshot_idx_vec_f]

    # Generate carrier wave replicas
    carrier_vec = np.exp(np.complex64(1j * 2.0 * np.pi / sampling_frequency)
                         * np.array([(intermediate_frequency_f + doppler_vec)],
                                    dtype=np.float32).T
                         @ np.array([sample_idx], dtype=np.float32))

    if system_identifier == 'C':
        # Zero Padding for BeiDou because 10 does not divide 12
        signals = np.hstack((signals, np.zeros((signals.shape[0], int((2*code_period-snapshot_duration)*sampling_frequency)), dtype=np.float32)))
        snapshot_duration = 20e-3

    # Number of code sequences
    n_codes = int(snapshot_duration / code_period)

    # Create signal chunks, 1 ms, 4 ms, or 10 ms each, new array dimension
    signals = np.array(np.hsplit(signals, n_codes), dtype=np.float32).transpose(1, 0, 2)
    signals = signals[snapshot_idx_vec_f]

    # Wipe-off carrier
    signals = signals * np.repeat(carrier_vec[:, np.newaxis, :], n_codes, axis=1)

    # Transform snapshot chunks into frequency domain
    signals = np.conj(fft_lib.fft(signals))

    # Adjust SBAS PRNs
    if system_identifier == 'S':
        prn -= 100
        prn_vec -= 100

    # Set number of signal channels
    if system_identifier == 'G' or system_identifier == 'S':
        n_channels = 1
    else:
        n_channels = 2

    # Satellite code replicas with single precision
    replicas = np.load("codes_" + system_identifier + ".npy")

    # Transform relevant replicas into frequency domain
    replicas_f = np.empty_like(replicas, dtype=np.complex64)
    replicas_f[prn-1] = fft_lib.fft(replicas[prn-1])
    # Get matching replica for each row
    replicas_f = replicas_f[prn_vec-1]
    # Repeat replica for each code chunk, new code chunk dimension
    replicas_f = np.repeat(replicas_f[:, np.newaxis, :, :], n_codes, axis=1)

    # Account for multiple channels, create channel dimension
    signals = np.repeat(signals[:, :, np.newaxis, :], n_channels, axis=2)

    # Correlate in frequency domain and transform back into time domain
    correlation = np.abs(fft_lib.ifft(np.repeat(replicas_f, n_bins, axis=0) * signals))**2

    # Sum all channels and all signals chunks of one
    # snapshot (non-coherent integration)
    correlation = np.sum(correlation, axis=2)
    correlation = np.sum(correlation, axis=1)

    # Normalize
    correlation = correlation * np.float32(snapshot_duration * 1e3 * code_period / 4e-3 / n_channels)

    # Create new dimension for frequency search space
    correlation = correlation.reshape((int(correlation.shape[0] / n_bins), n_bins, correlation.shape[1]))

    # Find correlogram peaks
    bin_vec = np.argmax(np.max(correlation, axis=-1), axis=-1)
    code_phase_vec = np.argmax(np.max(correlation, axis=-2), axis=-1)
    correlation = correlation[np.arange(correlation.shape[0]), bin_vec, :]  # Remove frequency dimension
    peak_vec = correlation[np.arange(code_phase_vec.shape[0]), code_phase_vec]
    doppler_vec = doppler_vec.reshape((int(doppler_vec.shape[0] / n_bins), n_bins))
    frequency_vec = doppler_vec[np.arange(code_phase_vec.shape[0]), bin_vec]
    frequency_error_vec = frequency_bins[bin_vec]

    # Quadratically interpolate code phases
    # Algorithm from Chapter 8.12 of
    # Tsui, James Bao-Yen. Fundamentals of Global Positioning System
    # receivers: a software approach. Vol. 173. John Wiley & Sons, 2005.
    # http://twanclik.free.fr/electricity/electronic/pdfdone7/Fundamentals%20of%20Global%20Positioning%20System%20Receivers.pdf
    Y = np.array([correlation[np.arange(code_phase_vec.shape[0]),
                              code_phase_vec - 1],  # y1 = left_val
                  peak_vec,  # y2 = max_val
                  correlation[np.arange(code_phase_vec.shape[0]),
                              np.mod(code_phase_vec + 1, sample)]  # y3 = right_val
                  ], dtype=np.float32)
    x1 = -1.0
    x2 = 0.0
    x3 = 1.0
    X = np.array([[x1**2, x1, 1.0],
                  [x2**2, x2, 1.0],
                  [x3**2, x3, 1.0]])
    A = np.linalg.lstsq(X, Y, rcond=None)[0]
    a = A[0]
    b = A[1]
    # c = A[2]
    x = -b / 2.0 / a
    code_phase_interp_vec = sample - code_phase_vec - x

    # Signal-to-noise ratio (SNR)
    code_freq_basis = 1.023e6  # C/A code frequency
    code_chip_shift = int(np.ceil(sampling_frequency / code_freq_basis))
    # Remove peaks
    correlation[np.repeat(np.arange(code_phase_vec.shape[0]),
                          2*code_chip_shift+1),
                np.mod(np.linspace(code_phase_vec-code_chip_shift,
                                   code_phase_vec+code_chip_shift,
                                   2*code_chip_shift+1, dtype=int).T.flatten(),
                       sample)] = np.nan
    # SNR
    snr_vec = 10.0 * np.log10(peak_vec**2 / (np.nansum(correlation**2, axis=-1)
                                             / correlation.shape[1]))

    # Convert code phases and SNR to convention used by CTN function
    code_phase_vec = code_phase_interp_vec / sampling_frequency / 1.0e-3

    # Adjust SBAS PRNs
    if system_identifier == 'S':
        prn_vec += 100

    return snapshot_idx_vec, prn_vec, code_phase_vec, snr_vec, \
        eph_idx_vec[vis_sat_idx], frequency_vec, frequency_error_vec


def topocent(X, dx):
    """Transform dx into topocentric coordinate system with origin at X.

    Inputs:
        X - Origin in ECEF XYZ coordinates
        dx - Point in ECEF XYZ coordinates

    Outputs:
        az - Azimuth from north positive clockwise [deg]
        el - Elevation angle [deg]
        dist - Length in units like the input
    """
    dtr = np.pi/180.0
    lat, lon, h = pm.ecef2geodetic(X[0], X[1], X[2])
    cl = np.cos(lon*dtr)
    sl = np.sin(lon*dtr)
    cb = np.cos(lat*dtr)
    sb = np.sin(lat*dtr)
    F = np.array([np.array([-sl, -sb*cl, cb*cl]),
                  np.array([cl, -sb*sl, cb*sl]),
                  np.array([0.0, cb, sb])])
    local_vector = F.T@dx
    E = local_vector[0]
    N = local_vector[1]
    U = local_vector[2]
    hor_dis = np.sqrt(E**2+N**2)
    if hor_dis < 1.e-20:
        az = 0.0
        el = 90.0
    else:
        az = np.arctan2(E, N)/dtr
        el = np.arctan2(U, hor_dis)/dtr
    if az < 0.0:
        az = az+360.0
    dist = np.sqrt(dx[0]**2+dx[1]**2+dx[2]**2)
    return az, el, dist


def tropo(sinel, hsta, p, tkel, hum, hp, htkel, hhum):
    """Calculate tropospheric correction.

    The range correction ddr in m is to be subtracted from pseudoranges and
    carrier phases.

    Inputs:
        sinel - Sin of elevation angle of satellite
        hsta - Height of station in km
        p - Atmospheric pressure in mb at height hp
        tkel - Surface temperature in degrees Kelvin at height htkel
        hum - Humidity in % at height hhum
        hp - Height of pressure measurement in km
        htkel - Height of temperature measurement in km
        hhum - Height of humidity measurement in km

    Output:
        ddr - Range correction [m]

    Reference
    Goad, C.C. & Goodman, L. (1974) A Modified Tropospheric Refraction
    Correction Model. Paper presented at the American Geophysical Union
    Annual Fall Meeting, San Francisco, December 12-17.

    Author: Jonas Beuchert
    """
    a_e = 6378.137  # Semi-major axis of earth ellipsoid
    b0 = 7.839257e-5
    tlapse = -6.5
    tkhum = tkel + tlapse * (hhum - htkel)
    atkel = 7.5 * (tkhum - 273.15) / (237.3 + tkhum - 273.15)
    e0 = 0.0611 * hum * 10**atkel
    tksea = tkel - tlapse * htkel
    em = -978.77 / (2.8704e6 * tlapse * 1.0e-5)
    tkelh = tksea + tlapse * hhum
    e0sea = e0 * (tksea / tkelh)**(4 * em)
    tkelp = tksea + tlapse * hp
    psea = p * (tksea / tkelp)**em
    if sinel < 0.0:
        sinel = 0.0
    tropo = 0.0
    done = False
    refsea = 77.624e-6 / tksea
    htop = 1.1385e-5 / refsea
    refsea = refsea * psea
    ref = refsea * ((htop - hsta) / htop)**4
    while True:
        rtop = (a_e + htop)**2 - (a_e + hsta)**2 * (1 - sinel**2)
        if rtop < 0.0:
            rtop = 0.0  # Check to see if geometry is crazy
        rtop = np.sqrt(rtop) - (a_e + hsta) * sinel
        a = -sinel / (htop - hsta)
        b = -b0 * (1.0 - sinel**2) / (htop - hsta)
        rn = np.zeros(8)
        for i in range(8):
            rn[i] = rtop**(i + 2)
        alpha = np.array([2 * a, 2 * a**2 + 4 * b / 3, a * (a**2 + 3 * b),
                          a**4 / 5 + 2.4 * a**2 * b + 1.2 * b**2,
                          2 * a * b * (a**2 + 3 * b) / 3,
                          b**2 * (6 * a**2 + 4 * b) * 1.428571e-1, 0, 0])
        if b**2 > 1.0e-35:
            alpha[6] = a * b**3 / 2
            alpha[7] = b**4 / 9
        dr = rtop
        dr = dr + np.sum(alpha * rn)
        tropo = tropo + dr * ref * 1000
        if done:
            return tropo
        done = True
        refsea = (371900.0e-6 / tksea - 12.92e-6) / tksea
        htop = 1.1385e-5 * (1255 / tksea + 0.05) / refsea
        ref = refsea * e0sea * ((htop - hsta) / htop)**4


def tropospheric_hopfield(pos_rcv, pos_sv, T_amb=20.0, P_amb=101.0,
                          P_vap=0.86):
    """Approximate troposspheric group delay.

    Inputs:
        pos_rcv - XYZ position of reciever [m,m,m]
        pos_sv - XYZ matrix position of GPS satellites [m,m,m]
        T_amb - Temperature at reciever antenna location [deg. Celsius]
        P_amb - Air pressure at reciever antenna location [hPa]
        P_vap - Water vapore pressure at reciever antenna location [hPa]

    Output:
        Delta_R_Trop - Tropospheric error correction [m]

    Author: Jonas Beuchert
    Reference:
        "GPS Theory and application", edited by B. Parkinson, J. Spilker.
    """
    # Receiver position in geodetic coordinates
    lat, lon, h = pm.ecef2geodetic(pos_rcv[0], pos_rcv[1], pos_rcv[2],
                                   deg=False)
    # Azimuth [rad], elevation [rad]
    az, El, dist = pm.ecef2aer(pos_sv[:, 0], pos_sv[:, 1], pos_sv[:, 2],
                               lat, lon, h, deg=False)

    # Zenith hydrostatic delay
    Kd = 1.55208e-4 * P_amb * (40136.0 + 148.72 * T_amb) / (T_amb + 273.16)

    # Zenith Wet Delay
    Kw = -0.282 * P_vap / (T_amb + 273.16) + 8307.2 * P_vap / (T_amb
                                                               + 273.16)**2

    Denom1 = np.sin(np.sqrt(El**2 + 1.904e-3))
    Denom2 = np.sin(np.sqrt(El**2 + 0.6854e-3))
    # Troposhpheric delay correctoion
    return Kd / Denom1 + Kw / Denom2  # Meter


def tropospheric_tsui(elevation):
    """Troposheric delay.

    Input:
        elevation - Elevation angle between user and satellite [deg]

    Output:
        tropospheric_delay - Estimated troposheric delay [m]

    Author: Jonas Beuchert
    Reference:
        Tsui, James Bao-Yen. Fundamentals of global positioning system
        receivers: a software approach. Vol. 173. John Wiley & Sons, 2005.
    """
    return 2.47 / (np.sin(np.deg2rad(elevation)) + 0.0121)


def ionospheric_klobuchar(r_pos, pos_sv, gps_time,
                          alpha=np.array([0.1676e-07, 0.1490e-07, -0.1192e-06,
                                          -0.5960e-07]),
                          beta=np.array([0.1085e+06, 0.3277e+05, -0.2621e+06,
                                         -0.6554e+05])):
    """Approximate ionospheric group delay.

    Compute an ionospheric range correction for the GPS L1 frequency from the
    parameters broadcasted in the GPS navigation message.

    Not validated yet.

    Inputs:
        r_pos - XYZ position of reciever [m]
        pos_sv - XYZ matrix position of GPS satellites [m]
        gps_time - Time of Week [s]
        alpha - Coefficients of a cubic equation representing the amplitude of
                the vertical delay (4 coefficients)
        beta - Coefficients of a cubic equation representing the period of the
               model (4 coefficients)

    Output:
       Delta_I - Ionospheric slant range correction for the L1 frequency [s]

    Author: Jonas Beuchert
    References:
       Klobuchar, J.A., (1996) "Ionosphercic Effects on GPS", in Parkinson,
           Spilker (ed), "Global Positioning System Theory and Applications,
           pp. 513-514.
       ICD-GPS-200, Rev. C, (1997), pp. 125-128
       NATO, (1991), "Technical Characteristics of the NAVSTAR GPS", pp. A-6-31
           - A-6-33
    """
    # Semicircles, latitude, and longitude
    GPS_Rcv = np.array([0.0, 0.0, 0.0])
    GPS_Rcv[0], GPS_Rcv[1], GPS_Rcv[2] = pm.ecef2geodetic(r_pos[0], r_pos[1],
                                                          r_pos[2])
    Lat = GPS_Rcv[0] / 180.0
    Lon = GPS_Rcv[1] / 180.0
    S = pos_sv.shape
    m = S[0]

    A0, El, dist = pm.ecef2aer(pos_sv[:, 0], pos_sv[:, 1], pos_sv[:, 2],
                               GPS_Rcv[0], GPS_Rcv[1], GPS_Rcv[2])
    # Semicircle elevation
    E = El / 180.0
    # Semicircle azimuth
    A = A0 / 180.0 * np.pi
    # Calculate the earth-centered angle, Psi (semicircle)
    Psi = 0.0137 / (E + 0.11) - 0.022

    # Compute the subionospheric latitude, Phi_L (semicircle)
    Phi_L = Lat + Psi * np.cos(A)
    Phi_L = np.clip(Phi_L, -0.416, 0.416)

    # Compute the subionospheric longitude, Lambda_L (semicircle)
    Lambda_L = Lon + (Psi * np.sin(A) / np.cos(Phi_L * np.pi))

    # Find the geomagnetic latitude, Phi_m, of the subionospheric location
    # looking towards each GPS satellite:
    Phi_m = Phi_L + 0.064 * np.cos((Lambda_L - 1.617) * np.pi)

    # Find the local time, t, at the subionospheric point
    t = 4.23e4 * Lambda_L + gps_time  # GPS_Time [s]
    for i in range(t.shape[0]):
        if t[i] > 86400:
            t[i] = t[i] - 86400.0
        elif t[i] < 0:
            t[i] = t[i] + 86400.0

    # Convert slant time delay, compute the slant factor, F
    F = 1.0 + 16.0 * (0.53 - E)**3

    # Compute the ionospheric time delay T_iono by first computing x
    Per = beta[0] + beta[1] * Phi_m + beta[2] * Phi_m**2 + beta[3] * Phi_m**3
    Per = np.clip(Per, 72000, None)  # Period
    x = 2.0 * np.pi * (t - 50400.0) / Per  # [rad]
    AMP = alpha[0] + alpha[1] * Phi_m + alpha[2] * Phi_m**2 + alpha[3] \
        * Phi_m**3
    AMP = np.clip(AMP, 0, None)
    T_iono = np.empty(m)
    for i in range(m):
        if np.abs(x[i]) > 1.57:
            T_iono[i] = F[i] * 5e-9
        else:
            T_iono[i] = F[i] * (5e-9 + AMP[i] * (1.0 - x[i]**2 / 2.0 + x[i]**4
                                                 / 24))
    return T_iono


def ionospheric_tsui(elevation, azimuth, latitude, longitude, gps_time,
                     alpha, beta):
    """Additional ionospheric delay time.

    Compute an ionospheric range correction for the GPS L1 frequency from the
    parameters broadcasted in the GPS navigation message.

    Inputs:
        elevation - Elevation angle between user and satellite [semicircles]
        azimuth - Azimuth angle between user and satellite, measured clockwise
                  positive from true North [semicircles]
        latitude - User geodetic latitude [semicircle]
        longitude - User geodetic longitude [semicircle]
        gps_time - System time [s]
        alpha - Coefficients of a cubic equation representing the amplitude of
                the vertical delay (4 coefficients)
        beta - Coefficients of a cubic equation representing the period of the
               model (4 coefficients)

    Output:
       T_iono - Additional ionospheric dealy time estimate [s]

    Author: Jonas Beuchert
    Reference:
        Tsui, James Bao-Yen. Fundamentals of global positioning system
        receivers: a software approach. Vol. 173. John Wiley & Sons, 2005.
    """
    # Central angle [semicircle] (typos in the book)
    psi = 0.0137 / (elevation + 0.11) - 0.022
    # Geomagnetic latitude[semicircle]
    phi_i = latitude + psi * np.cos(azimuth * np.pi)
    if phi_i > 0.416:
        phi_i = 0.416
    elif phi_i < -0.416:
        phi_i = -0.416
    # Geomagnetic latitude [semicircle] (typo in the book)
    lambda_i = longitude + psi * np.sin(azimuth*np.pi) / np.cos(phi_i*np.pi)
    # Local time [s]
    t = 4.32e4 * lambda_i + gps_time
    t = np.mod(t, 86400)
    # Geomagnetic latitude [semicircles]
    phi_m = phi_i + 0.064*np.cos((lambda_i - 1.617)*np.pi)
    # Obliquity factor
    T = 1.0 + 16.0 * (0.53 - elevation)**3
    # PER [s]
    PER = 0.0
    for n in range(4):
        PER = PER + beta[n] * phi_m**n
    if PER < 72000:
        PER = 72000.0
    # Phase [rad]
    x = 2.0*np.pi*(t-50400.0) / PER
    # AMP [s]
    AMP = 0.0
    for n in range(4):
        AMP = AMP + alpha[n] * phi_m**n
    if AMP < 0:
        AMP = 0.0
    # Additional delay time [s]
    if np.abs(x) < 1.57:
        T_iono = T * (5.0e-9 + AMP * (1.0 - x**2/2.0 + x**4/24.0))
    else:
        T_iono = T * 5.0e-9
    return T_iono


# global_relief_model = None
global_relief_interpolator = None
digital_elevation_model = None
geo_interpolator = None
geo_interpolator_type = None


def get_elevation(latitude, longitude, model='ETOPO1', geoid='egm96-5'):
    """Return coarse elevation for given coordinates on Earth surface.

    Use ETOPO1 Global Relief Model from
    https://www.ngdc.noaa.gov/mgg/global/global.html , which has a 1-arc-minute
    resolution or SRTM 1 Arc-Second Global from
    https://doi.org/10.5066/F7PR7TFT , which has a 1-arc-second resolution.

    Inputs:
        latitude - Latitude of receiver [deg], float or numpy.ndarray
        longitude - Longitude of receiver [deg], float or numpy.ndarray
        model - Relief / elevation model to use, 'ETOPO1' or 'SRTM1'
                [default='ETOPO1'], download SRTM1 models (.hgt files) from
                https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/
                and unpack into 'new_utilities/digital_elevation_models'
                directory
        geoid - Geoid type and grid to use, 'egm96-5' for EGM96 geoid with 5'
                grid [default='egm96-5'], download other geoids / resolutions
                from https://geographiclib.sourceforge.io/html/geoid.html#geoidinst
                and unpack into 'new_utilities' directory
    Output:
        elevation - Coarse elevation estimate for receiver position [m]
                    (GPS ellipsoidal height w.r.t. WGS84 ellipsoid, not w.r.t.
                     sea level), float or numpy.ndarray

    Author: Jonas Beuchert
    Reference:
        Amante, C. and B.W. Eakins, 2009. ETOPO1 1 Arc-Minute Global Relief
        Model: Procedures, Data Sources and Analysis. NOAA Technical Memorandum
        NESDIS NGDC-24. National Geophysical Data Center, NOAA.
        doi:10.7289/V5C8276M [2020-10-05].
    """
    import pygeodesy as pg
    import os

    global geo_interpolator
    global geo_interpolator_type

    if not geo_interpolator or geoid != geo_interpolator_type:
        # Load EGM96 geoid
        path = os.path.realpath(__file__)
        path = os.path.split(path)[0]
        geoid_file = geoid + '.pgm'
        path = os.path.join(path, geoid_file)
        try:
            geo_interpolator = pg.geoids.GeoidPGM(path)
            geo_interpolator_type = geoid
        except:
            raise ValueError("Geoid not found. Select 'egm96-5' or download "
                             + "desired geoid from "
                             + "https://geographiclib.sourceforge.io/html/"
                             + "geoid.html#geoidinst"
                             )

    # Get height of reference geoid (sea level)
    geoid_height = geo_interpolator.height(latitude, longitude)

    # if model == 'ETOPO1':

    #     import rockhound as rh

    #     global global_relief_model

    #     # Check if topography grid has not been loaded already
    #     if not global_relief_model:
    #         # Load a version of the topography grid
    #         global_relief_model = rh.fetch_etopo1(version="ice")

    #     if (isinstance(latitude, np.ndarray)
    #             and isinstance(longitude, np.ndarray)):
    #         # Vectorized version

    #         # Interpolate grid at desired coordinates (w.r.t. sea level)
    #         # Extract values from xarray.Dataset
    #         elevation = np.array([float(
    #             global_relief_model.interp(latitude=lat_i,
    #                                        longitude=lon_i,
    #                                        assume_sorted=True).ice
    #             ) for lat_i, lon_i in zip(latitude, longitude)])

    #         return elevation + geoid_height

    #     # Non-vectorized version

    #     # Interpolate grid at desired coordinates (w.r.t. sea level)
    #     elevation = global_relief_model.interp(latitude=latitude,
    #                                            longitude=longitude,
    #                                            assume_sorted=True)

    #     # Extract value from xarray.Dataset
    #     return float(elevation.ice) + geoid_height

    if model == 'ETOPO1':

        import rockhound as rh
        import scipy.interpolate as sip

        global global_relief_interpolator

        # Check if topography grid has not been loaded already
        if not global_relief_interpolator:
            # Load a version of the topography grid
            global_relief_model = rh.fetch_etopo1(version="ice")
            # Create (linear) interpolation function
            global_relief_interpolator = sip.RegularGridInterpolator(
                (global_relief_model.latitude.values,
                 global_relief_model.longitude.values),
                global_relief_model.ice.values)

        # Interpolate grid at desired coordinates (w.r.t. sea level)
        elevation = global_relief_interpolator(np.array([latitude,
                                                         longitude]).T)

    elif model == 'SRTM1':

        import srtm

        global digital_elevation_model

        # Check if topography grid has not been loaded already
        if not digital_elevation_model:
            # Load a version of the topography grid
            path = os.path.realpath(__file__)
            path = os.path.split(path)[0]
            path = os.path.join(path, "digital_elevation_models")
            digital_elevation_model = srtm.Srtm1HeightMapCollection(hgt_dir
                                                                    =path)

        if (not isinstance(latitude, np.ndarray)
                and not isinstance(longitude, np.ndarray)):
            # Non-vectorized version
            # Get height from model
            elevation = digital_elevation_model.get_altitude(latitude=latitude,
                                                             longitude=longitude)
        else:
            # Vectorized version
            # Get height from model
            elevation = np.array([
                digital_elevation_model.get_altitude(latitude=lat_i,
                                                     longitude=lon_i)
                for lat_i, lon_i in zip(latitude, longitude)
                ])

    else:

        raise Exception(
            "Chosen model not supported, select 'ETOPO1' or 'SRTM1'.")

    # Add elevation above sea level to sea level
    elevation = elevation + geoid_height

    if (not isinstance(latitude, np.ndarray)
            and not isinstance(longitude, np.ndarray)):
        # Non-vectorized version
        return float(elevation)
    # Vectorized version
    return elevation


def get_relative_height_from_pressure(measured_pressure,
                                      reference_pressure=101325.0,
                                      hypsometric=True, temperature=288.15):
    """Estimate height difference between 2 points from pressure difference.

    Inputs:
        measured_pressure - Observed pressure [Pa]
        reference_pressure - Pressure at reference location [Pa],
                             default=101325.0 (standard pressure at sea level)
        hypsometric - Flag if hypsometric equation shall be used or
                      simplified equation, default=True
        temperature - Observed temperature [K], only used if hypsometric=True,
                      default=288.15 (standard temperature)
    Output:
        h - Estimated height difference [m]

    Author: Jonas Beuchert
    """
    if not hypsometric:
        # Wikipedia
        return np.log(measured_pressure / reference_pressure) / (-0.00012)
    # Hypsometric formula
    return (
        np.power(reference_pressure / measured_pressure, 1 / 5.257) - 1
            ) * temperature / 0.0065
