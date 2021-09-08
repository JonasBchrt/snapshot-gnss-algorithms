# -*- coding: utf-8 -*-
"""Coarse-time navigation and coarse-time Doppler navigation.

Author: Jonas Beuchert
"""

try:
    import autograd.numpy as np
except(ImportError):
    print("""Package 'autograd' not found. 'autograd.numpy' is necessary for
          coarse-time navigation via maximum-likelihood estimation. Falling
          back to 'numpy'.""")
    import numpy as np
import eph_util as ep
import pymap3d as pm
import itertools as it
import jacobian_height as jh
import scipy.optimize as so
import scipy.stats as ss
from pseudorange_prediction import PseudorangePrediction
from timeout import timeout
import numpy as np2


def coarse_time_doppler_nav(X, d, t, eph, PRN):
    """Coarse-time Doppler navigation.

    Works for GPS or Galileo. If using both, concatenate navigation data
    (ephemeris) and make sure that satellite indices are unique.

    Implemented according to
    Fernández-Hernández, I., Borre, K. Snapshot positioning without initial
    information. GPS Solutions 20, 605–616 (2016).
    https://doi.org/10.1007/s10291-016-0530-4
    https://link.springer.com/article/10.1007/s10291-016-0530-4

    Inputs:
      X - Initial state:
          X[0:3] - Receiver position in ECEF XYZ coordinates
          X[3] - Receiver clock frequency drift
          X[4] - Time difference between initial time and actual time
      d - Measured Doppler shifts [Hz]
      t - Initial time (coarse time) [s]
      eph - ephemeris
      PRN - PRNs of visible (acquired) satellites

    Outputs:
      X - Estimated state
      dD - Residuals [m/s]

    Author: Jonas Beuchert
    """
    fL1 = 1575.42e6  # GPS signal frequency
    c = 299792458.0  # Speed of light [m/s]

    nIt = 50  # Number of iterations; TODO: Adaptive termination

    for k in range(nIt):

        # Coarse time + coarse-time error
        tow = np.mod(t + X[4], 7*24*60*60)  # Time of week [s]

        # Identify matching columns in ephemeris matrix, closest column in time
        # for each satellite
        col = np.array([ep.find_eph(eph, sat, tow) for sat in PRN])

        # Extract columns with available data
        d = d[~np.isnan(col)]
        PRN = PRN[~np.isnan(col)]
        col = col[~np.isnan(col)]
        eph = eph[:, col]

        # Satellite positions, velocities, and accelerations
        P, V, A = ep.get_sat_pos_vel_acc(tow, eph)

        e = P - X[:3]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)

        rho = np.linalg.norm(P - X[:3], axis=1, keepdims=True)

        # Range rate / satellite-to-receiver relative velocity
        rhoDot = np.sum(V * e, axis=1, keepdims=True)

        # Satellite-to-reciever relative acceleration
        rhoDotDot = np.sum(A * e, axis=1, keepdims=True)

        eDot = 1.0 / rho * (V - e * rhoDot)

        # Observation matrix
        H = np.hstack((eDot, np.ones((d.shape[0], 1)), -rhoDotDot))

        Dhat = -np.sum(e * V, axis=1) + X[3]  # Predict Doppler [m/s]
        D = d * c / fL1   # Doppler measurement [m/s]
        dD = D - Dhat  # Error

        dX = np.linalg.lstsq(H, dD, rcond=None)[0]  # Least-squares solution
        X = X + dX  # Update states

    return X, dD


def coarse_time_nav(obs, sats, Eph, TOW_assist_ms, rec_loc_assist, sort=True,
                    observed_height=None, inter_system_bias=False,
                    weights=None, hard_constraint=False, tropo='goad',
                    atm_pressure=1013.0, surf_temp=293.0, humidity=50.0,
                    iono=None, ion_alpha=np.array([]), ion_beta=np.array([]),
                    code_period_ms=1):
    """Compute receiver position using coarse-time navigation.

    Compute receiver position from fractional pseudoranges using coarse-time
    navigation and non-linear least-squares optimisation.
    The initial position should be within 100 - 150 km of the true position and
    the initial coarse time within about 1 min of the true time.
    Works for multiple GNSS, too, e.g., GPS and Galileo. If using more than
    one, then concatenate navigation data (ephemerides) and make sure that
    satellite indices are unique. E.g., use 1-32 for GPS and 201-250 for
    Galileo.

    Inputs:
        obs - Observations, the fractional pseudo-ranges (sub-millisecond).
        sats - SV numbers associated with each observation.
        Eph - Table of ephemerides, each column associated with a satellite.
        TOW_assist_ms - Coarse time of week [ms], single value or array for
                        different GNSS times with one value for each satellite.
        rec_loc_assist - Initial receiver position in ECEF XYZ coordinates.
        sort - Re-sort satellites according to distance [default=True].
        observed_height - Height observation [m], default=None
        inter_system_bias - Flag indicating if a bias between 2 GNSS is
                            added as optimisation variable assuming that all
                            satellites with PRN > 100 belong to the 2nd GNSS
                            [default=False]
        weights - Weight for each observation (height at the end, if present)
                  [default=None]
        hard_constraint - False: Use oberserved_height as additional
                          observation, i.e., as soft constraint. True: Use
                          observed_height as hard equality constraint.
                          [default=False]
        tropo - Model for troposheric correction: either None, 'goad' for the
                model of C. C. Goad and L. Goodman, 'hopfield' for the model of
                H. S. Hopfield, or 'tsui' for the model of J. B.-Y. Tsui
                [default='goad']
        atm_pressure - Atmospheric pressure at receiver location [mbar] for
                       troposheric correction, [default=1013.0]
        surf_temp - Surface temperature at receiver location [K] for
                    troposheric corrrection [default=293.0]
        humidity - Humidity at receiver location [%] for troposheric correction
                   [default=50.0]
        iono - Model for ionospheric correction: either None or 'klobuchar'
               for the model of J. Klobuchar [default=None]
        ion_alpha - Alpha parameters for Klobuchar model [default=np.array([])]
        ion_beta - Beta parameters for Klobuchar model [default=np.array([])]
        code_period_ms - Length of code [ms], either a single value for all
                         satellites or a numpy array with as many elements as
                         satellites [default=1]

    Outputs:
        state - ECEF XYZ position [m,m,m], common bias [m], coarse-time error
                [m]; np.NaN if optimization failed
        delta_z - Residuals (of pseudoranges) [m]

    Author: Jonas Beuchert
    """
    def assign_integers(sats, svInxListByDistance, obs, Eph, approx_distances,
                        code_period_ms=1):
        """Assign Ns according to van Diggelen's algorithm.

        Author: Jonas Beuchert
        """
        light_ms = 299792458.0 * 0.001
        N = np.zeros(sats.shape)
        approx_distances = approx_distances / light_ms  # Distances in millisec
        # Distances from ms to code periods
        approx_distances = approx_distances / code_period_ms

        unique_code_periods = np.unique(code_period_ms)
        n_different_code_periods = len(unique_code_periods)
        if not isinstance(code_period_ms, np.ndarray):
            N0_inx = svInxListByDistance[0]
        else:
            N0_inx = np.empty(n_different_code_periods)
            for c_idx in np.arange(n_different_code_periods):
                N0_inx[c_idx] = svInxListByDistance[
                    code_period_ms == unique_code_periods[c_idx]
                    ][0]
            N0_inx = N0_inx.astype(int)
        N[N0_inx] = np.floor(approx_distances[N0_inx])

        delta_t = Eph[18, :] * 1000.0  # From sec to millisec
        # Time errors from ms to code periods
        delta_t = delta_t / code_period_ms

        # Observed code phases from ms to code periods
        obs = obs / code_period_ms

        if not isinstance(code_period_ms, np.ndarray):
            for k in svInxListByDistance[1:]:
                N[k] = np.round(N[N0_inx] + obs[N0_inx] - obs[k] +
                                (approx_distances[k] - delta_t[k]) -
                                (approx_distances[N0_inx] - delta_t[N0_inx]))
        else:
            for c_idx in np.arange(n_different_code_periods):
                curr_svInxListByDistance = svInxListByDistance[
                    code_period_ms == unique_code_periods[c_idx]
                    ]
                for k in curr_svInxListByDistance[1:]:
                    N[k] = np.round(N[N0_inx[c_idx]] + obs[N0_inx[c_idx]]
                                    - obs[k] +
                                    (approx_distances[k] - delta_t[k]) -
                                    (approx_distances[N0_inx[c_idx]]
                                     - delta_t[N0_inx[c_idx]]))

        # Floored distances from code periods to ms
        N = N * code_period_ms

        return N, N0_inx

    # def othieno_assignNs(obs, eph, approx_distances, code_period_ms=1):
    #     """https://spectrum.library.concordia.ca/973909/1/Othieno_MASc_S2012.pdf"""
    #     # Approximate time errors
    #     delta_t = eph[18, :]
    #     # Speed of light
    #     c = 299792458.0
    #     # Predict pseudoranges
    #     predicted_pr = approx_distances / c / 1e-3 - delta_t / 1e-3
    #     # Roughly estimate Ns
    #     N = np.round(predicted_pr - obs)
    #     # Determine observed / time-free pseudoranges
    #     observed_pr = N + obs
    #     # Obtain difference between time-free pseudoranges and predcited ones
    #     diff = observed_pr - predicted_pr
    #     # Sort differences in ascending order
    #     min_idx = np.argmin(np.abs(diff))
    #     min_diff = diff[min_idx]
    #     # Compare all differences against 1st one
    #     diff_2_min = diff - min_diff
    #     # Adjust pseudoranges that differ by more than half a code period
    #     threshold = code_period_ms / 2.0
    #     observed_pr += (diff_2_min < -threshold) * threshold
    #     observed_pr -= (diff_2_min > threshold) * threshold
    #     return observed_pr * 1e-3 * c, \
    #         np.floor(observed_pr) - np.floor(observed_pr[min_idx])

    def tx_RAW2tx_GPS(tx_RAW, Eph):
        """Refactoring.

        Author: Jonas Beuchert
        """
        t0c = Eph[20]
        dt = ep.check_t(tx_RAW - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        dt = ep.check_t(tx_GPS - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        return tx_GPS, tcorr

    def e_r_corr(traveltime, X_sat):
        """Rotate satellite by earth rotation during signal travel time.

        Author: Jonas Beuchert
        """
        Omegae_dot = 7.292115147e-5  # rad/sec

        omegatau = Omegae_dot * traveltime
        R3 = np.array([np.array([np.cos(omegatau), np.sin(omegatau), 0.0]),
                       np.array([-np.sin(omegatau), np.cos(omegatau), 0.0]),
                       np.array([0.0, 0.0, 1.0])])
        return R3 @ X_sat

    no_iterations = 6
    v_light = 299792458.0
    dtr = np.pi / 180.0
    numSVs = obs.shape[0]  # Number of satellites
    el = np.zeros(numSVs)

    TOW_assist = TOW_assist_ms * 1e-3
    if not isinstance(TOW_assist, np.ndarray) or TOW_assist.shape == ():
        TOW_assist = TOW_assist * np.ones(sats.shape)

    # Identify ephemerides columns in Eph
    col_Eph = np.array([ep.find_eph(Eph, sats[k], TOW_assist[k])
                        for k in range(numSVs)])
    Eph = Eph[:, col_Eph]  # Sort according to sats argument

    # If one common bias shall be used for all systems, then all code phases
    # must have same range
    if isinstance(code_period_ms, np.ndarray) and not inter_system_bias:
        # Greatest common divider of code periods
        code_period_ms = np2.gcd.reduce(code_period_ms.astype(int))
        obs = np.mod(obs, code_period_ms)

    # Number of common-bias optimisation variables
    if isinstance(code_period_ms, np.ndarray):
        # Assume one entry in code_period_ms for each satellite
        # One bias for each present code period
        n_bias_opt = np.unique(code_period_ms).shape[0]
    elif inter_system_bias:
        # Same code period for all systems, nevertheless use inter-system bias
        n_bias_opt = 2
    else:
        # Same code period for all systems, do not use inter-system bias
        n_bias_opt = 1

    if hard_constraint:
        # Transform ECEF XYZ coordinates to geodetic coordinates
        latitude, longitude, _ = pm.ecef2geodetic(
            rec_loc_assist[0], rec_loc_assist[1], rec_loc_assist[2])
        initPosGeo = np.array([latitude, longitude, observed_height])
        # Initial receiver postion in EN coordinates
        state = np.zeros(2+n_bias_opt+1)
    else:
        # Preliminary guess for receiver position, common bias, and assistance
        # error ([x y z b et] or [x y z b1 b2 et])
        state = np.concatenate((rec_loc_assist, np.zeros(n_bias_opt+1)))

    T_tilde = TOW_assist

    # Find satellite positions at T_tilde
    tx_GPS = np.array([TOW_assist[k]
                       - ep.get_sat_clk_corr(TOW_assist[k],
                                             np.array([sats[k]]),
                                             Eph[:, k, np.newaxis])
                       for k in range(numSVs)])
    satPos_at_T_tilde = np.array([np.squeeze(
        ep.get_sat_pos(tx_GPS[k], Eph[:, k, np.newaxis]))
        for k in range(numSVs)])

    # And then find closest one
    approx_distances = np.sqrt(np.sum(
        (rec_loc_assist - satPos_at_T_tilde)**2, axis=-1))
    if sort:
        svInxListByDistance = np.argsort(approx_distances)
    else:
        svInxListByDistance = np.arange(numSVs)

    # if othieno:

    #     fullPRs, Ks = othieno_assignNs(
    #         obs, Eph, approx_distances, code_period_ms)
    #     Ks = Ks * 0

    # else:

    # Assign N numbers:
    Ns, N0_inx = assign_integers(sats, svInxListByDistance, obs, Eph,
                                 approx_distances, code_period_ms)

    # Now find K numbers:
    if isinstance(code_period_ms, np.ndarray):
        unique_code_periods = np.unique(code_period_ms)
        n_different_code_periods = len(unique_code_periods)
        Ks = np.empty(Ns.shape)
        for c_idx in np.arange(n_different_code_periods):
            Ks[code_period_ms == unique_code_periods[c_idx]] \
                = Ns[code_period_ms == unique_code_periods[c_idx]] \
                - Ns[N0_inx[c_idx]]
    else:
        Ks = Ns - Ns[N0_inx]

    fullPRs = Ns + obs  # Full pseudoranges reconstruction in ms
    fullPRs = fullPRs * (v_light * 1e-3)  # In meters

    for iter in range(no_iterations):
        if hard_constraint:
            H = np.empty((numSVs, 3+n_bias_opt))
        else:
            H = np.empty((numSVs, 4+n_bias_opt))
        delta_z = np.empty(numSVs)  # Observed minus computed observation

        # Coarse-time error
        Et = state[-1]  # In seconds

        for k in svInxListByDistance:

            # Common bias [m]
            if isinstance(code_period_ms, np.ndarray):
                code_period_idx = np.where(
                    unique_code_periods == code_period_ms[k])[0]
                b = state[-1-n_bias_opt+code_period_idx]
            elif inter_system_bias and sats[k] > 100:
                # Common bias of 2nd GNSS
                b = state[-2]
            else:
                # Common bias of 1st GNSS
                b = state[-1-n_bias_opt]

            Kk = Ks[k]

            tx_GPS, tcorr = tx_RAW2tx_GPS(T_tilde[k] - Et - Kk * 1e-3,
                                          Eph[:, k])

            X = ep.get_sat_pos(tx_GPS, Eph[:, k])
            X_fut = ep.get_sat_pos(tx_GPS + 1, Eph[:, k])
            satECEF = X
            sat_futECEF = X_fut
            if hard_constraint:
                # Transform ECEF XYZ coordinates to ENU coordinates
                X[0], X[1], X[2] = pm.ecef2enu(X[0], X[1], X[2], initPosGeo[0],
                                               initPosGeo[1], initPosGeo[2])
                X_fut[0], X_fut[1], X_fut[2] = pm.ecef2enu(X_fut[0], X_fut[1],
                                                           X_fut[2],
                                                           initPosGeo[0],
                                                           initPosGeo[1],
                                                           initPosGeo[2])
                state_memory = state
                state = np.array([state[0], state[1], 0.0])

            # This if case calculates something about trop
            if iter == 0:
                traveltime = 0.072
                Rot_X = satECEF
                Rot_X_fut = sat_futECEF
                trop = 0.0
            else:
                if hard_constraint:
                    posECEF = np.empty(3)
                    posECEF[0], posECEF[1], posECEF[2] \
                        = pm.enu2ecef(state[0], state[1], 0.0,
                                      initPosGeo[0], initPosGeo[1],
                                      initPosGeo[2])
                else:
                    posECEF = state[:3]

                rho2 = (satECEF[0] - posECEF[0])**2 \
                    + (satECEF[1] - posECEF[1])**2 \
                    + (satECEF[2] - posECEF[2])**2  # Distance squared
                traveltime = np.sqrt(rho2) / v_light
                Rot_X = e_r_corr(traveltime, satECEF)

                Rot_X_fut = e_r_corr(traveltime, sat_futECEF)

                rho2 = (Rot_X[0] - posECEF[0])**2 \
                    + (Rot_X[1] - posECEF[1])**2 \
                    + (Rot_X[2] - posECEF[2])**2

                if tropo == 'goad':
                    az, el, dist = ep.topocent(posECEF, Rot_X-posECEF)
                    trop = ep.tropo(np.sin(el * dtr), 0.0, atm_pressure,
                                    surf_temp, humidity, 0.0, 0.0, 0.0)
                elif tropo == 'hopfield':
                    surf_temp_celsius = surf_temp-273.15
                    saturation_vapor_pressure = 6.11*10.0**(
                        7.5*surf_temp_celsius/(237.7+surf_temp_celsius))
                    vapor_pressure = humidity/100.0 * saturation_vapor_pressure
                    trop = ep.tropospheric_hopfield(
                        posECEF, np.array([Rot_X]), surf_temp_celsius,
                        atm_pressure/10.0, vapor_pressure/10.0)
                elif tropo == 'tsui':
                    lat, lon, h = pm.ecef2geodetic(posECEF[0], posECEF[1],
                                                   posECEF[2])
                    az, el, srange = pm.ecef2aer(Rot_X[0], Rot_X[1], Rot_X[2],
                                                 lat, lon, h)
                    trop = ep.tropospheric_tsui(el)
                else:
                    trop = 0.0

                if iono == 'klobuchar':
                    trop = trop + ep.ionospheric_klobuchar(
                        posECEF, np.array([Rot_X]),
                        np.mod(TOW_assist[k], 24*60*60),
                        ion_alpha, ion_beta) * v_light
                elif iono == 'tsui':
                    lat, lon, h = pm.ecef2geodetic(posECEF[0], posECEF[1],
                                                   posECEF[2])
                    az, el, srange = pm.ecef2aer(Rot_X[0], Rot_X[1], Rot_X[2],
                                                 lat, lon, h)
                    # Convert degrees to semicircles
                    el = el / 180.0
                    az = az / 180.0
                    lat = lat / 180.0
                    lon = lon / 180.0
                    # Ionospheric delay [s]
                    T_iono = ep.ionospheric_tsui(
                        el, az, lat, lon, TOW_assist[k], ion_alpha, ion_beta)
                    trop = trop + T_iono * v_light

            # Subtraction of state[3] corrects for receiver clock offset and
            # v_light*tcorr is the satellite clock offset
            predictedPR = np.linalg.norm(Rot_X - state[:3]) + b \
                - tcorr * v_light + trop  # meters
            delta_z[k] = fullPRs[k] - predictedPR  # Meters

            # Now add row to matrix H according to:
            # -e_k 1 v_k
            # Notice that it is easier to plug in the location of the satellite
            # at its T_dot estimation, i.e., Rot_X
            sat_vel_mps = Rot_X_fut - Rot_X

            e_k = (Rot_X - (Et + Kk * 1e-3) * sat_vel_mps - state[:3])
            e_k = e_k / np.linalg.norm(e_k)

            v_k = np.sum(-sat_vel_mps * e_k, keepdims=True)  # Relative speed

            if hard_constraint:
                # Optimise only over E and N coordinate
                e_k = e_k[:2]
                # Restore state
                state = state_memory

            if isinstance(code_period_ms, np.ndarray):
                code_period_idx = np.where(
                    unique_code_periods == code_period_ms[k])
                jac_common_bias = np.zeros(n_bias_opt)
                jac_common_bias[code_period_idx] = 1.0
                # Jacobian w.r.t. to common bias of this GNSS
                H_row = np.concatenate((-e_k, jac_common_bias,
                                        v_k))
            elif not inter_system_bias:
                # Matrix for 5 optimisation variables
                H_row = np.concatenate((-e_k, np.ones(1), v_k))
            else:
                # Matrix for 6 optimisation variables
                if sats[k] <= 100:
                    # Jacobian w.r.t. to common bias of 1st GNSS
                    H_row = np.concatenate((-e_k, np.ones(1), np.zeros(1),
                                            v_k))
                else:
                    # Jacobian w.r.t. to common bias of 2nd GNSS
                    H_row = np.concatenate((-e_k, np.zeros(1), np.ones(1),
                                            v_k))
            # Append Jacobian to end of matrix
            H[k] = H_row

        # Check if height measurement is provided
        if observed_height is not None and not hard_constraint:
            # Add Jacobian of height observation
            H = np.vstack((H, jh.jacobian_height(state)))
            # Predict height based on current state
            predicted_height = pm.ecef2geodetic(state[0], state[1], state[2]
                                                )[2]
            # Add height measurement
            delta_z = np.append(delta_z, observed_height - predicted_height)

        if weights is not None:
            H = H * np.sqrt(weights[:, np.newaxis])
            delta_z = delta_z * np.sqrt(weights)

        x = np.linalg.lstsq(H, delta_z, rcond=None)[0]
        state = state + x

    if hard_constraint:
        # Convert ENU to ECEF XYZ coordinates
        [pos_x, pos_y, pos_z] = pm.enu2ecef(state[0], state[1], 0.0,
                                            initPosGeo[0], initPosGeo[1],
                                            initPosGeo[2])
        state = np.concatenate((np.array([pos_x, pos_y, pos_z]), state[2:]))

    return state, delta_z[svInxListByDistance]


def coarse_time_nav_simplified(obs, sats, Eph, TOW_assist_ms, rec_loc_assist, sort=True,
                    observed_height=None, inter_system_bias=False,
                    weights=None, hard_constraint=False, tropo='goad',
                    atm_pressure=1013.0, surf_temp=293.0, humidity=50.0,
                    iono=None, ion_alpha=np.array([]), ion_beta=np.array([]),
                    code_period_ms=1, hdop=False, no_iterations=6):
    """Compute receiver position using coarse-time navigation.

    Compute receiver position from fractional pseudoranges using coarse-time
    navigation and non-linear least-squares optimisation.
    The initial position should be within 100 - 150 km of the true position and
    the initial coarse time within about 1 min of the true time.
    Works for multiple GNSS, too, e.g., GPS and Galileo. If using more than
    one, then concatenate navigation data (ephemerides) and make sure that
    satellite indices are unique. E.g., use 1-32 for GPS and 201-250 for
    Galileo.

    Inputs:
        obs - Observations, the fractional pseudo-ranges (sub-millisecond).
        sats - SV numbers associated with each observation.
        Eph - Table of ephemerides, each column associated with a satellite.
        TOW_assist_ms - Coarse time of week [ms], single value or array for
                        different GNSS times with one value for each satellite.
        rec_loc_assist - Initial receiver position in ECEF XYZ coordinates.
        sort - Obsolete.
        observed_height - Height observation [m], default=None
        inter_system_bias - Obsolete.
        weights - Obsolete.
        hard_constraint - Obsolete.
        tropo - Obsolete.
        atm_pressure - Obsolete.
        surf_temp - Obsolete.
        humidity - Obsolete.
        iono - Obsolete.
        ion_alpha - Obsolete.
        ion_beta - Obsolete.
        code_period_ms - Length of code [ms], either a single value for all
                         satellites or a numpy array with as many elements as
                         satellites [default=1]
        hdop - Flag if horizontal dilution of precision is returned as 3rd
               output, default=False
        no_iterations - Number of non-linear least-squares iterations,
                        default=6

    Outputs:
        state - ECEF XYZ position [m,m,m], common bias [m], coarse-time error
                [m]; np.NaN if optimization failed
        delta_z - Residuals (of pseudoranges) [m]
        hdop - (Only if hdop=True) Horizontal dilution of precision

    Author: Jonas Beuchert
    """
    def assign_integers(sats, svInxListByDistance, obs, Eph, approx_distances,
                        code_period_ms=1):
        """Assign Ns according to van Diggelen's algorithm.

        Author: Jonas Beuchert
        """
        light_ms = 299792458.0 * 0.001
        N = np.zeros(sats.shape)
        approx_distances = approx_distances / light_ms  # Distances in millisec
        # Distances from ms to code periods
        approx_distances = approx_distances / code_period_ms

        # To integer
        N[0] = np.floor(approx_distances[0])

        delta_t = Eph[18, :] * 1000.0  # From sec to millisec
        # Time errors from ms to code periods
        delta_t = delta_t / code_period_ms

        # Observed code phases from ms to code periods
        obs = obs / code_period_ms

        N[1:] = np.round(N[0] + obs[0] - obs[1:] +
                         (approx_distances[1:] - delta_t[1:]) -
                         (approx_distances[0] - delta_t[0]))

        # Floored distances from code periods to ms
        N = N * code_period_ms

        return N

    def tx_RAW2tx_GPS(tx_RAW, Eph):
        """Refactoring.

        Author: Jonas Beuchert
        """
        t0c = Eph[20]
        dt = ep.check_t_vectorized(tx_RAW - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        dt = ep.check_t_vectorized(tx_GPS - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        return tx_GPS, tcorr

    def e_r_corr(traveltime, sat_pos, n_sats):
        """Rotate satellite by earth rotation during signal travel time.

        Author: Jonas Beuchert
        """
        Omegae_dot = 7.292115147e-5  # rad/sec

        omegatau = Omegae_dot * traveltime
        # Vector of rotation matrices
        R3 = np.transpose(np.array([
            np.array([np.cos(omegatau), np.sin(omegatau), np.zeros(n_sats)]),
            np.array([-np.sin(omegatau), np.cos(omegatau), np.zeros(n_sats)]),
            np.array([np.zeros(n_sats), np.zeros(n_sats),  np.ones(n_sats)])]),
            axes=(2, 0, 1))
        # Turn satellite positions into vector of column vectors
        sat_pos = np.array([sat_pos]).transpose(1, 2, 0)
        return np.matmul(R3, sat_pos).reshape(n_sats, 3)

    v_light = 299792458.0
    numSVs = obs.shape[0]  # Number of satellites

    TOW_assist = TOW_assist_ms * 1e-3
    if not isinstance(TOW_assist, np.ndarray) or TOW_assist.shape == ():
        TOW_assist = TOW_assist * np.ones(sats.shape)

    if Eph.shape[1] != sats.shape[0] or np.any(Eph[0] != sats):
        # Identify ephemerides columns in Eph
        col_Eph = np.array([ep.find_eph(Eph, sats[k], TOW_assist[k])
                            for k in range(numSVs)])
        Eph = Eph[:, col_Eph]  # Sort according to sats argument

    # If one common bias shall be used for all systems, then all code phases
    # must have same range
    if isinstance(code_period_ms, np.ndarray):
        # Greatest common divider of code periods
        code_period_ms = np2.gcd.reduce(code_period_ms.astype(int))
    obs = np.mod(obs, code_period_ms)

    # Number of common-bias optimisation variables
    # Same code period for all systems, do not use inter-system bias
    n_bias_opt = 1

    # Preliminary guess for receiver position, common bias, and assistance
    # error ([x y z b et] or [x y z b1 b2 et])
    state = np.concatenate((rec_loc_assist, np.zeros(n_bias_opt+1)))

    T_tilde = TOW_assist

    # Find satellite positions at T_tilde
    tx_GPS = TOW_assist - ep.get_sat_clk_corr_vectorized(TOW_assist, sats, Eph)
    satPos_at_T_tilde = ep.get_sat_pos(tx_GPS, Eph)

    # And then find closest one
    approx_distances = np.sqrt(np.sum(
        (rec_loc_assist - satPos_at_T_tilde)**2, axis=-1))

    sat_idx = np.arange(numSVs)

    # Assign N numbers:
    Ns = assign_integers(sats, sat_idx, obs, Eph,
                         approx_distances, code_period_ms)

    Ks = Ns - Ns[0]

    fullPRs = Ns + obs  # Full pseudoranges reconstruction in ms
    fullPRs = fullPRs * (v_light * 1e-3)  # In meters

    # Intialization
    H = np.empty((numSVs, 4+n_bias_opt))  # Linear system matrix
    delta_z = np.empty(numSVs)  # Observed minus computed observation

    for iteration_idx in np.arange(no_iterations):

        # Coarse-time error
        Et = state[-1]  # In seconds

        # Common bias [m]
        b = state[-1-n_bias_opt]

        tx_GPS, tcorr = tx_RAW2tx_GPS(T_tilde - Et - Ks * 1e-3, Eph)

        sat_pos, sat_vel = ep.get_sat_pos_vel(tx_GPS, Eph)

        if iteration_idx == 0:
            traveltime = 0.072
            rot_sat_pos = sat_pos
        else:
            posECEF = state[:3]
            rho2 = (sat_pos[:, 0] - posECEF[0])**2 \
                + (sat_pos[:, 1] - posECEF[1])**2 \
                + (sat_pos[:, 2] - posECEF[2])**2  # Distance squared
            traveltime = np.sqrt(rho2) / v_light
            rot_sat_pos = e_r_corr(traveltime, sat_pos, numSVs)

        # Subtraction of state[3] corrects for receiver clock offset and
        # v_light*tcorr is the satellite clock offset
        predictedPR = np.linalg.norm(rot_sat_pos - state[:3], axis=-1) \
            + b - tcorr * v_light  # meters
        delta_z = fullPRs - predictedPR  # Meters

        # Receiver-satellite vector
        e_k = (rot_sat_pos
               - np.tile(np.array([Et + Ks * 1e-3]).T, (1, 3)) * sat_vel
               - state[:3])
        # Normalize receiver-satellite vector
        e_k = e_k / np.array([np.linalg.norm(e_k, axis=-1)]).T

        # Relative satellite velocity along this vector, i.e.,
        # project satellite velocity on normalized receiver-satellite vector
        v_k = np.sum(-sat_vel * e_k, axis=1, keepdims=True)

        # Now build Jacobian matrix H according to:
        # -e_k 1 v_k
        # Notice that it is easier to plug in the location of the satellite
        # at its T_dot estimation, i.e., rot_sat_pos
        # Matrix for 5 optimisation variables
        H = np.hstack((-e_k, np.ones((numSVs, 1)), v_k))

        # Check if height measurement is provided
        if observed_height is not None:
            # Add Jacobian of height observation
            H = np.vstack((H, jh.jacobian_height(state)))
            # Predict height based on current state
            predicted_height = pm.ecef2geodetic(state[0], state[1], state[2]
                                                )[2]
            # Add height measurement
            delta_z = np.append(delta_z, observed_height - predicted_height)

        x = np.linalg.lstsq(H, delta_z, rcond=None)[0]
        state = state + x

    if hdop:
        try:
            Q = np.linalg.inv(H.T @ H)
            dilution_east_squared = Q[0, 0]
            dilution_north_squared = Q[1, 1]
            hdop = np.sqrt(dilution_east_squared + dilution_north_squared)
            return state, delta_z, hdop
        except:
            print("Cannot calculate HDOP.")
            return state, delta_z, np.nan

    return state, delta_z


def cold_snapshot(c, d, t_min, t_max, eph, PRN, init_pos=np.zeros(3)):
    """Cold snapshot algorithm.

    Coarse-time Doppler navigation combined with coarse-time navigation.
    Works for GPS or Galileo with code-phases in the intervall 0 - 1 ms. If
    using both, concatenate navigation data (ephemeris) and make sure that
    satellite indices are unique.

    Inputs:
      c - Measured code phases [ms]
      d - Measured Doppler shifts [Hz]
      t_min - Start of time interval (datetime or GPS time [s])
      t_max - End of time interval (datetime or GPS time [s])
      eph - ephemeris
      PRN - PRNs of visible (acquired) satellites
      init_pos - [Optional] initial psoition in ECEF XYZ coordinates [m,m,m]
    Outputs:
      X_opt - Estimated state
      res_opt - Pseudorange residuals [m]
      t_opt - Estimated coarse time (GPS time) [s]; subtract X_opt[4] for more
              precise time estimate; X_opt[4] is the coarse time error

    Implemented according to
    Fernández-Hernández, I., Borre, K. Snapshot positioning without initial
    information. GPS Solut 20, 605–616 (2016)
    https://doi.org/10.1007/s10291-016-0530-4
    https://link.springer.com/article/10.1007/s10291-016-0530-4

    Author: Jonas Beuchert
    """
    dt = 3.0 * 60.0 * 60.0  # Step size for iterations [s]

    if isinstance(t_min, np.datetime64):
        t_min = ep.utc_2_gps_time(t_min)
    if isinstance(t_max, np.datetime64):
        t_max = ep.utc_2_gps_time(t_max)

    T = np.arange(t_min, t_max, dt)  # Coarse time grid
    if T[-1] < t_max:
        np.append(T, t_max)

    X0 = np.zeros(5)  # Initialize
    X0[:3] = init_pos  # Set initial position

    dDmin = np.inf

    # Intialize outputs
    Xopt = X0
    resOpt = np.full(5, np.inf)
    tOpt = np.mean(np.array([t_min, t_max]))

    for t in T:  # Calculate one solution every 3 h
        X, dD = coarse_time_doppler_nav(X0, d, t, eph, PRN)
        # Convert code phases from s to ms
        code_phases = c / 1e-3
        # Time of week in ms incl. correction from coarse time Doppler nav.
        initTimeMs = np.mod(t + X[4], 7 * 24 * 60 * 60) / 1e-3
        # Get initial position from coarse time Doppler navigation
        init_pos = X[:3]
        # Coarse-time navigation
        lsEstimate, resPseudoranges = coarse_time_nav(code_phases, PRN, eph,
                                                      initTimeMs, init_pos)
        # If norm of pseudorange residuals is smaller than current optimum
        if np.linalg.norm(resPseudoranges) < dDmin:
            # Set new optimum
            dDmin = np.linalg.norm(resPseudoranges)
            # Store state as best solution
            Xopt = lsEstimate
            # Store residuals as smallest residuals
            resOpt = resPseudoranges
            # Store coarse time
            tOpt = t + X[4]

        return Xopt, resOpt, tOpt


def warm_start_acquisition(data, interm_freq, sampling_freq, init_pos,
                           init_bias, init_time, eph, ms_to_process=1,
                           elev_mask=10, pos_err_max=1.414248917270224e+04,
                           bias_err_max=1.5e4, time_err_max=1, gnss='gps',
                           ref=True, corr=False):
    """Warm-started acquisition and code phase estimation.

    Perform acquisition and keep only satellites whose code phase estimates
    are close to the predicted values based on an initial position, common
    bias, and time.

    Inputs:
        data - GNSS snapshot
        interm_freq - Intermediate frequency [Hz]
        sampling_freq - Sampling freqeuncy [Hz]
        init_pos - Initial receiver position in ECEF XYZ [m,m,m]
        init_bias - Initial common pseudorange bias [m]
        init_time - Initial coarse time [s]
        eph - Ephemeris matrix
        ms_to_process - Number of milliseconds of the snapshot to use
        elev_mask - Elevation mask for predicting satellites [deg]
        pos_err_max - Maximum expected horizontal receiver position error
                      w.r.t. initial position [m]
        bias_err_max - Maximum expected common bias deviation w.r.t. initial
                       bias [m]
        time_err_max - Maximum expected receiver clock deviation w.r.t. initial
                       coarse time [s]
        gnss - Type of GNSS, 'gps' or 'galileo', default='gps'
        ref - Switch for type of code-phase prediction, True for algorithm
              according to van Diggelen based on one reference satellite, False
              for algorithm according to Bissig et al. with two iterations for
              each satellite independently, default=True
        corr - Switch for atmospheric correction when prediciting code phases,
               default=False

    Outputs:
        PRN - Indices of satellites whose estimated code phases are close to
              the predicted values
        code_phases - Estimated code phases that are close to the predicted
                      values [ms]
        peak_metrics - Estimated signal-to-noise ratios (SNR) of satellites
                       whose estimated code phases are close to the predicted
                       values [dB]
        code_phase_err - Absolute differences between estimated and predicted
                         code phases of returned satellites [ms]

    Author: Jonas Beuchert
    """
    if gnss == 'gps':
        code_duration = 1e-3
    elif gnss == 'galileo':
        code_duration = 4e-3
    else:
        raise ValueError(
            "Chosen GNSS not supported, select 'gps' or 'galileo'.")
    # Maximum expected satellite speed relative to receiver [m/s]
    sat_vel_max = 14.0e3 / 60.0 / 60.0
    # Speed of light [m/s]
    c = 299792458.0
    # Convert initial position from Cartesian to geodetic coordinates
    init_pos_geo = np.empty(3)
    init_pos_geo[0], init_pos_geo[1], init_pos_geo[2] = pm.ecef2geodetic(
        init_pos[0], init_pos[1], init_pos[2])
    # Predict set of visible satellites
    acq_satellite_list = ep.get_visible_sats(init_time, init_pos_geo, eph,
                                             elev_mask)
    # Calculate maximum code phase estimation error [ms]
    code_phase_err_max = (time_err_max * sat_vel_max + pos_err_max
                          + bias_err_max) / c / 1e-3
    if ref:
        # Predict pseudoranges
        pr = ep.predict_pseudoranges(acq_satellite_list, eph, init_time,
                                     init_pos, init_bias, corr)
        # Convert predicted pseudoranges to code phases in milliseconds
        init_code_phases = np.mod(pr / c, code_duration) / 1e-3
    else:
        # Predict code phases
        init_code_phases = ep.get_code_phase(init_time, init_pos, init_bias,
                                             acq_satellite_list, eph,
                                             code_duration, corr) / 1e-3
    # Estimate code phases
    acquired_sv, acquired_snr, acquired_doppler, acquired_codedelay,\
        acquired_fine_freq, results_doppler, results_code_phase,\
        results_peak_metric = ep.acquisition(data, interm_freq, sampling_freq,
                                             ms_to_process=ms_to_process,
                                             prn_list=acq_satellite_list,
                                             fine_freq=False, gnss=gnss)
    # Convert code phases from number of samples to ms
    code_phases = results_code_phase[acq_satellite_list - 1] / sampling_freq \
        / 1e-3
    # Calculate difference between extimated and observed code phases
    code_phase_err = np.abs(code_phases - init_code_phases)
    code_phase_err = np.min(np.array([code_phase_err, 1.0 - code_phase_err]),
                            axis=0)
    # Keep only satellites where difference is small
    PRN = acq_satellite_list[code_phase_err < code_phase_err_max]
    code_phases = results_code_phase[PRN - 1] / sampling_freq / 1e-3
    peak_metrics = results_peak_metric[PRN - 1]
    code_phase_err = code_phase_err[code_phase_err < code_phase_err_max]
    return PRN, code_phases, peak_metrics, code_phase_err


def selective_coarse_time_nav(PRN, code_phases, peak_metrics, eph, init_pos,
                              init_time, init_height=0, height_err_max=200,
                              sort_key='elevation', code_phase_err=None,
                              mode='SNR', return_prn=False,
                              observed_height=None, inter_system_bias=False,
                              weights=None, hard_constraint=False,
                              tropo='goad', atm_pressure=1013.0,
                              surf_temp=293.0, humidity=50.0, iono='none',
                              ion_alpha=np.array([]), ion_beta=np.array([]),
                              code_period_ms=1,
                              max_dist=None, max_time=None,
                              max_residual=100):
    """Coarse-time navigation (CTN) with iterative solution checking.

    Coarse-time navigation (CTN) with iterative solution checking and
    discarding of unreliable satellites.
    The initial position should be within 100 - 150 km of the true position and
    the initial coarse time within about 1 min of the true time.

    Inputs:
        PRN - Indices of acquired satellites
        code_phases - Code phases of acquired satellites [ms], 0-1
        peak_metrics - Realiability metrics for acquired satellites, e.g.,
                       signal-to-noise ratios (SNR)
        init_pos - Initial receiver position in ECEF XYZ [m,m,m]
        init_time - Initial coarse time [s], single value or array for
                    different GNSS times with one value for each satellite
        eph - Ephemeris matrix
        init_height - Expected height above sea level (WGS84) [m], default=0,
                      None to disable height check
        height_err_max - Maximum expected deviation from expected height [m],
                         default=200, None to disable
        sort_key - Property for sorting satellites according to their
                   reliability:
                       'SNR' - Signal-to-noise ratio
                       'error' - Error between predicted & estimated code phase
                       'elevation' - Satellite elevation (default)
                       'distance' - Satellite-receiver distance
        code_phase_err - Absolute differences between estimated and predicted
                         code phases of returned satellites [ms], only required
                         if sort_key='error'
        mode - Method for satellite selection:
                   'SNR' - Rank satellite according to peak metrics and
                           eliminate weakest satellite first (default)
                   'combinatoric' - Test all combinations of satellites
                                    starting with as many as possible
        return_prn - Flag idicating if satellite indices that were used to
                     calculate the solution are returned as 2nd output
        observed_height - Height observation [m], default=None
        inter_system_bias - Flag indicating if a bias between 2 GNSS is
                            added as optimisation variable, default=False
        weights - Weight for each observation (height at the end, if present)
                  [default=None]
        hard_constraint - False: Use oberserved_height as additional
                          observation, i.e., as soft constraint. True: Use
                          observed_height as hard equality constraint.
                          [default=False]
        tropo - Model for troposheric correction: either 'none', 'goad' for the
                model of C. C. Goad and L. Goodman, 'hopfield' for the model of
                H. S. Hopfield, or 'tsui' for the model of J. B.-Y. Tsui
                [default='goad']
        atm_pressure - Atmospheric pressure at receiver location [mbar] for
                       troposheric correction, [default=1013.0]
        surf_temp - Surface temperature at receiver location [K] for
                    troposheric corrrection [default=293.0]
        humidity - Humidity at receiver location [%] for troposheric correction
                   [default=50.0]
        iono - Model for ionospheric correction: either 'none' or 'klobuchar'
               for the model of J. Klobuchar [default='none']
        ion_alpha - Alpha parameters for Klobuchar model [default=np.array([])]
        ion_beta - Beta parameters for Klobuchar model [default=np.array([])]
        code_period_ms - Length of code [ms], either a single value for all
                         satellites or a numpy array with as many elements as
                         satellites [default=1]
        max_dist - Maximum spatial distance between 2 consecutive fixes to be
                   plausible [m], default=None
        max_time - Maximum temporal distance between 2 consecutive fixes to be
                   plausible [s], i.e., the change of the coarse-time error,
                   default=None
        max_residual - Maximum pseudorange residual to be plausible [m],
                       default=100

    Output:
        state - Receiver ECEF position, common bias [m], coarse-time error [s];
                np.inf for all values if no plausible was found
        PRN - Satellite indices that were used to calculate the solution, only
              returned if return_prn=True

    Author: Jonas Beuchert
    """
    # Convert initial position from Cartesian to geodetic coordinates
    init_pos_geo = np.empty(3)
    init_pos_geo[0], init_pos_geo[1], init_pos_geo[2] = pm.ecef2geodetic(
        init_pos[0], init_pos[1], init_pos[2])
    # Get time of week in milliseconds
    init_time_ms = np.mod(init_time, 7 * 24 * 60 * 60) / 1e-3
    # Sort satellites
    eph_match = eph
    ind = np.arange(PRN.shape[0])
    if sort_key == 'SNR':
        # Sort satellites according to signal-to-noise ratio
        ind = np.argsort(peak_metrics)[::-1]
    elif sort_key == 'error':
        # Sort satellites according to prediction error
        ind = np.argsort(code_phase_err)
    elif sort_key == 'elevation' or sort_key == 'distance':
        if isinstance(init_time, np.ndarray) and init_time.shape != ():
            init_time = init_time[0]
        # GPS time since 1980 to time of week (TOW) [s]
        coarse_time_tow = np.mod(init_time, 7 * 24 * 60 * 60)
        # Identify matching columns in ephemeris matrix, closest column in time
        # for each satellite
        col = np.array([ep.find_eph(eph, sat, coarse_time_tow) for sat in PRN])
        # Extract columns with available data
        eph_match = eph[:, col]
        # Find satellite positions at coarse transmission time
        tx_gps = coarse_time_tow - ep.get_sat_clk_corr(coarse_time_tow, PRN,
                                                       eph_match)
        sat_pos_coarse = ep.get_sat_pos(tx_gps, eph_match)
        if sort_key == 'elevation':
            # Calculate satellite elevation
            az, elev, rng = pm.ecef2aer(sat_pos_coarse[:, 0],
                                        sat_pos_coarse[:, 1],
                                        sat_pos_coarse[:, 2], init_pos_geo[0],
                                        init_pos_geo[1], init_pos_geo[2])
            # Sort satellites according to elevation
            ind = np.argsort(elev)[::-1]
        elif sort_key == 'distance':
            # Calculate distances to satellites
            distances_coarse = np.sqrt(np.sum((init_pos - sat_pos_coarse)**2,
                                              axis=-1))
            # Sort satellites by distance
            ind = np.argsort(distances_coarse)
    peak_metrics = peak_metrics[ind]
    PRN = PRN[ind]
    code_phases = code_phases[ind]
    if isinstance(code_period_ms, np.ndarray):
        code_period_ms = code_period_ms[ind]
    if isinstance(init_time_ms, np.ndarray):
        init_time_ms = init_time_ms[ind]
    if mode == 'SNR':
        # Iterate until plausible solution is found
        plausible_solution = False
        while not plausible_solution and PRN.shape[0] > 1:
            # Coarse-time navigation
            pos, res = coarse_time_nav(code_phases, PRN, eph_match,
                                       init_time_ms, init_pos, sort=False,
                                       observed_height=observed_height,
                                       inter_system_bias=inter_system_bias,
                                       weights=weights,
                                       hard_constraint=hard_constraint,
                                       tropo=tropo, atm_pressure=atm_pressure,
                                       surf_temp=surf_temp, humidity=humidity,
                                       iono=iono,
                                       ion_alpha=ion_alpha, ion_beta=ion_beta,
                                       code_period_ms=code_period_ms)
            # Check plausibility of solution by checking pseudorange
            # residuals and altitude (height) or distance to intialization
            if height_err_max is not None and init_height is not None:
                lat, lon, h = pm.ecef2geodetic(pos[0], pos[1], pos[2])
                h_ref = init_height  # ep.get_elevation(lat, lon)
                height_ok = h > h_ref - height_err_max \
                    and h < h_ref + height_err_max
            else:
                height_ok = True
            if max_dist is not None:
                dist_ok = np.linalg.norm(pos[:3] - init_pos) < max_dist
            else:
                dist_ok = True
            if max_time is not None:
                time_ok = np.abs(pos[-1]) < max_time
            else:
                time_ok = True
            if max_residual is not None:
                residuals_ok = np.all(np.abs(res) <= max_residual)
            else:
                residuals_ok = True
            if residuals_ok and height_ok and dist_ok and time_ok:
                plausible_solution = True
            else:
                # Get code phases only for satellites with high peak metric
                ind = np.argmin(peak_metrics)
                PRN = np.delete(PRN, ind)
                code_phases = np.delete(code_phases, ind)
                peak_metrics = np.delete(peak_metrics, ind)
                if weights is not None:
                    weights = np.delete(weights, ind)
                if isinstance(code_period_ms, np.ndarray):
                    code_period_ms = np.delete(code_period_ms, ind)
                if isinstance(init_time_ms, np.ndarray):
                    init_time_ms = np.delete(init_time_ms, ind)
        if plausible_solution:
            if return_prn:
                return pos, PRN
            else:
                return pos
        else:
            if return_prn:
                return np.full(5, np.inf), np.array([])
            else:
                return np.full(5, np.inf)
    elif mode == 'combinatoric':
        # Start elimination with maximum number of available satellites
        for n_sats_used in range(PRN.shape[0], 0, -1):
            # Get all possible satellite combinations with
            # n_sats_used satellites
            sat_combinations = it.combinations(range(PRN.shape[0]),
                                               n_sats_used)
            # Iterate over all rows of sat_combiantions, i.e., all combinations
            # with n_sat_used satellites
            for chosen_sats in sat_combinations:
                if isinstance(code_period_ms, np.ndarray):
                    code_period_ms_current \
                        = code_period_ms[np.asarray(chosen_sats)]
                else:
                    code_period_ms_current = code_period_ms
                if isinstance(init_time_ms, np.ndarray):
                    init_time_ms_current \
                        = init_time_ms[np.asarray(chosen_sats)]
                else:
                    init_time_ms_current = init_time_ms
                # Coarse-time navigation with chosen satellites
                pos, res = coarse_time_nav(
                    code_phases[np.asarray(chosen_sats)],
                    PRN[np.asarray(chosen_sats)], eph_match,
                    init_time_ms_current,
                    init_pos, sort=False, observed_height=observed_height,
                    inter_system_bias=inter_system_bias,
                    weights=None if weights is None else weights[
                        np.asarray(chosen_sats)],
                    hard_constraint=hard_constraint,
                    tropo=tropo, atm_pressure=atm_pressure,
                    surf_temp=surf_temp, humidity=humidity, iono=iono,
                    ion_alpha=ion_alpha, ion_beta=ion_beta,
                    code_period_ms=code_period_ms_current)
                # Check plausibility of solution by checking pseudorange
                # residuals and altitude (height) or distance to intialization
                if height_err_max is not None and init_height is not None:
                    lat, lon, h = pm.ecef2geodetic(pos[0], pos[1], pos[2])
                    h_ref = init_height  # ep.get_elevation(lat, lon)
                    height_ok = h > h_ref - height_err_max \
                        and h < h_ref + height_err_max
                else:
                    height_ok = True
                if max_dist is not None:
                    dist_ok = np.linalg.norm(pos[:3] - init_pos) < max_dist
                else:
                    dist_ok = True
                if max_time is not None:
                    time_ok = np.abs(pos[-1]) < max_time
                else:
                    time_ok = True
                if max_residual is not None:
                    residuals_ok = np.all(np.abs(res) <= max_residual)
                else:
                    residuals_ok = True
                if residuals_ok and height_ok and dist_ok and time_ok:
                    # Return solution if plausible
                    if return_prn:
                        return pos, PRN[np.asarray(chosen_sats)]
                    else:
                        return pos
        # Return infinity array, indicating absence of plausible solution
        if return_prn:
            return np.full(5, np.inf), np.array([])
        else:
            return np.full(5, np.inf)
    elif mode == 'ransac':
        # Coarse-time navigation with chosen satellites
        pos, res, useful_sats = coarse_time_nav_ransac(
            code_phases,
            PRN, eph_match, init_time_ms,
            init_pos, sort=False, observed_height=observed_height,
            inter_system_bias=inter_system_bias, weights=weights,
            hard_constraint=hard_constraint,
            tropo=tropo, atm_pressure=atm_pressure,
            surf_temp=surf_temp, humidity=humidity, iono=iono,
            ion_alpha=ion_alpha, ion_beta=ion_beta,
            code_period_ms=code_period_ms,
            inlier_probability=np.exp(peak_metrics),
            min_ransac_iterations=1,
            max_ransac_iterations=3,
            min_combo_probability=0.0,
            inlier_threshold=max_residual,
            min_inliers=None,
            max_dist=max_dist, max_time=max_time,
            no_iterations=3
            )
        # Return solution
        if return_prn:
            return pos, useful_sats
        else:
            return pos
    else:
        raise Exception(
            "Chosen mode not supported, select 'SNR' or 'combinatoric'.")


def selective_warm_start_coarse_time_nav(data, interm_freq, sampling_freq,
                                         init_pos, init_bias, init_time,
                                         eph_gps=None, eph_galileo=None,
                                         init_height=0, ms_to_process=1,
                                         elev_mask=10,
                                         pos_err_max=1.414248917270224e+04,
                                         bias_err_max=1.5e4, time_err_max=1,
                                         height_err_max=200,
                                         sort_key='elevation'):
    """Warm-started acquisition follwed by repeated coarse-time navigation.

    Consider no inter-system bias if both, GPS and Galileo, are used.

    Inputs:
        data - GPS snapshot
        interm_freq - Intermediate frequency [Hz]
        sampling_freq - Sampling freqeuncy [Hz]
        init_pos - Initial receiver position in ECEF XYZ [m,m,m]
        init_bias - Initial common pseudorange bias [m]
        init_time - Initial coarse time [s]
        eph_gps - Ephemeris matrix (GPS), None if only Galileo, default=None
        eph_galileo - Ephemeris matrix (Galileo), None if only GPS,
                      default=None
        init_height - Expected height above sea level (WGS84) [m]
        ms_to_process - Number of milliseconds of the snapshot to use
        elev_mask - Elevation mask for predicting satellites [deg]
        pos_err_max - Maximum expected horizontal receiver position error
                      w.r.t. initial position [m]
        bias_err_max - Maximum expected common bias deviation w.r.t. initial
                       bias [m]
        time_err_max - Maximum expected receiver clock deviation w.r.t. initial
                       coarse time [s]
        height_err_max - Maximum expected deviation from expected height [m]
        sort_key - Property for sorting satellites according to their
                   reliability:
                       'SNR' - Signal-to-noise ratio
                       'error' - Error between predicted & estimated code phase
                       'elevation' - Satellite elevation
                       'distance' - Satellite-receiver distance

    Output:
        state - Receiver ECEF position, common bias [m], coarse-time error [s]

    Author: Jonas Beuchert
    """
    if eph_gps is None and eph_galileo is None:
        raise Exception(
            "Chosen GNSS not supported, select 'gps' or 'galileo'.")
    if eph_gps is not None:
        # Acquire GPS satellites
        PRN_gps, code_phases_gps, peak_metrics_gps, code_phase_err_gps = \
            warm_start_acquisition(data, interm_freq, sampling_freq, init_pos,
                                   init_bias, init_time, eph_gps,
                                   ms_to_process, elev_mask, pos_err_max,
                                   bias_err_max, time_err_max)
    else:
        PRN_gps = np.array([])
        code_phases_gps = np.array([])
        peak_metrics_gps = np.array([])
        code_phase_err_gps = np.array([])
        eph_gps = np.empty((21, 0))
    if eph_galileo is not None:
        # Acquire Galileo satellites
        PRN_gal, code_phases_gal, peak_metrics_gal, code_phase_err_gal = \
            warm_start_acquisition(data, interm_freq, sampling_freq, init_pos,
                                   init_bias, init_time, eph_galileo,
                                   ms_to_process, elev_mask, pos_err_max,
                                   bias_err_max, time_err_max, gnss='galileo')
        # Map 1-4 ms to 0-1 ms
        code_phases_gal = np.mod(code_phases_gal, 1)
        # Make Galileo PRNs unique
        eph_galileo[0, :] = eph_galileo[0, :] + 100
        PRN_gal = PRN_gal + 100
    else:
        PRN_gal = np.array([])
        code_phases_gal = np.array([])
        peak_metrics_gal = np.array([])
        code_phase_err_gal = np.array([])
        eph_galileo = np.empty((21, 0))

    # Concatenate arrays
    eph = np.hstack((eph_gps, eph_galileo))
    PRN = np.hstack((PRN_gps, PRN_gal))
    code_phases = np.hstack((code_phases_gps, code_phases_gal))
    peak_metrics = np.hstack((peak_metrics_gps, peak_metrics_gal))
    code_phase_err = np.hstack((code_phase_err_gps, code_phase_err_gal))

    # CTN
    return selective_coarse_time_nav(PRN, code_phases, peak_metrics, eph,
                                     init_pos, init_time, init_height,
                                     height_err_max, sort_key, code_phase_err)


def coarse_time_nav_mle(init_pos, init_time, code_phase, vis, eph, peak_height,
                        observed_height=np.nan, code_period=1.0e-3,
                        search_space_pos=np.array([20.0e3, 20.0e3, 0.2e3]),
                        search_space_time=0.02, hard_constraint=False,
                        linear_pr_prediction=False, trop=True,
                        atm_pressure=1013.0, surf_temp=293.0,
                        humidity=50.0, iono=None, ion_alpha=np.array([]),
                        ion_beta=np.array([]), time_out=2,
                        optim_opt=0,
                        std=2.0**np.arange(5, -7, -2) * 8.3333e-07):
    """Gradient-based maximum-likelihood estimation for coarse-time navigation.

    Does not consider inter-system bias.

    Inputs:
        init_pos - Coarse receiver position in ECEF XYZ coordinates [m,m,m]
        init_time - Coarse absolute GNSS times [s], list with one value for
                    each GNSS
        code_phase - Code phases of satellites that are expected to be visible
                     [ms], list of numpy arrays, one for each GNSS
        vis - PRNs of satellites that are expected to be visible, list of numpy
              arrays, one for each GNSS
        eph - Matching navigation data, list of 21-row 2D numpy arrays
        peak_height - Reliability measures for satellites that are expected to
                      be visible, e.g., SNR [dB], list of numpy arrays
        observed_height - Height measurement [m], use NaN if not available,
                          default=np.nan
        code_period - Code length [s], list with one value for each GNSS,
                      default=1.0e-3
        search_space_pos - Spatial search space width in ENU coordinates
                           [m,m,m], default=np.array([20.0e3, 20.0e3, 0.2e3])
        search_space_time - Temporal search space width [s], default=0.02
        hard_constraint - Flag indicating if observed_height is used as
                          additional observation / soft constraint
                          (hard_constraint=False) or hard equality constraint
                          (hard_constraint=True), default=False
        linear_pr_prediction - Flag indicating if pseudoranges are predicted
                               using non-linear (linear_pr_prdcition=False) or
                               linear (linear_pr_prdcition=True) approximation,
                               default=False
        trop - Model for troposheric correction: either None or False
               for no correction, 'goad' or True for the model of C. C.
               Goad and L. Goodman, 'hopfield' for the model of H. S.
               Hopfield, or 'tsui' for the model of J. B.-Y. Tsui
               [default=True]
        atm_pressure - Atmospheric pressure at receiver location [mbar] for
                       troposheric correction, [default=1013.0]
        surf_temp - Surface temperature at receiver location [K] for
                    troposheric corrrection [default=293.0]
        humidity - Humidity at receiver location [%] for troposheric
                   correction [default=50.0]
        iono - Model for ionospheric correction: either None for no
               correction, 'klobuchar' for the model of J. Klobuchar, or
               'tsui' for the model of J. B.-Y. Tsui [default=None]
        ion_alpha - Alpha parameters of Klobuchar model for ionospheric
                    correction [default=np.array([])]
        ion_beta - Beta parameters of Klobuchar model for ionospheric
                   correction [default=np.array([])]
        time_out - Duration per iteration until timeout [s], use non-positive
                   number, np.nan, or np.inf to disable, only if optim_opt=2,
                   default=2
        optim_opt - Solver selection:
                    0 - autoptim based on autograd and scipy.minimize (default)
                    1 - L-BFGS-B from scipy.minimize
                    2 - LN_NELDERMEAD from nlopt
    Outputs:
        h_opt - Coarse-time navigation solution: receiver position in ECEF XYZ
                coordinates [m,m,m], common bias [m], and coarse-time error [s]
        useful_sats - PRNs of satellites whose observed pseudoranges are
                      within 200 m of the obtained solution, list of numpy,
                      arrays, one for each GNSS

    Author: Jonas Beuchert
    """
    def bayes_classifier(x, galileo=False):
        """Probability of code phase being invalid or valud based on SNR.

        Inputs:
            x - Signal-to-noise ratio (SNR) array [dB]
            galileo - Type of GNSS, GPS (galileo=False) or Galileo
                      (galileo=True), default=False
        Output:
            p - 2xN array with probabilities for code phases being invalid
                (1st row) or valid (2nd row)

        Author: Jonas Beuchert
        """
        if not galileo:  # GPS
            # Class means
            mu0 = 7.7
            mu1 = 15.1
            # Class standard deviations
            sigma0 = 0.67
            sigma1 = 4.65
            # Class priors
            p0 = 0.27
            p1 = 0.73
        else:  # Galileo
            # Class means
            mu0 = 14.2
            mu1 = 19.1
            # Class standard deviations
            sigma0 = 1.13
            sigma1 = 4.12
            # Class priors
            p0 = 0.62
            p1 = 0.38
        # if np.isclose(code_period, 1e-3, atol=0.5e-3):  # GPS L1
        #     # Class means
        #     mu0 = 7.71  # 7.7
        #     mu1 = 13.45  # 15.1
        #     # Class standard deviations
        #     sigma0 = 0.74  # 0.67
        #     sigma1 = 3.16  # 4.65
        #     # Class priors
        #     p0 = 0.16  # 0.27
        #     p1 = 0.84  # 0.73
        # elif np.isclose(code_period, 4e-3, atol=0.5e-3):  # Galileo E1
        #     # Class means
        #     mu0 = 11.17  # 14.2
        #     mu1 = 16.71  # 19.1
        #     # Class standard deviations
        #     sigma0 = 0.69  # 1.13
        #     sigma1 = 3.93  # 4.12
        #     # Class priors
        #     p0 = 0.28  # 0.62
        #     p1 = 0.72  # 0.38
        # else:  # BeiDou B1C, GPS L1
        #     # Class means
        #     mu0 = 15.43
        #     mu1 = 21.92
        #     # Class standard deviations
        #     sigma0 = 0.75
        #     sigma1 = 3.99
        #     # Class priors
        #     p0 = 0.55
        #     p1 = 0.45
        px0 = ss.norm(mu0, sigma0).pdf(x) * p0
        px1 = ss.norm(mu1, sigma1).pdf(x) * p1
        return np.array([px0, px1]) / (px0 + px1)

    def neg_log_likelihood(h, initPosGeo, initBias, initTime, codePhase,
                           pValid, observedHeight, std, hard_constraint,
                           pr_prediction_obj, linear_pr_prediction,
                           init, final, code_period):
        """Negative log-likelihood of a given position, common bias, and time.

        Inputs:
            h - Hypothesis: position in ENU coordinates relative to initial
                position [m,m,m], common bias relative to initial common bias
                [m], and coarse time relative to initial coarse time [s]
            initPosGeo - Initial receiver position in geodetic coordinates
                         [deg,deg,m]
            initBias - Initial common bias [m]
            initTime - Initial GNSS times [s], list with one value for each
                       GNSS
            codePhase - Code phases of satellites that are expected to be
                        visible [ms], list of numpy arrays, one for each GNSS
            pValid - A-priori probability for a codephase to be valid, list
                     with one element for each GNSS
            observedHeight - Height measurement [m]
            std - Standard deviation of Gaussian noise on pseudoranges [m]
            hard_constraint - Flag indicating if observed_height is used as
                              additional observation / soft constraint
                              (hard_constraint=False) or hard equality
                              constraint (hard_constraint=True)
            pr_prediction_obj - List of PseudorangePrediction objects, one for
                                each GNSS
            linear_pr_prediction - Flag indicating if pseudoranges are
                                   predicted using non-linear
                                   (linear_pr_prdcition=False) or linear
                                   (linear_pr_prdcition=True) approximation
            init - Flag indicating if initial bias estimation is performed
            final - Flag indicating if used satellites shall be returned
            code_period - Code length [s], list with one value for each GNSS
        Output:
            if init=True and final=False
                b - Coarse common bias estimate [m]
            if init=False and final=False
                cd - Negative log-likelihood of hypothesis given observations
            if final=True
                useful_sats_idx - Indices of satellites whose observed
                                  pseudoranges are within 200 m of the obtained
                                  solution, list of numpy arrays, one for each
                                  GNSS

        Author: Jonas Beuchert
        """
        # Convert ENU coordinates to ECEF XYZ coordinates
        if not hard_constraint:
            x, y, z = pm.enu2ecef(h[0], h[1], h[2], initPosGeo[0],
                                  initPosGeo[1], initPosGeo[2])
        else:
            # Fix height
            x, y, z = pm.enu2ecef(h[0], h[1], 0.0, initPosGeo[0],
                                  initPosGeo[1], initPosGeo[2])

        # Restore bias
        b = initBias + h[-2]

        # Restore GPS time [s]
        t = initTime + h[-1]

        # Speed of light [m/s]
        c = 299792458.0

        # Predict pseudoranges and code phases
        n_gnss = len(codePhase)
        codePhasePred = []
        for i_gnss in range(n_gnss):
            if codePhase[i_gnss].size > 0:
                if not linear_pr_prediction:
                    # Non-linear approximation
                    pr = pr_prediction_obj[i_gnss].predict_approx(
                        t[i_gnss], np.array([x, y, z]), b)
                else:
                    # Linear approximation
                    pr = pr_prediction_obj[i_gnss].predict_linear(
                        t[i_gnss], np.array([x, y, z]), b)
                # Convert pseudoranges in code phases [s]
                codePhasePred.append(np.mod(pr/c, code_period[i_gnss]))
            else:
                codePhasePred.append(np.array([]))

            # Transform all code phases into same range (0 - 1 ms)
            # codePhasePred[i_gnss] = np.mod(codePhasePred[i_gnss], 1e-3)
            # codePhase[i_gnss] = np.mod(codePhase[i_gnss], 1)

        if final:
            # Calculate residuals
            useful_sats_idx = []
            for i_gnss in range(n_gnss):
                res = np.mod(codePhase[i_gnss] * 1e-3 - codePhasePred[i_gnss],
                             code_period[i_gnss])  # [s]
                res = np.minimum(res, code_period[i_gnss] - res)  # [s]
                res = res * c  # [m]
                useful_sats_idx.append(np.where(res < 200))
            return useful_sats_idx

        # Initialize log-likelihood
        cd = 0.0

        # Define Gaussian kernel
        # kernel = ss.norm(0.0, std)

        def kernel_pdf(x):
            var = std**2
            denom = (2*np.pi*var)**.5
            num = np.exp(-x**2/(2*var))
            return num/denom

        if init:
            # Initial common bias estimation
            # Number of points for objective function discretisation per ms
            supportingPoints = int(2.0**-1.0 * 4092)
            # Initalize result
            cd = 0.0
            # Least common multiple of code lengths [ms]
            max_code_period = np2.lcm.reduce((code_period/1.0e-3).astype(int))
            # Lenght of correlogram that fits all GNSS
            max_cd_length = max_code_period * supportingPoints
            # Loop over all GNSS
            for i_gnss in range(n_gnss):
                # Lenght of correlogram of current GNSS
                curr_cd_length = (code_period[i_gnss]/1.0e-3
                                  * supportingPoints).astype(int)
                # Get indices in final correlogram
                curr_cd_idx = np.mod(np.arange(max_cd_length), curr_cd_length)
                # Get corresponding times
                timeIdx = np.linspace(0.0, code_period[i_gnss],
                                      curr_cd_length, endpoint=False)
                # Sum kernels centered at GPS code phases
                for satIdx in range(codePhase[i_gnss].shape[0]):
                    # Sum kernels centered at GPS code phases
                    idx = np.mod(timeIdx
                                 + codePhasePred[i_gnss][satIdx]
                                 - codePhase[i_gnss][satIdx]*1.0e-3
                                 + code_period[i_gnss]/2.0,
                                 code_period[i_gnss]) - code_period[i_gnss]/2.0
                    # Likelihood of code phase obs. given that it is valid
                    pPhiValid = kernel_pdf(idx)  # kernel.pdf(idx)
                    # Likelihood of code phase obs. given that it is invalid
                    pPhiInvalid = 1.0 / code_period[i_gnss]
                    # Probability of code phase being valid / invalid given SNR
                    pInvalid_sat = 1.0 - pValid[i_gnss][satIdx]  # p[0]
                    pValid_sat = pValid[i_gnss][satIdx]  # p[1]
                    # Likelihood of code phase
                    pPhi = pPhiValid * pValid_sat + pPhiInvalid * pInvalid_sat
                    # Log-likelihood
                    curr_cd = np.log(pPhi)
                    cd = cd + curr_cd[curr_cd_idx]

        else:
            for i_gnss in range(n_gnss):
                if codePhase[i_gnss].size > 0:
                    # Sum kernels centered at GPS code phases
                    idx = np.mod(codePhasePred[i_gnss]
                                 - codePhase[i_gnss]*1.0e-3
                                 + code_period[i_gnss]/2.0,
                                 code_period[i_gnss]) - code_period[i_gnss]/2.0
                    # Likelihood of code phase obs. given that it is valid
                    pPhiValid = kernel_pdf(idx)  # kernel.pdf(idx)
                    # Likelihood of code phase obs. given that it is invalid
                    pPhiInvalid = 1.0 / code_period[i_gnss]
                    # Probability of code phase being valid / invalid given SNR
                    pInvalid_sat = 1.0 - pValid[i_gnss]  # p[0]
                    pValid_sat = pValid[i_gnss]  # p[1]
                    # Likelihood of code phase
                    pPhi = pPhiValid * pValid_sat + pPhiInvalid * pInvalid_sat
                    # Log-likelihood
                    cd = cd + np.sum(np.log(pPhi))

        if (not hard_constraint and observedHeight is not None
           and not np.isnan(observedHeight) and not np.isinf(observedHeight)):
            # Use height measurement as additional observation
            _, _, predictedHeight = pm.ecef2geodetic(x, y, z)

            def height_kernel_pdf(x):
                var = (c*std)**2
                denom = (2*np.pi*var)**.5
                num = np.exp(-x**2/(2*var))
                return num/denom

            cd = cd + np.log(height_kernel_pdf(
                observedHeight - predictedHeight))
            # cd = cd + np.log(ss.norm(predictedHeight, c*std).pdf(
            #     observedHeight))

        cd = np.clip(cd, -np.finfo(np2.float).max, np.finfo(np2.float).max)

        if init:
            # Account for (unknown) common bias
            maxIdx = np.argmax(cd)
            maxTime = maxIdx / supportingPoints * 1e-3  # [s]
            b = c * maxTime
            # Return initial bias estimate
            return b

        return -cd

    # Constrain search space
    if not hard_constraint:
        # Optimisation without height constraint
        searchSpaceBound = np.array([search_space_pos[0], search_space_pos[1],
                                     search_space_pos[2], np.inf,
                                     search_space_time]) / 2.0
    else:
        # Optimisation with height constraint
        searchSpaceBound = np.array([search_space_pos[0], search_space_pos[1],
                                     np.inf, search_space_time]) / 2.0
    # bounds = so.Bounds(-searchSpaceBound, searchSpaceBound)
    # lb = [x if x > -np.inf else None for x in -searchSpaceBound]
    # ub = [x if x < np.inf else None for x in searchSpaceBound]
    # bounds = list(zip(lb, ub))

    # Transform ECEF XYZ coordinates to geodetic coordinates
    initPosGeo = np.empty(3)
    initPosGeo[0], initPosGeo[1], initPosGeo[2] \
        = pm.ecef2geodetic(init_pos[0], init_pos[1], init_pos[2])

    # Check if observed_height shall be used as hard constraint
    if (hard_constraint and not np.isnan(observed_height)
       and not np.isinf(observed_height)):
        # Fix height
        initPosGeo[2] = observed_height

    # Check if code duration is provided as list; if not correct
    if not (isinstance(code_period, list)
            or isinstance(code_period, np.ndarray)):
        code_period = np.ones(len(vis)) * code_period

    # Probability of code phase being valid / invalid given SNR
    n_gnss = len(vis)
    pValid = []
    pInvalid = []
    for i_gnss in range(n_gnss):
        if i_gnss == 0:
            galileo = False
        else:
            galileo = True
        p = bayes_classifier(peak_height[i_gnss], galileo=galileo)
        pInvalid.append(p[0])
        pValid.append(p[1])

    # Decreasing standard deviation of Gaussian noise on pseudoranges [m]
    # std = 2.0**np.arange(5, -7, -2) * 8.3333e-07

    # Decreasing typical step sizes (east [m], north [m], up [m], common bias
    # [m], coarse-time error [s])
    typicalX = 2.0**np.arange(0, -1, -6) * np.array([500.0, 500.0, 5.0, 300.0,
                                                     1.0e-3])

    # Initial position in ENU coordinates, common bias, and time hypothesis
    # (relative to initial values)
    if not hard_constraint:
        h_opt = np.zeros(5)
    else:
        # Do not optimise over height
        h_opt = np.zeros(4)

    # Initialize approximate pseudorange prediction
    pr_prediction_obj = []
    for i_gnss in range(n_gnss):
        if vis[i_gnss].shape[0] > 0:
            pr_prediction_obj.append(PseudorangePrediction(
                vis[i_gnss], eph[i_gnss], init_time[i_gnss],
                init_pos, 0, trop=trop, atm_pressure=atm_pressure,
                surf_temp=surf_temp, humidity=humidity, iono=iono,
                ion_alpha=ion_alpha, ion_beta=ion_beta))
        else:
            pr_prediction_obj.append(None)

    # Get initial common bias estimate
    initBias = neg_log_likelihood(
        h_opt, initPosGeo, 0.0, init_time, code_phase, pValid, observed_height,
        std[0], hard_constraint, pr_prediction_obj, linear_pr_prediction, True,
        False, code_period)

    # Multiple iterations with decreasing pseudorange standard deviation
    for itIdx in range(std.shape[0]):
        print('Start iteration {}'.format(itIdx))

        try:
            if optim_opt == 2:
                import nlopt
                algorithm = nlopt.LN_NELDERMEAD  # LN_SBPLX (works, too)
                opt = nlopt.opt(algorithm, 5)

                def objective_fun(h, grad):
                    return neg_log_likelihood(
                        h,
                        initPosGeo, initBias, init_time, code_phase,
                        pValid, observed_height, std[itIdx],
                        hard_constraint, pr_prediction_obj,
                        linear_pr_prediction, False, False, code_period
                        )
                opt.set_min_objective(objective_fun)

                opt.set_lower_bounds(-searchSpaceBound)
                opt.set_upper_bounds(searchSpaceBound)

                h_opt = opt.optimize(h_opt)
                # h_opt = timeout(timeout=time_out)(so.minimize)(
                #     neg_log_likelihood, h_opt, args=(
                #         initPosGeo, initBias, init_time, code_phase, pValid,
                #         observed_height, std[itIdx], hard_constraint,
                #         pr_prediction_obj, linear_pr_prediction, False, False),
                #     method='L-BFGS-B', bounds=bounds, jac={'2-points'},
                #     options={'finite_diff_rel_step': typicalX})#,
                #                             # 'gtol': 0.0001})
                #                             # 'maxiter': 100-itIdx*10,
                #                             # 'maxfun': 700-itIdx*100})
            elif (optim_opt == 1 and time_out is not np.nan and time_out > 0
                  and time_out < np.inf):
                bounds = so.Bounds(-searchSpaceBound, searchSpaceBound)
                h_opt = timeout(timeout=time_out)(so.minimize)(
                    neg_log_likelihood, h_opt, args=(
                        initPosGeo, initBias, init_time, code_phase,
                        pValid, observed_height, std[itIdx],
                        hard_constraint, pr_prediction_obj,
                        linear_pr_prediction, False, False, code_period
                        ),
                    method='L-BFGS-B', bounds=bounds)
                h_opt = h_opt.x
            elif optim_opt == 0:
                import autoptim as ao
                lb = [x if x > -np.inf else None
                      for x in -searchSpaceBound]
                ub = [x if x < np.inf else None
                      for x in searchSpaceBound]
                bounds = list(zip(lb, ub))
                h_opt = ao.minimize(neg_log_likelihood, h_opt, args=(
                        initPosGeo, initBias, init_time, code_phase,
                        pValid, observed_height, std[itIdx],
                        hard_constraint, pr_prediction_obj,
                        linear_pr_prediction, False, False, code_period),
                    bounds=bounds)
                h_opt = h_opt[0]

                #     h_opt = timeout(timeout=time_out)(so.minimize)(
                #         neg_log_likelihood, h_opt, args=(
                #             initPosGeo, initBias, init_time, code_phase_gps,
                #             code_phase_galileo, pValidGPS, pValidGalileo,
                #             observed_height, std[itIdx], hard_constraint,
                #             pr_prediction_obj_gps, pr_prediction_obj_galileo,
                #             linear_pr_prediction, False, False), method='SLSQP',
                #         bounds=bounds, jac={'2-points'},
                #         options={'finite_diff_rel_step': typicalX})
                # elif optim_opt == 3:
                #     h_opt = timeout(timeout=time_out)(so.minimize)(
                #         neg_log_likelihood, h_opt, args=(
                #             initPosGeo, initBias, init_time, code_phase_gps,
                #             code_phase_galileo, pValidGPS, pValidGalileo,
                #             observed_height, std[itIdx], hard_constraint,
                #             pr_prediction_obj_gps, pr_prediction_obj_galileo,
                #             linear_pr_prediction, False, False), method='SLSQP',
                #         bounds=bounds, jac={'2-points'})
                # elif optim_opt == 4:
                #     h_opt = timeout(timeout=time_out)(so.minimize)(
                #         neg_log_likelihood, h_opt, args=(
                #             initPosGeo, initBias, init_time, code_phase_gps,
                #             code_phase_galileo, pValidGPS, pValidGalileo,
                #             observed_height, std[itIdx], hard_constraint,
                #             pr_prediction_obj_gps, pr_prediction_obj_galileo,
                #             linear_pr_prediction, False, False), method='TNC',
                #         bounds=bounds, jac={'2-points'})
                # elif optim_opt == 5:
                #     h_opt = timeout(timeout=time_out)(so.minimize)(
                #         neg_log_likelihood, h_opt, args=(
                #             initPosGeo, initBias, init_time, code_phase_gps,
                #             code_phase_galileo, pValidGPS, pValidGalileo,
                #             observed_height, std[itIdx], hard_constraint,
                #             pr_prediction_obj_gps, pr_prediction_obj_galileo,
                #             linear_pr_prediction, False, False), method='trust-constr',
                #         bounds=bounds, jac={'2-points'})
            elif optim_opt == 1:
                h_opt = so.minimize(
                    neg_log_likelihood, h_opt, args=(
                        initPosGeo, initBias, init_time, code_phase, pValid,
                        observed_height, std[itIdx], hard_constraint,
                        pr_prediction_obj, linear_pr_prediction, False, False),
                    options={'finite_diff_rel_step': typicalX})
                h_opt = h_opt.x#,
                                            # 'gtol': 0.0001})
                                            # 'maxiter': 100-itIdx*10,
                                            # 'maxfun': 700-itIdx*100})
            else:
                raise ValueError(
                    "Chosen option not available, use optim_opt=0,1,2.")
        except (KeyboardInterrupt, ImportError):
            # Do not ignore those errors
            raise
        except:
            print("Single iteration unsuccesful.")
            pass

    # Get satellites with small residuals
    useful_sats_idx = neg_log_likelihood(
        h_opt,
        initPosGeo, initBias, init_time, code_phase, pValid, observed_height,
        std[-1], hard_constraint, pr_prediction_obj, linear_pr_prediction,
        False, True, code_period)
    useful_sats = [vis[i_gnss][useful_sats_idx[i_gnss]]
                   for i_gnss in range(n_gnss)]

    # Convert ENU to ECEF XYZ coordinates
    h_opt[0], h_opt[1], h_opt[2] = pm.enu2ecef(h_opt[0], h_opt[1], h_opt[2],
                                               initPosGeo[0], initPosGeo[1],
                                               initPosGeo[2])

    # Convert relative common bias to absolute one [m]
    h_opt[3] = initBias + h_opt[3]

    # Convert relative time to absolute GPS time [s]
    # h_opt[-1] = init_time + h_opt[-1]

    return h_opt, useful_sats


def coarse_time_nav_mle_wrapper(codePhasesGPS, PRNGPS, eph, initTime, initPos,
                                peakMetricsGPS, mode, optim_opt,
                                search_space_time=0.02,
                                std=2.0**np.arange(5, -7, -2) * 8.3333e-07,
                                trop=True, iono=None, ion_alpha=[], ion_beta=[]
                                ):
    init_pos = np.array(initPos)
    init_time = initTime
    code_phase_gps = np.array(codePhasesGPS)
    vis_gps = np.array(PRNGPS).astype('int')
    eph_gps = np.asarray(eph)
    peak_height_gps = np.array(peakMetricsGPS)
    print(trop)
    print(iono)
    ion_alpha = np.array(ion_alpha)
    ion_beta = np.array(ion_beta)
    if mode == 'no':
        print('No height observation.')
        observed_height = np.NaN
        hard_constraint = False
    elif mode == 'soft':
        print('Height observation as soft constraint.')
        observed_height = 74.154
        hard_constraint = False
    elif mode == 'hard':
        print('Height observation as hard constraint.')
        observed_height = 74.154
        hard_constraint = True

    try:
        pos, _ = timeout(timeout=10)(coarse_time_nav_mle)(
            init_pos, [init_time], [code_phase_gps], vis=[vis_gps],
            eph=[eph_gps],
            observed_height=observed_height,
            peak_height=[peak_height_gps],
            search_space_pos=np.array([20.0e3, 20.0e3, 0.2e3]),
            search_space_time=search_space_time,
            hard_constraint=hard_constraint,
            linear_pr_prediction=False, optim_opt=int(optim_opt),
            std=np.asarray(std),
            trop=trop, iono=iono, ion_alpha=ion_alpha, ion_beta=ion_beta)
        return pos
    except:
        return np.full(5, np.inf)


def horizontal_dilution_of_precision(sats, eph, coarse_time, rec_pos):
    """Calculate horizontal dilution of precision (HDOP).

    HDOP represents an approximate ratio factor between the precision in
    the measurements and in positioning. This ratio is computed only from
    the satellites-receiver geometry.

    Inputs:
        sats - PRNs of satellites that are useful, numpy array with unique IDs
        eph - Navigation data as matrix, 21-row 2D numpy array
        coarse_time - Coarse absolute GNSS times [s], single value or array for
                      different GNSS times with one value for each satellite
        rec_pos - Receiver position in ECEF XYZ coordinates [m,m,m]

    Output:
        hdop - Horizontal dilution of precision

    Author: Jonas Beuchert
    Implemented according to: Van Diggelen, Frank Stephen Tromp. A-GPS:
        Assisted GPS, GNSS, and SBAS. Artech house, 2009.
    """
    # Speed of light [m/s]
    c = 299792458.0
    # Number of satellites
    nSats = sats.shape[0]

    if not isinstance(coarse_time, np.ndarray) or coarse_time.shape == ():
        coarse_time = coarse_time * np.ones(sats.shape)

    # GPS time since 1980 to time of week (TOW) [s]
    coarseTimeTOW = np.mod(coarse_time, 7 * 24 * 60 * 60)

    if eph.shape[1] != nSats or np.any(eph[0] != sats):
        # Identify matching columns in ephemeris matrix, closest column in time
        # for each satellite
        col = np.array([ep.find_eph(eph, s_i, t_i)
                        for s_i, t_i in zip(sats, coarseTimeTOW)])
        if col.size == 0:
            print("Cannot find satellite in navigation data.")
            return np.nan
        # Extract matching columns
        eph = eph[:, col]

    # Find satellite positions at coarse transmission time
    t_corr = np.array([ep.get_sat_clk_corr(coarseTimeTOW[k],
                                           np.array([sats[k]]),
                                           eph[:, k, np.newaxis])[0]
                       for k in range(nSats)])
    txGPS = coarseTimeTOW - t_corr
    satPosCoarse = ep.get_sat_pos(txGPS, eph)

    distancesCoarse = np.sqrt(np.sum((rec_pos - satPosCoarse)**2, axis=-1))

    travel_times_coarse = distancesCoarse / c

    rel_travel_times = travel_times_coarse - np.min(travel_times_coarse)

    # Correct for satellite clock error
    tCorr = np.empty(nSats)
    for i in range(nSats):
        k = np.array([sats[i]])
        tCorr[i] = ep.get_sat_clk_corr(coarseTimeTOW[i] - rel_travel_times[i],
                                       k, eph[:, i, np.newaxis])
    txGPS = coarseTimeTOW - rel_travel_times - tCorr

    # Get satellite position at corrected transmission time
    satPos = ep.get_sat_pos(txGPS, eph)
    satPos_fut = ep.get_sat_pos(txGPS+1, eph)

    # Calculate rough propagation delay
    travelTime = np.linalg.norm(satPos - rec_pos, axis=1) / c

    # Rotate satellite ECEF coordinates due to earth rotation during
    # signal travel time
    OmegaEdot = 7.292115147e-5  # Earth's angular velocity [rad/s]
    omegaTau = OmegaEdot * travelTime  # Angle [rad]
    R3 = np.array(
        [np.array([[np.cos(omegaTau[k]), np.sin(omegaTau[k]), 0.0],
                   [-np.sin(omegaTau[k]), np.cos(omegaTau[k]), 0.0],
                   [0.0, 0.0, 1.0]]) for k in range(nSats)]
        )  # Rotation matrix
    rotSatPos = np.array([np.matmul(R3[k], satPos[k])
                          for k in range(nSats)])
    rotSatPos_fut = np.array([np.matmul(R3[k], satPos_fut[k])
                              for k in range(nSats)])

    rec_pos_geo = pm.ecef2geodetic(rec_pos[0], rec_pos[1], rec_pos[2])
    sat_pos_enu = np.array([pm.ecef2enu(
        sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2],
        rec_pos_geo[0], rec_pos_geo[1], rec_pos_geo[2]
        ) for sat_pos_ecef in rotSatPos])
    sat_pos_normalized = sat_pos_enu / np.linalg.norm(sat_pos_enu,
                                                      axis=1)[:, np.newaxis]
    # Calculate pseudorange rate (~ range rate)
    sat_pos_fut_enu = np.array([pm.ecef2enu(
        sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2],
        rec_pos_geo[0], rec_pos_geo[1], rec_pos_geo[2]
        ) for sat_pos_ecef in rotSatPos_fut])
    sat_vel = sat_pos_fut_enu - sat_pos_enu
    v = np.array([np.array([np.dot(-sat_vel_k, sat_pos_normalized_k)])
                  for sat_vel_k, sat_pos_normalized_k
                  in zip(sat_vel, sat_pos_normalized)])
    # Build H matrix
    A = np.hstack((sat_pos_normalized, np.ones((nSats, 1)), v))
    try:
        Q = np.linalg.inv(A.T @ A)
        dilution_east_squared = Q[0, 0]
        dilution_north_squared = Q[1, 1]
        return np.sqrt(dilution_east_squared + dilution_north_squared)
    except:
        print("Cannot calculate HDOP.")
        return np.nan


def coarse_time_nav_ransac(obs, sats, Eph, TOW_assist_ms, rec_loc_assist,
                           sort=False, observed_height=None,
                           inter_system_bias=False, weights=None,
                           hard_constraint=False, tropo='goad',
                           atm_pressure=1013.0, surf_temp=293.0, humidity=50.0,
                           iono=None, ion_alpha=np.array([]),
                           ion_beta=np.array([]), code_period_ms=1,
                           inlier_probability=None,
                           min_ransac_iterations=1,
                           max_ransac_iterations=1000,
                           desired_probability=None,
                           inlier_threshold=200,
                           min_inliers=6,
                           max_dist=150.0e3, max_time=60.0,
                           min_combo_probability=0.0,
                           max_combo_size=6,
                           hdop=False, no_iterations=6):
    """Compute receiver position using coarse-time navigation and RANSAC.

    Compute receiver position from fractional pseudoranges using coarse-time
    navigation and non-linear least-squares optimisation.
    The initial position should be within 100 - 150 km of the true position and
    the initial coarse time within about 1 min of the true time.
    Works for multiple GNSS, too, e.g., GPS and Galileo. If using more than
    one, then concatenate navigation data (ephemerides) and make sure that
    satellite indices are unique. E.g., use 1-32 for GPS and 201-250 for
    Galileo.

    Use RANSAC algorithm for outlier detection.

    Inputs:
        obs - Observations, the fractional pseudo-ranges (sub-millisecond).
        sats - SV numbers associated with each observation.
        Eph - Table of ephemerides, each column associated with a satellite.
        TOW_assist_ms - Coarse time of week [ms], single value or array for
                        different GNSS times with one value for each satellite.
        rec_loc_assist - Initial receiver position in ECEF XYZ coordinates.
        sort - Re-sort satellites according to distance [default=False].
        observed_height - Height observation [m], default=None
        inter_system_bias - Flag indicating if a bias between 2 GNSS is
                            added as optimisation variable assuming that all
                            satellites with PRN > 100 belong to the 2nd GNSS
                            [default=False]
        weights - Weight for each observation (height at the end, if present)
                  [default=None]
        hard_constraint - False: Use oberserved_height as additional
                          observation, i.e., as soft constraint. True: Use
                          observed_height as hard equality constraint.
                          [default=False]
        tropo - Model for troposheric correction: either None, 'goad' for the
                model of C. C. Goad and L. Goodman, 'hopfield' for the model of
                H. S. Hopfield, or 'tsui' for the model of J. B.-Y. Tsui
                [default='goad']
        atm_pressure - Atmospheric pressure at receiver location [mbar] for
                       troposheric correction, [default=1013.0]
        surf_temp - Surface temperature at receiver location [K] for
                    troposheric corrrection [default=293.0]
        humidity - Humidity at receiver location [%] for troposheric correction
                   [default=50.0]
        iono - Model for ionospheric correction: either None or 'klobuchar'
               for the model of J. Klobuchar [default=None]
        ion_alpha - Alpha parameters for Klobuchar model [default=np.array([])]
        ion_beta - Beta parameters for Klobuchar model [default=np.array([])]
        code_period_ms - Length of code [ms], either a single value for all
                         satellites or a numpy array with as many elements as
                         satellites [default=1]
        inlier_probability - default=None
        min_ransac_iterations - Minimum number of combos tested per certain
                                combo size, default=1
        max_ransac_iterations - Maximum number of combos tested per certain
                                combo size, default=1000
        desired_probability - Obsolet, default=None
        inlier_threshold - Upper limit for residuals of satellites to be
                           plausible / to be inliers [m], default=200
        min_inliers - Minimum number of inliers for plausible solution,
                      default=6
        max_dist - Maximum spatial distance between 2 consecutive fixes to be
                   plausible [m], default=150.0e3
        max_time - Maximum temporal distance between 2 consecutive fixes to be
                   plausible [s], i.e., the change of the coarse-time error,
                   default=60.0
        min_combo_probability - Minimum a-priori reliability probability of
                                combo to be tested, default=0.0
        max_combo_size - Maximum number of satellites in subset initially used
                         by RANSAC, default=6
        hdop - Flag if horizontal dilution of precision is returned as 4th
               output, default=False
        no_iterations - Number of non-linear least-squares iterations,
                        default=6

    Outputs:
        state - ECEF XYZ position [m,m,m], common bias [m], coarse-time error
                [m]; np.NaN if optimization failed
        delta_z - Residuals (of pseudoranges) [m]
        useful_sats - PRNs of satellites that were used to obtain the solution
        hdop - (Only if hdop=True) Horizontal dilution of precision

    Author: Jonas Beuchert
    """
    from satellite_combos import get_combos

    def assign_integers(sats, svInxListByDistance, obs, Eph, approx_distances,
                        code_period_ms=1):
        """Assign Ns according to van Diggelen's algorithm.

        Author: Jonas Beuchert
        """
        light_ms = 299792458.0 * 0.001
        N = np.zeros(sats.shape)
        approx_distances = approx_distances / light_ms  # Distances in millisec
        # Distances from ms to code periods
        approx_distances = approx_distances / code_period_ms

        unique_code_periods = np.unique(code_period_ms)
        n_different_code_periods = len(unique_code_periods)
        if not isinstance(code_period_ms, np.ndarray):
            N0_inx = svInxListByDistance[0]
        else:
            N0_inx = np.empty(n_different_code_periods)
            for c_idx in np.arange(n_different_code_periods):
                N0_inx[c_idx] = svInxListByDistance[
                    code_period_ms == unique_code_periods[c_idx]
                    ][0]
            N0_inx = N0_inx.astype(int)
        N[N0_inx] = np.floor(approx_distances[N0_inx])

        delta_t = Eph[18, :] * 1000.0  # From sec to millisec
        # Time errors from ms to code periods
        delta_t = delta_t / code_period_ms

        # Observed code phases from ms to code periods
        obs = obs / code_period_ms

        if not isinstance(code_period_ms, np.ndarray):
            for k in svInxListByDistance[1:]:
                N[k] = np.round(N[N0_inx] + obs[N0_inx] - obs[k] +
                                (approx_distances[k] - delta_t[k]) -
                                (approx_distances[N0_inx] - delta_t[N0_inx]))
        else:
            for c_idx in np.arange(n_different_code_periods):
                curr_svInxListByDistance = svInxListByDistance[
                    code_period_ms == unique_code_periods[c_idx]
                    ]
                for k in curr_svInxListByDistance[1:]:
                    N[k] = np.round(N[N0_inx[c_idx]] + obs[N0_inx[c_idx]]
                                    - obs[k] +
                                    (approx_distances[k] - delta_t[k]) -
                                    (approx_distances[N0_inx[c_idx]]
                                     - delta_t[N0_inx[c_idx]]))

        # Floored distances from code periods to ms
        N = N * code_period_ms

        return N, N0_inx

    def tx_RAW2tx_GPS(tx_RAW, Eph):
        """Refactoring.

        Author: Jonas Beuchert
        """
        t0c = Eph[20]
        dt = ep.check_t(tx_RAW - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        dt = ep.check_t(tx_GPS - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        return tx_GPS, tcorr

    def e_r_corr(traveltime, X_sat):
        """Rotate satellite by earth rotation during signal travel time.

        Author: Jonas Beuchert
        """
        Omegae_dot = 7.292115147e-5  # rad/sec

        omegatau = Omegae_dot * traveltime
        R3 = np.array([np.array([np.cos(omegatau), np.sin(omegatau), 0.0]),
                       np.array([-np.sin(omegatau), np.cos(omegatau), 0.0]),
                       np.array([0.0, 0.0, 1.0])])
        return R3 @ X_sat

    v_light = 299792458.0
    dtr = np.pi / 180.0
    numSVs = obs.shape[0]  # Number of satellites
    el = np.zeros(numSVs)

    TOW_assist = TOW_assist_ms * 1e-3
    if not isinstance(TOW_assist, np.ndarray) or TOW_assist.shape == ():
        TOW_assist = TOW_assist * np.ones(sats.shape)

    if Eph.shape[1] != numSVs or np.any(Eph[0] != sats):
        # Identify ephemerides columns in Eph
        col_Eph = np.array([ep.find_eph(Eph, sats[k], TOW_assist[k])
                            for k in range(numSVs)])
        Eph = Eph[:, col_Eph]  # Sort according to sats argument

    # If one common bias shall be used for all systems, then all code phases
    # must have same range
    if isinstance(code_period_ms, np.ndarray) and not inter_system_bias:
        # Greatest common divider of code periods
        code_period_ms = np2.gcd.reduce(code_period_ms.astype(int))
        obs = np.mod(obs, code_period_ms)

    # Number of common-bias optimisation variables
    if isinstance(code_period_ms, np.ndarray):
        # Assume one entry in code_period_ms for each satellite
        # One bias for each present code period
        n_bias_opt = np.unique(code_period_ms).shape[0]
    elif inter_system_bias:
        # Same code period for all systems, nevertheless use inter-system bias
        n_bias_opt = 2
    else:
        # Same code period for all systems, do not use inter-system bias
        n_bias_opt = 1

    T_tilde = TOW_assist

    # Find satellite positions at T_tilde
    tx_GPS = np.array([TOW_assist[k]
                       - ep.get_sat_clk_corr(TOW_assist[k],
                                             np.array([sats[k]]),
                                             Eph[:, k, np.newaxis])
                       for k in range(numSVs)])
    satPos_at_T_tilde = np.array([np.squeeze(
        ep.get_sat_pos(tx_GPS[k], Eph[:, k, np.newaxis]))
        for k in range(numSVs)])

    # And then find closest one
    approx_distances = np.sqrt(np.sum(
        (rec_loc_assist - satPos_at_T_tilde)**2, axis=-1))
    if sort:
        svInxListByDistance = np.argsort(approx_distances)
    else:
        svInxListByDistance = np.arange(numSVs)

    plausible_solution = False
    # potential_solution = False
    # min_res_norm = np.inf

    max_sat = max_combo_size+1  # Start trying subsets of 6 satellites
    while not plausible_solution and max_sat > 1:
        max_sat -= 1  # Reduce size of considered satellite subset
        # Get all possible satellite combinations with max_sat satellites
        if sort or numSVs != 15 or max_sat > 6:
            # Cannot use pre-calculated values
            sat_combinations = it.combinations(iter(svInxListByDistance),
                                               max_sat)
            sat_combinations = np.array([*sat_combinations])
        else:
            # Can use pre-calculated values
            sat_combinations = get_combos(15, max_sat)
        # sat_combinations = np.array([np.arange(max_sat)])

        # if inlier_probability is not None:
        #     # Sort combinations according to probability
        #     # Multiply individual probabilities to get joined ones for each combo
        #     inlier_probability_combo = np.prod(
        #         inlier_probability[sat_combinations], axis=-1)
        #     # Sort probabilities across all combos
        #     sorted_idx = np.argsort(inlier_probability_combo)[::-1]
        #     # Sort satellite combos in descending order
        #     sat_combinations = sat_combinations[sorted_idx]

        #     if desired_probability is not None:
        #         # How many combos must we try to exceed the desired probability?
        #         larger_than_desired = np.where(
        #             1-np.cumprod(1-inlier_probability_combo) > desired_probability
        #             )[0]
        #         # Can we achieve the desired probability at all?
        #         if len(larger_than_desired) > 0:
        #             # Select minimum iteration number to achieve desired probability
        #             # Do not violate limits
        #             max_ransac_iterations = max(min_ransac_iterations,
        #                                         min(max_ransac_iterations,
        #                                             larger_than_desired[0]+1))

        # # Use only most likely satellite combos
        # sat_combinations = sat_combinations[0:int(max_ransac_iterations)]

        # Iterate over all rows of sat_combinations, i.e., all combinations
        # with 5 satellites
        # for chosen_sats in sat_combinations:
        max_combo_probability = 1.0
        inlier_probability_updated = inlier_probability
        ransac_iteration_idx = 0
        while (not plausible_solution
               and ransac_iteration_idx < max_ransac_iterations
               and (max_combo_probability >= min_combo_probability
                    or ransac_iteration_idx < min_ransac_iterations)
               and sat_combinations.shape[0] > 0):

            # Calculate probability for each set to contain only inliers
            inlier_probability_combo = np.prod(
                inlier_probability_updated[sat_combinations], axis=-1)
            # Pick combo with highest probabiliy that has not been picked yet
            # Sort probabilities across all combos
            max_combo_probability_idx = np.argmax(inlier_probability_combo)
            chosen_sats = sat_combinations[max_combo_probability_idx]
            max_combo_probability = inlier_probability_combo[max_combo_probability_idx]
            if (max_combo_probability >= min_combo_probability
                    or ransac_iteration_idx < min_ransac_iterations):

                # Delete chosen combo
                sat_combinations = np.delete(sat_combinations,
                                             max_combo_probability_idx,
                                             axis=0)

                # Bring 1st satellite of combo to the front
                svInxListByDistance = np.concatenate((
                    np.array([chosen_sats[0]]),
                    np.arange(0, chosen_sats[0]),
                    np.arange(chosen_sats[0]+1, numSVs)
                    ))
                # Could use only top max_sat satellites here

                # Assign N numbers:
                Ns, N0_inx = assign_integers(sats, svInxListByDistance, obs, Eph,
                                             approx_distances, code_period_ms)
                # Could do only top max_sat here and other later if necessary

                # Now find K numbers:
                if isinstance(code_period_ms, np.ndarray):
                    unique_code_periods = np.unique(code_period_ms)
                    n_different_code_periods = len(unique_code_periods)
                    Ks = np.empty(Ns.shape)
                    for c_idx in np.arange(n_different_code_periods):
                        Ks[code_period_ms == unique_code_periods[c_idx]] \
                            = Ns[code_period_ms == unique_code_periods[c_idx]] \
                            - Ns[N0_inx[c_idx]]
                else:
                    Ks = Ns - Ns[N0_inx]

                fullPRs = Ns + obs  # Full pseudoranges reconstruction in ms
                fullPRs = fullPRs * (v_light * 1e-3)  # In meters

                if hard_constraint:
                    # Transform ECEF XYZ coordinates to geodetic coordinates
                    latitude, longitude, _ = pm.ecef2geodetic(
                        rec_loc_assist[0], rec_loc_assist[1], rec_loc_assist[2])
                    initPosGeo = np.array([latitude, longitude, observed_height])
                    # Initial receiver postion in EN coordinates
                    state = np.zeros(2+n_bias_opt+1)
                else:
                    # Preliminary guess for receiver position, common bias, and assistance
                    # error ([x y z b et] or [x y z b1 b2 et])
                    state = np.concatenate((rec_loc_assist, np.zeros(n_bias_opt+1)))

                for iter_idx in range(no_iterations):
                # iter_idx = 0
                # while iter_idx < no_iterations:
                    if hard_constraint:
                        H = np.empty((numSVs, 3+n_bias_opt))
                        # H = np.empty((max_sat, 3+n_bias_opt))
                    else:
                        H = np.empty((numSVs, 4+n_bias_opt))
                        # H = np.empty((max_sat, 4+n_bias_opt))
                    delta_z = np.empty(numSVs)  # Observed minus computed observation
                    # delta_z = np.empty(max_sat)  # Observed minus computed observation

                    # Coarse-time error
                    Et = state[-1]  # In seconds

                    for k in svInxListByDistance:
                    # for k in chosen_sats:
                        # Alternatively, iterate just over top max_sat satellites
                        # Check if satellite is in  desired subset
                        # or last iteration is reached
                        if (iter_idx == no_iterations-1
                                or np.in1d(k, chosen_sats)[0]):

                            # Common bias [m]
                            if isinstance(code_period_ms, np.ndarray):
                                code_period_idx = np.where(
                                    unique_code_periods == code_period_ms[k])[0]
                                b = state[-1-n_bias_opt+code_period_idx]
                            elif inter_system_bias and sats[k] > 100:
                                # Common bias of 2nd GNSS
                                b = state[-2]
                            else:
                                # Common bias of 1st GNSS
                                b = state[-1-n_bias_opt]

                            Kk = Ks[k]

                            tx_GPS, tcorr = tx_RAW2tx_GPS(T_tilde[k] - Et - Kk * 1e-3,
                                                          Eph[:, k])

                            X = ep.get_sat_pos(tx_GPS, Eph[:, k])
                            X_fut = ep.get_sat_pos(tx_GPS + 1, Eph[:, k])
                            satECEF = X
                            sat_futECEF = X_fut
                            if hard_constraint:
                                # Transform ECEF XYZ coordinates to ENU coordinates
                                X[0], X[1], X[2] = pm.ecef2enu(X[0], X[1], X[2], initPosGeo[0],
                                                               initPosGeo[1], initPosGeo[2])
                                X_fut[0], X_fut[1], X_fut[2] = pm.ecef2enu(X_fut[0], X_fut[1],
                                                                           X_fut[2],
                                                                           initPosGeo[0],
                                                                           initPosGeo[1],
                                                                           initPosGeo[2])
                                state_memory = state
                                state = np.array([state[0], state[1], 0.0])

                            # This if case calculates something about trop
                            if iter_idx == 0:
                                traveltime = 0.072
                                Rot_X = satECEF
                                Rot_X_fut = sat_futECEF
                                trop = 0.0
                            else:
                                if hard_constraint:
                                    posECEF = np.empty(3)
                                    posECEF[0], posECEF[1], posECEF[2] \
                                        = pm.enu2ecef(state[0], state[1], 0.0,
                                                      initPosGeo[0], initPosGeo[1],
                                                      initPosGeo[2])
                                else:
                                    posECEF = state[:3]

                                rho2 = (satECEF[0] - posECEF[0])**2 \
                                    + (satECEF[1] - posECEF[1])**2 \
                                    + (satECEF[2] - posECEF[2])**2  # Distance squared
                                traveltime = np.sqrt(rho2) / v_light
                                Rot_X = e_r_corr(traveltime, satECEF)

                                Rot_X_fut = e_r_corr(traveltime, sat_futECEF)

                                rho2 = (Rot_X[0] - posECEF[0])**2 \
                                    + (Rot_X[1] - posECEF[1])**2 \
                                    + (Rot_X[2] - posECEF[2])**2

                                if tropo == 'goad':
                                    az, el, dist = ep.topocent(posECEF, Rot_X-posECEF)
                                    trop = ep.tropo(np.sin(el * dtr), 0.0, atm_pressure,
                                                    surf_temp, humidity, 0.0, 0.0, 0.0)
                                elif tropo == 'hopfield':
                                    surf_temp_celsius = surf_temp-273.15
                                    saturation_vapor_pressure = 6.11*10.0**(
                                        7.5*surf_temp_celsius/(237.7+surf_temp_celsius))
                                    vapor_pressure = humidity/100.0 * saturation_vapor_pressure
                                    trop = ep.tropospheric_hopfield(
                                        posECEF, np.array([Rot_X]), surf_temp_celsius,
                                        atm_pressure/10.0, vapor_pressure/10.0)
                                elif tropo == 'tsui':
                                    lat, lon, h = pm.ecef2geodetic(posECEF[0], posECEF[1],
                                                                   posECEF[2])
                                    az, el, srange = pm.ecef2aer(Rot_X[0], Rot_X[1], Rot_X[2],
                                                                 lat, lon, h)
                                    trop = ep.tropospheric_tsui(el)
                                else:
                                    trop = 0.0

                                if iono == 'klobuchar':
                                    trop = trop + ep.ionospheric_klobuchar(
                                        posECEF, np.array([Rot_X]),
                                        np.mod(TOW_assist[k], 24*60*60),
                                        ion_alpha, ion_beta) * v_light
                                elif iono == 'tsui':
                                    lat, lon, h = pm.ecef2geodetic(posECEF[0], posECEF[1],
                                                                   posECEF[2])
                                    az, el, srange = pm.ecef2aer(Rot_X[0], Rot_X[1], Rot_X[2],
                                                                 lat, lon, h)
                                    # Convert degrees to semicircles
                                    el = el / 180.0
                                    az = az / 180.0
                                    lat = lat / 180.0
                                    lon = lon / 180.0
                                    # Ionospheric delay [s]
                                    T_iono = ep.ionospheric_tsui(
                                        el, az, lat, lon, TOW_assist[k], ion_alpha, ion_beta)
                                    trop = trop + T_iono * v_light

                            # Subtraction of state[3] corrects for receiver clock offset and
                            # v_light*tcorr is the satellite clock offset
                            predictedPR = np.linalg.norm(Rot_X - state[:3]) + b \
                                - tcorr * v_light + trop  # meters
                            delta_z[k] = fullPRs[k] - predictedPR  # Meters

                            # Now add row to matrix H according to:
                            # -e_k 1 v_k
                            # Notice that it is easier to plug in the location of the satellite
                            # at its T_dot estimation, i.e., Rot_X
                            sat_vel_mps = Rot_X_fut - Rot_X

                            e_k = (Rot_X - (Et + Kk * 1e-3) * sat_vel_mps - state[:3])
                            e_k = e_k / np.linalg.norm(e_k)

                            v_k = np.sum(-sat_vel_mps * e_k, keepdims=True)  # Relative speed

                            if hard_constraint:
                                # Optimise only over E and N coordinate
                                e_k = e_k[:2]
                                # Restore state
                                state = state_memory

                            if isinstance(code_period_ms, np.ndarray):
                                code_period_idx = np.where(
                                    unique_code_periods == code_period_ms[k])
                                jac_common_bias = np.zeros(n_bias_opt)
                                jac_common_bias[code_period_idx] = 1.0
                                # Jacobian w.r.t. to common bias of this GNSS
                                H_row = np.concatenate((-e_k, jac_common_bias,
                                                        v_k))
                            elif not inter_system_bias:
                                # Matrix for 5 optimisation variables
                                H_row = np.concatenate((-e_k, np.ones(1), v_k))
                            else:
                                # Matrix for 6 optimisation variables
                                if sats[k] <= 100:
                                    # Jacobian w.r.t. to common bias of 1st GNSS
                                    H_row = np.concatenate((-e_k, np.ones(1), np.zeros(1),
                                                            v_k))
                                else:
                                    # Jacobian w.r.t. to common bias of 2nd GNSS
                                    H_row = np.concatenate((-e_k, np.zeros(1), np.ones(1),
                                                            v_k))
                            # Append Jacobian to end of matrix
                            H[k] = H_row

                    # RANSAC: Select only max_sat satellites
                    delta_z_chosen = delta_z[chosen_sats]
                    H_chosen = H[chosen_sats]
                    # Do not need this if delta_z and H are only build for chosen sats

                    # Check if height measurement is provided
                    if observed_height is not None and not hard_constraint:
                        # Add Jacobian of height observation
                        H_chosen = np.vstack((H_chosen, jh.jacobian_height(state)))
                        # Predict height based on current state
                        predicted_height = pm.ecef2geodetic(state[0], state[1], state[2]
                                                            )[2]
                        # Add height measurement
                        delta_z_chosen = np.append(delta_z_chosen,
                                                   observed_height - predicted_height)

                    if weights is not None:
                        H_chosen = H_chosen * np.sqrt(weights[np.array(chosen_sats),
                                                              np.newaxis])
                        delta_z_chosen = delta_z_chosen * np.sqrt(weights[np.array(chosen_sats)])

                    x = np.linalg.lstsq(H_chosen, delta_z_chosen, rcond=None)[0]
                    prev_state = state
                    state = state + x

                # # RANSAC check
                # if np.all(np.abs(delta_z_chosen) < inlier_threshold):
                #     potential_solution = True
                #     if np.linalg.norm(delta_z_chosen) < min_res_norm:
                #         state_potential = state
                #         delta_z_potential = delta_z[svInxListByDistance]
                #         sats_potential = sats[np.array(chosen_sats)]
                #         H_potential = H_chosen
                #         min_res_norm = np.linalg.norm(delta_z_chosen)
                #     # Check plausibility of solution
                #     if np.where(np.abs(delta_z) < inlier_threshold
                #                 )[0].shape[0] >= min_inliers \
                #             and np.linalg.norm(state[:3] - rec_loc_assist) < max_dist \
                #             and np.abs(state[-1]) < max_time:
                #         plausible_solution = True
                #         break

                # if iter_idx == no_iterations-1:
                # Plausibility check
                if max_dist is not None:
                    dist_ok = np.linalg.norm(state[:3] - rec_loc_assist) < max_dist
                else:
                    dist_ok = True
                if max_time is not None:
                    time_ok = np.abs(state[-1]) < max_time
                else:
                    time_ok = True
                if inlier_threshold is not None:
                    residuals_ok = np.all(np.abs(delta_z_chosen) < inlier_threshold)
                else:
                    residuals_ok = True
                if min_inliers is not None:
                    n_inliers_ok = np.where(
                        np.abs(delta_z) < inlier_threshold
                        )[0].shape[0] >= min_inliers
                else:
                    n_inliers_ok = True
                if residuals_ok and dist_ok and time_ok and n_inliers_ok:
                    plausible_solution = True

                if not plausible_solution:
                    if max_combo_probability < 1.0:
                        # Update inlier probabilities
                        for sat_idx in np.arange(numSVs):
                            if np.in1d(sat_idx, chosen_sats):
                                inlier_probability_updated[sat_idx] \
                                    = (inlier_probability_updated[sat_idx]
                                       - max_combo_probability) \
                                    / (1.0-max_combo_probability)

                    ransac_iteration_idx += 1
                # else:
                    # Do one more iteration with all inlier sats
                    # chosen_sats = np.where(np.abs(delta_z) < inlier_threshold)[0]
                    # if chosen_sats.shape[0] > max_sat
                        # no_iterations += 1
                        # max_sat = chosen_sats.shape[0]
                # iter_idx += 1

    # if potential_solution and not plausible_solution:
        # print("Found only solution with 5 sats.")
        # print("Sats used for minimum solution: {}".format(sats_potential))

        # if hard_constraint:
        #     # Convert ENU to ECEF XYZ coordinates
        #     [pos_x, pos_y, pos_z] = pm.enu2ecef(state_potential[0],
        #                                         state_potential[1], 0.0,
        #                                         initPosGeo[0], initPosGeo[1],
        #                                         initPosGeo[2])
        #     state_potential = np.concatenate((np.array([pos_x, pos_y, pos_z]),
        #                                       state_potential[2:]))

        # if hdop:
        #     try:
        #         Q = np.linalg.inv(H_potential.T @ H_potential)
        #         dilution_east_squared = Q[0, 0]
        #         dilution_north_squared = Q[1, 1]
        #         hdop = np.sqrt(dilution_east_squared + dilution_north_squared)
        #         return state_potential, delta_z_potential, sats_potential, hdop
        #     except:
        #         print("Cannot calculate HDOP.")
        #         return state_potential, delta_z_potential, sats_potential, np.nan
        # return state_potential, delta_z_potential, sats_potential

    # elif plausible_solution:
    if plausible_solution:

        # print("Sats used for minimum solution: {}".format(chosen_sats))
        # print("Residuals: {}".format(np.abs(delta_z_chosen)))

        # Might have to create full delta_z and full H here
        chosen_sats = np.where(np.abs(delta_z) < inlier_threshold)[0]

        # print("Plausible sats: {}".format(chosen_sats))
        # print("Residuals: {}".format(np.abs(delta_z[chosen_sats])))

        # RANSAC
        delta_z_chosen = delta_z[chosen_sats]
        H_chosen = H[chosen_sats]

        # Check if height measurement is provided
        if observed_height is not None and not hard_constraint:
            # Add Jacobian of height observation
            H_chosen = np.vstack((H_chosen, jh.jacobian_height(state)))
            # Predict height based on current state
            predicted_height = pm.ecef2geodetic(state[0], state[1], state[2]
                                                )[2]
            # Add height measurement
            delta_z_chosen = np.append(delta_z_chosen, observed_height - predicted_height)

        if weights is not None:
            H_chosen = H_chosen * np.sqrt(weights[chosen_sats, np.newaxis])
            delta_z_chosen = delta_z_chosen * np.sqrt(weights[chosen_sats])

        x = np.linalg.lstsq(H_chosen, delta_z_chosen, rcond=None)[0]
        # print("Refinement: {}".format(x[:3]))
        # if np.all(x[:3]) < 200:
        state = prev_state + x
        # else:
        #     print("Could not refine solution. "
        #           + "Return estimate using 5 satellites.")

        if hard_constraint:
            # Convert ENU to ECEF XYZ coordinates
            [pos_x, pos_y, pos_z] = pm.enu2ecef(state[0], state[1], 0.0,
                                                initPosGeo[0], initPosGeo[1],
                                                initPosGeo[2])
            state = np.concatenate((np.array([pos_x, pos_y, pos_z]), state[2:]))

        if hdop:
            try:
                Q = np.linalg.inv(H.T @ H)
                dilution_east_squared = Q[0, 0]
                dilution_north_squared = Q[1, 1]
                hdop = np.sqrt(dilution_east_squared + dilution_north_squared)
                return state, delta_z[svInxListByDistance], sats[chosen_sats], hdop
            except:
                print("Cannot calculate HDOP.")
                return state, delta_z[svInxListByDistance], sats[chosen_sats], np.nan

        return state, delta_z[svInxListByDistance], sats[chosen_sats]

    else:
        # Could not find plausible solution
        # print("Could not find plausible solution.")
        if hdop:
            return np.full(state.shape[0], np.inf), delta_z[svInxListByDistance], np.array([]), np.nan
        return np.full(state.shape[0], np.inf), delta_z[svInxListByDistance], np.array([])


def coarse_time_nav_ransac_simplified(obs, sats, Eph,
                                      TOW_assist_ms, rec_loc_assist,
                                      observed_height=None,
                                      code_period_ms=1,
                                      inlier_probability=None,
                                      min_ransac_iterations=1,
                                      max_ransac_iterations=3,
                                      inlier_threshold=200,
                                      min_inliers=None,
                                      max_dist=15.0e3, max_time=30.0,
                                      min_combo_probability=0.0,
                                      max_combo_size=6,
                                      hdop=False, no_iterations=3):
    """Compute receiver position using coarse-time navigation and RANSAC.

    Compute receiver position from fractional pseudoranges using coarse-time
    navigation and non-linear least-squares optimisation.
    The initial position should be within 100-150 km of the true position and
    the initial coarse time within about 1 min of the true time.
    Works for multiple GNSS, too, e.g., GPS and Galileo. If using more than
    one, then concatenate navigation data (ephemerides) and make sure that
    satellite indices are unique. E.g., use 1-32 for GPS and 201-250 for
    Galileo.

    Use RANSAC-inspired algorithm for outlier detection.

    Inputs:
        obs - Observations, the fractional pseudo-ranges (sub-millisecond).
        sats - SV numbers associated with each observation.
        Eph - Table of ephemerides, each column associated with a satellite.
        TOW_assist_ms - Coarse time of week [ms], single value or array for
                        different GNSS times with one value for each satellite.
        rec_loc_assist - Initial receiver position in ECEF XYZ coordinates.
        observed_height - Height observation [m], default=None
        code_period_ms - Length of code [ms], either a single value for all
                         satellites or a numpy array with as many elements as
                         satellites [default=1]
        inlier_probability - default=None
        min_ransac_iterations - Minimum number of combos tested per certain
                                combo size, default=1
        max_ransac_iterations - Maximum number of combos tested per certain
                                combo size, default=3
        inlier_threshold - Upper limit for residuals of satellites to be
                           plausible / to be inliers [m], default=200
        min_inliers - Minimum number of inliers for plausible solution,
                      default=None
        max_dist - Maximum spatial distance between 2 consecutive fixes to be
                   plausible [m], default=15.0e3
        max_time - Maximum temporal distance between 2 consecutive fixes to be
                   plausible [s], i.e., the change of the coarse-time error,
                   default=30.0
        min_combo_probability - Minimum a-priori reliability probability of
                                combo to be tested, default=0.0
        max_combo_size - Maximum number of satellites in subset initially used
                         by RANSAC, default=6
        hdop - Flag if horizontal dilution of precision is returned as 3rd
               output, default=False
        no_iterations - Number of non-linear least-squares iterations,
                        default=3

    Outputs:
        state - ECEF XYZ position [m,m,m], common bias [m], coarse-time error
                [m]; np.NaN if optimization failed
        delta_z - Residuals (of pseudoranges) [m]
        hdop - (Only if hdop=True) Horizontal dilution of precision

    Author: Jonas Beuchert
    """
    from satellite_combos import get_combos

    def assign_integers(sats, svInxListByDistance, obs, Eph, approx_distances,
                        code_period_ms=1):
        """Assign Ns according to van Diggelen's algorithm.

        Author: Jonas Beuchert
        """
        light_ms = 299792458.0 * 0.001
        N = np.zeros(sats.shape)
        approx_distances = approx_distances / light_ms  # Distances in millisec
        # Distances from ms to code periods
        approx_distances = approx_distances / code_period_ms

        # To integer
        N[0] = np.floor(approx_distances[0])

        delta_t = Eph[18, :] * 1000.0  # From sec to millisec
        # Time errors from ms to code periods
        delta_t = delta_t / code_period_ms

        # Observed code phases from ms to code periods
        obs = obs / code_period_ms

        N[1:] = np.round(N[0] + obs[0] - obs[1:] +
                         (approx_distances[1:] - delta_t[1:]) -
                         (approx_distances[0] - delta_t[0]))

        # Floored distances from code periods to ms
        N = N * code_period_ms

        return N

    def tx_RAW2tx_GPS(tx_RAW, Eph):
        """Refactoring.

        Author: Jonas Beuchert
        """
        t0c = Eph[20]
        dt = ep.check_t_vectorized(tx_RAW - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        dt = ep.check_t_vectorized(tx_GPS - t0c)
        tcorr = (Eph[1] * dt + Eph[19]) * dt + Eph[18]
        tx_GPS = tx_RAW - tcorr
        return tx_GPS, tcorr

    def e_r_corr(traveltime, sat_pos, n_sats):
        """Rotate satellite by earth rotation during signal travel time.

        Author: Jonas Beuchert
        """
        Omegae_dot = 7.292115147e-5  # rad/sec

        omegatau = Omegae_dot * traveltime
        # Vector of rotation matrices
        R3 = np.transpose(np.array([
            np.array([np.cos(omegatau), np.sin(omegatau), np.zeros(n_sats)]),
            np.array([-np.sin(omegatau), np.cos(omegatau), np.zeros(n_sats)]),
            np.array([np.zeros(n_sats), np.zeros(n_sats),  np.ones(n_sats)])]),
            axes=(2, 0, 1))
        # Turn satellite positions into vector of column vectors
        sat_pos = np.array([sat_pos]).transpose(1, 2, 0)
        return np.matmul(R3, sat_pos).reshape(n_sats, 3)

    v_light = 299792458.0  # Speed of light
    numSVs = obs.shape[0]  # Number of satellites

    TOW_assist = TOW_assist_ms * 1e-3  # Milliseonds -> seconds
    if not isinstance(TOW_assist, np.ndarray) or TOW_assist.shape == ():
        # Make sure that times are array
        TOW_assist = TOW_assist * np.ones(sats.shape)

    # Check if correct navigation data columns have been identified already
    if Eph.shape[1] != sats.shape[0] or np.any(Eph[0] != sats):
        # Identify ephemerides columns in Eph
        col_Eph = np.array([ep.find_eph(Eph, sats[k], TOW_assist[k])
                            for k in range(numSVs)])
        Eph = Eph[:, col_Eph]  # Sort according to sats argument

    # If one common bias shall be used for all systems, then all code phases
    # must have same range
    if isinstance(code_period_ms, np.ndarray):
        # Greatest common divider of code periods
        code_period_ms = np2.gcd.reduce(code_period_ms.astype(int))
    obs = np.mod(obs, code_period_ms)

    # Number of common-bias optimisation variables
    # Same code period for all systems, do not use inter-system bias
    n_bias_opt = 1

    T_tilde = TOW_assist

    # Find satellite positions at T_tilde
    tx_GPS = TOW_assist - ep.get_sat_clk_corr_vectorized(TOW_assist, sats, Eph)
    satPos_at_T_tilde = ep.get_sat_pos(tx_GPS, Eph)

    # Find rough distances
    approx_distances = np.sqrt(np.sum(
        (rec_loc_assist - satPos_at_T_tilde)**2, axis=-1))

    # Oder satellites in given order
    sat_idx = np.arange(numSVs)

    # Stop RANSAC when plausible solution is found
    plausible_solution = False
    # Start trying subsets of 6 satellites
    max_sat = max_combo_size+1
    while not plausible_solution and max_sat > 1:
        max_sat -= 1  # Reduce size of considered satellite subset
        # Get all possible satellite combinations with max_sat satellites
        sat_combinations = get_combos(numSVs, max_sat)

        max_combo_probability = 1.0
        inlier_probability_updated = inlier_probability
        ransac_iteration_idx = 0
        while (not plausible_solution
               and ransac_iteration_idx < max_ransac_iterations
               and (max_combo_probability >= min_combo_probability
                    or ransac_iteration_idx < min_ransac_iterations)
               and sat_combinations.shape[0] > 0):

            # Calculate probability for each set to contain only inliers
            inlier_probability_combo = np.prod(
                inlier_probability_updated[sat_combinations], axis=-1)
            # Pick combo with highest probabiliy that has not been picked yet
            # Sort probabilities across all combos
            max_combo_probability_idx = np.argmax(inlier_probability_combo)
            chosen_sats = sat_combinations[max_combo_probability_idx]
            max_combo_probability = inlier_probability_combo[max_combo_probability_idx]
            if (max_combo_probability >= min_combo_probability
                    or ransac_iteration_idx < min_ransac_iterations):

                # Delete chosen combo
                sat_combinations = np.delete(sat_combinations,
                                             max_combo_probability_idx,
                                             axis=0)

                # Bring 1st satellite of combo to the front
                sat_idx = np.concatenate((
                    np.array([chosen_sats[0]]),
                    np.arange(0, chosen_sats[0]),
                    np.arange(chosen_sats[0]+1, numSVs)
                    ))

            # Assign N numbers:
            Ns = assign_integers(sats, sat_idx, obs, Eph,
                                 approx_distances, code_period_ms)

            Ks = Ns - Ns[0]

            fullPRs = Ns + obs  # Full pseudoranges reconstruction in ms
            fullPRs = fullPRs * (v_light * 1e-3)  # In meters

            # Preliminary guess for receiver position, common bias, and
            # assistance error [x y z b et]
            state = np.concatenate((rec_loc_assist, np.zeros(n_bias_opt+1)))

            for iteration_idx in np.arange(no_iterations):

                # Coarse-time error
                Et = state[-1]  # In seconds

                # Common bias [m]
                b = state[-1-n_bias_opt]

                if iteration_idx == no_iterations-1:
                    # Last iteration -> all satellites
                    Eph_it = Eph
                    T_tilde_it = T_tilde
                    Ks_it = Ks
                    fullPRs_it = fullPRs
                else:
                    # Other iteration -> selected satellites only
                    Eph_it = Eph[:, chosen_sats]
                    T_tilde_it = T_tilde[chosen_sats]
                    Ks_it = Ks[chosen_sats]
                    fullPRs_it = fullPRs[chosen_sats]

                tx_GPS, tcorr = tx_RAW2tx_GPS(T_tilde_it - Et - Ks_it * 1e-3,
                                              Eph_it)
                sat_pos, sat_vel = ep.get_sat_pos_vel(tx_GPS, Eph_it)

                if iteration_idx == 0:
                    traveltime = 0.072
                    rot_sat_pos = sat_pos
                else:
                    posECEF = state[:3]
                    rho2 = (sat_pos[:, 0] - posECEF[0])**2 \
                        + (sat_pos[:, 1] - posECEF[1])**2 \
                        + (sat_pos[:, 2] - posECEF[2])**2  # Distance squared
                    traveltime = np.sqrt(rho2) / v_light
                    rot_sat_pos = e_r_corr(traveltime, sat_pos, sat_pos.shape[0])

                # Subtraction of state[3] corrects for receiver clock offset and
                # v_light*tcorr is the satellite clock offset
                predictedPR = np.linalg.norm(rot_sat_pos - state[:3], axis=-1) \
                    + b - tcorr * v_light  # meters
                delta_z = fullPRs_it - predictedPR  # Meters
                # Receiver-satellite vector
                e_k = (rot_sat_pos
                       - np.tile(np.array([Et + Ks_it * 1e-3]).T, (1, 3)) * sat_vel
                       - state[:3])
                # Normalize receiver-satellite vector
                e_k = e_k / np.array([np.linalg.norm(e_k, axis=-1)]).T

                # Relative satellite velocity along this vector, i.e.,
                # project satellite velocity on normalized receiver-satellite vector
                v_k = np.sum(-sat_vel * e_k, axis=1, keepdims=True)

                # Now build Jacobian matrix H according to:
                # -e_k 1 v_k
                # Notice that it is easier to plug in the location of the satellite
                # at its T_dot estimation, i.e., rot_sat_pos
                # Matrix for 5 optimisation variables
                H = np.hstack((-e_k, np.ones((sat_pos.shape[0], 1)), v_k))

                if iteration_idx == no_iterations-1:
                    # Last iteration
                    # RANSAC: Select only max_sat satellites
                    delta_z_chosen = delta_z[chosen_sats]
                    H_chosen = H[chosen_sats]
                else:
                    delta_z_chosen = delta_z
                    H_chosen = H

                # Check if height measurement is provided
                if observed_height is not None:
                    # Add Jacobian of height observation
                    H_chosen = np.vstack((H_chosen, jh.jacobian_height(state)))
                    # Predict height based on current state
                    predicted_height = pm.ecef2geodetic(state[0], state[1], state[2]
                                                        )[2]
                    # Add height measurement
                    delta_z_chosen = np.append(delta_z_chosen, observed_height - predicted_height)

                x = np.linalg.lstsq(H_chosen, delta_z_chosen, rcond=None)[0]
                prev_state = state
                state = state + x

            # Plausibility check
            if max_dist is not None:
                dist_ok = np.linalg.norm(state[:3] - rec_loc_assist) < max_dist
            else:
                dist_ok = True
            if max_time is not None:
                time_ok = np.abs(state[-1]) < max_time
            else:
                time_ok = True
            if inlier_threshold is not None:
                residuals_ok = np.all(np.abs(delta_z_chosen) < inlier_threshold)
            else:
                residuals_ok = True
            if min_inliers is not None:
                n_inliers_ok = np.where(
                    np.abs(delta_z) < inlier_threshold
                    )[0].shape[0] >= min_inliers
            else:
                n_inliers_ok = True
            if residuals_ok and dist_ok and time_ok and n_inliers_ok:
                plausible_solution = True

            if not plausible_solution:
                if max_combo_probability < 1.0:
                    # Update inlier probabilities
                    for sat_idx in np.arange(numSVs):
                        if np.in1d(sat_idx, chosen_sats):
                            inlier_probability_updated[sat_idx] \
                                = (inlier_probability_updated[sat_idx]
                                   - max_combo_probability) \
                                / (1.0-max_combo_probability)

                ransac_iteration_idx += 1

    if plausible_solution:

        # Find inliers
        chosen_sats = np.where(np.abs(delta_z) < inlier_threshold)[0]

        # Select inliers (RANSAC)
        delta_z_chosen = delta_z[chosen_sats]
        H_chosen = H[chosen_sats]

        # Check if height measurement is provided
        if observed_height is not None:
            # Add Jacobian of height observation
            H_chosen = np.vstack((H_chosen, jh.jacobian_height(prev_state)))
            # Predict height based on current state
            predicted_height = pm.ecef2geodetic(state[0], state[1], state[2]
                                                )[2]
            # Add height measurement
            delta_z_chosen = np.append(delta_z_chosen, observed_height - predicted_height)

        # Update state based on all inliersprev_
        x = np.linalg.lstsq(H_chosen, delta_z_chosen, rcond=None)[0]
        state = state + x

        if hdop:
            try:
                Q = np.linalg.inv(H.T @ H)
                dilution_east_squared = Q[0, 0]
                dilution_north_squared = Q[1, 1]
                hdop = np.sqrt(dilution_east_squared + dilution_north_squared)
                return state, delta_z, hdop
            except:
                print("Cannot calculate HDOP.")
                return state, delta_z, np.nan

        return state, delta_z

    else:
        # Could not find plausible solution
        if hdop:
            return np.full(state.shape[0], np.inf), delta_z, np.nan
        return np.full(state.shape[0], np.inf), delta_z


state = {"n_failed": 0,
         "latitude": np.inf,
         "longitude": np.inf,
         "height": np.inf,
         "time_error": 0.0}


def positioning_simplified(snapshot_idx_dict,
                           prn_dict,
                           code_phase_dict,
                           snr_dict,
                           eph_dict,
                           utc,
                           latitude_init, longitude_init, height_init,
                           observed_heights=None,
                           pressures=None, temperatures=15.0,
                           ls_mode='ransac', mle=False, max_sat_count=10,
                           max_dist=50.0e3, max_time=30.0,
                           time_error=0.0,
                           search_space_time=2.0,
                           n_failed=0):
    """Positioning for snapper with non-linear least-squares or MLE.

    Inputs:
        snapshot_idx_dict - Dictionary from acquisition_simplified
        prn_dict - Dictionary from acquisition_simplified
        code_phase_dict - Dictionary from acquisition_simplified
        snr_dict - Dictionary from acquisition_simplified
        eph_dict - Dictionary from acquisition_simplified
        utc - Timestamps in UTC, one value for each snapshot, 1D NumPy array
              with elements of type numpy.datetime64
        latitude_init - Initial latitude for 1st snapshot [°]
        longitude_init - Initial longitude for 1st snapshot [°]
        height_init -  Initial height w.r.t. WGS84 for 1st snapshot [m]
        observed_heights - Measured heights w.r.t. WGS84 [m] , one value for
                           each snapshot, 1D NumPy array, default=None
        pressures - Measured pressureses [Pa], one value for each snapshot, 1D
                    NumPy array, default=None
        temperatures - Measured temperatures [°C], either one value for each
                       snapshot, 1D NumPy array, or a single value,
                       default=15.0
        ls_mode - Non-linear least-squares method:
                      'single' - Single non-linear least-sqaures run,
                      'snr' - Linear iterative satellite selection,
                      'combinatorial' - Combinatorial satellite selection,
                      'ransac' - RANSAC for satellite selection,
                      None - No non-linear least-squares,
                  default='ransac'
        mle - Enable maximum-likelihood estimation (MLE); if ls_mode=None and
              mle=True, then MLE is used for positioning; if ls_mode is a
              valid string and mle=True, then MLE is only used for positioning
              if least-squares cannot find a plausible solution, default=False
        max_sat_count - Maximum number of satellites to use for LS-single, LS-
                        snr, LS-combinatorial, and LS-ransac default=10
        max_dist - Maximum spatial distance between 2 consecutive fixes to be
                   plausible [m], default=50.0e3
        max_time - Maximum temporal distance between 2 consecutive fixes to be
                   plausible [s], i.e., the change of the coarse-time error,
                   default=30.0
        time_error - Coarse-time error of 1st snapshot [s], default=0.0
        n_failed - Number of position fixes that failed before 1st snapshot,
                   default=0

    Outputs:
        latitude_vec - Latitude estimate for each snapshot [°] or numpy.nan, 1D
                       NumPy array
        longitude_vec - Longitude estimate for each snapshot [°] or numpy.nan,
                        1D NumPy array
        utc_vec - Corrected timestamps in UTC for each snapshot 1D NumPy array
                  with elements of type numpy.datetime64
        uncertainty_vec - Horizontal uncertainty estimate for each snapshot [m]
                          or np.inf, 1D NumPy array

    Author: Jonas Beuchert
    """
    from itertools import combinations
    global state

    if ls_mode is not None and ls_mode != "single" and ls_mode != "snr" \
         and ls_mode != "combinatorial" and ls_mode != "ransac":
        raise Exception(
            """Least-squares method ls_mode='{}' not recognized. Use None,
            'single', 'snr', 'combinatorial', or 'ransac'.""".format(ls_mode)
            )
    if ls_mode is None and not mle:
        raise Exception(
            """ls_mode=None and mle=False. Please select either a valid value
            for ls_mode that is not None or set mle=True."""
            )
    if pressures is not None and temperatures is None:
        raise Warning(
            """Pressures provided, but no temperature(s).
            Will not use pressures for height estimation."""
            )
    # if type(snapshot_idx_dict) != dict or type(prn_dict) != dict \
    #      or type(code_phase_dict) != dict or type(snr_dict) != dict \
    #      or type(tow_vec) != dict or type(eph_dict) != dict:

    # Check which GNSS are present
    gnss_list = snapshot_idx_dict.keys()

    # How many snapshots?
    n_snapshots = utc.shape[0]

    # Convert UTC to GPS time
    reference_date = np.datetime64('1980-01-06')  # GPS reference date
    leap_seconds = np.timedelta64(18, 's')  # Hardcoded 18 leap seconds
    time = (utc - reference_date + leap_seconds) / np.timedelta64(1, 's')

    # Convert GPS time to BeiDou time
    time = {gnss: time - 820108814.0 if gnss == 'C' else time
            for gnss in gnss_list}

    # Absolute system time to time of week (TOW)
    tow_dict = {gnss: np.mod(time[gnss], 7 * 24 * 60 * 60)
                for gnss in gnss_list}

    # Initialize outputs
    latitude_vec = np.full(n_snapshots, np.nan)
    longitude_vec = np.full(n_snapshots, np.nan)
    time_error_vec = np.zeros(n_snapshots)
    uncertainty_vec = np.full(n_snapshots, np.inf)

    # Index of last successful fix
    last_good_idx = None

    # Convert geodetic coordinates to Cartesian ECEF XYZ coordinates
    pos_init = np.array(pm.geodetic2ecef(
        latitude_init, longitude_init, height_init
        ))

    # Algorithm specific initializations
    if ls_mode is not None:
        # Coarse-time navigation using least-squares

        # Load Bayes classifier to rate satellite reliability based on SNR
        classifier = np.load("bayes_snapper.npy", allow_pickle=True).item()

        # Estimate (log) probability that satellite is useful based on SNR
        reliability_vec = {gnss: classifier[gnss].predict_log_proba(
            snr_dict[gnss].reshape(-1, 1))[:, 1] for gnss in gnss_list}

        # Greatest common divider of code periods
        if 'G' in gnss_list or 'S' in gnss_list:
            code_period_ms = 1
        elif 'E' in gnss_list and 'C' in gnss_list:
            code_period_ms = 2
        elif 'E' in gnss_list:
            code_period_ms = 4
        else:  # 'C'
            code_period_ms = 10

    if pressures is not None and temperatures is not None:
        # Make sure that measurements are reasonable
        pressures = np.clip(pressures, 87000.0, 108400.0)
        temperatures = np.clip(temperatures, -89.2, 56.7)
        # Height change w.r.t. last measurement from temp. & pressure
        height_change = ep.get_relative_height_from_pressure(
            measured_pressure=pressures[1:],
            reference_pressure=pressures[:-1],
            hypsometric=True,
            temperature=(temperatures[1:] if type(temperatures)==np.ndarray else temperatures)+273.15
            )

    # Loop over all snapshots
    for snapshot_idx in np.arange(n_snapshots):

        if pressures is not None and temperatures is not None \
             and last_good_idx is not None:
            # Height change w.r.t. last measurement from temp. & pressure
            pos_init = np.array(pm.enu2ecef(
                0.0, 0.0, np.sum(height_change[last_good_idx:snapshot_idx]),
                latitude_init, longitude_init, height_init
                ))

        # Merge individual GNSS for each snapshot
        tow = np.concatenate([
            tow_dict[gnss][snapshot_idx]*np.ones(
                np.where(snapshot_idx_dict[gnss] == snapshot_idx)[0].shape[0]
                )
            for gnss in gnss_list
            ])
        eph = np.concatenate([
            eph_dict[gnss][:, snapshot_idx_dict[gnss] == snapshot_idx]
            for gnss in gnss_list
            ], axis=1)
        prn = np.concatenate([
            prn_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
            for gnss in gnss_list
            ])

        # Iterate until plausible solution is found
        plausible_solution = False

        if ls_mode is not None:
            # Coarse-time navigation using least-squares

            # Merge individual GNSS for each snapshot
            code_phases = np.concatenate([
                code_phase_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
                for gnss in gnss_list
                ])
            reliability = np.concatenate([
                reliability_vec[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
                for gnss in gnss_list
                ])

            # Number of satellites that are potentially visible
            n_sats = code_phases.shape[0]
            # Sort satellites according to SNR-induced reliability likelihood
            sorted_idx = np.argsort(reliability)[::-1]

            if ls_mode == 'single':
                # Use only satellites with highest reliability
                # Use exactly max_sat_count satellites, if possible
                sat_combinations = [sorted_idx[:min(max_sat_count, n_sats)]]
            elif ls_mode == 'snr':
                # Use only satellites with highest reliability
                # Use max_sat_count satellites or less
                sat_combinations = [
                    sorted_idx[:max_idx]
                    for max_idx in np.arange(min(max_sat_count, n_sats), 0,
                                             step=-1)
                    ]
            elif ls_mode == 'combinatorial':
                # Use only satellites with highest reliability
                # Use all combinations of those satellites
                sat_combinations = [
                    sorted_idx[list(idx)]
                    for max_idx in np.arange(min(max_sat_count, n_sats), 0,
                                             step=-1)
                    for idx in list(combinations(
                            range(min(max_sat_count, n_sats)), max_idx))
                    ]
            elif ls_mode == 'ransac':
                # Use all satellites
                # Let RANSAC decide which ones are useful
                # sat_combinations = [sorted_idx]
                sat_combinations = [sorted_idx[:max_sat_count]]

            # Project all code phases onto the same range
            code_phases_mod = np.mod(code_phases, code_period_ms)

            # Iterate until plausible solution is found
            # or all combos are exhausted
            sat_combinations = iter(sat_combinations)
            iterator_empty = False
            while not plausible_solution and not iterator_empty:
                try:
                    # Get next unexamined combo
                    combo = next(sat_combinations)
                    try:
                        # Coarse-time navigation
                        if ls_mode != 'ransac':
                            pos, res, hdop = coarse_time_nav_simplified(
                                code_phases_mod[combo],
                                prn[combo],
                                eph[:, combo],
                                (tow[combo] - time_error)*1e3,
                                pos_init,
                                observed_height=observed_heights[snapshot_idx] if observed_heights is not None else None,
                                code_period_ms=code_period_ms, hdop=True,
                                no_iterations=3)
                        else:
                            # pos, res, _, hdop = coarse_time_nav_ransac(
                            #     code_phases_mod[combo],
                            #     prn[combo],
                            #     eph[:, combo],
                            #     (tow[combo] - time_error)*1e3,
                            #     pos_init,
                            #     observed_height=observed_heights[snapshot_idx] if observed_heights is not None else None,
                            #     code_period_ms=code_period_ms,
                            #     tropo=None,
                            #     inlier_probability=np.exp(reliability[combo]),
                            #     min_ransac_iterations=1,
                            #     max_ransac_iterations=3,
                            #     min_combo_probability=0.0,
                            #     inlier_threshold=200,
                            #     min_inliers=None,
                            #     hdop=True,
                            #     max_dist=max_dist, max_time=max_time,
                            #     no_iterations=3)
                            pos, res, hdop = coarse_time_nav_ransac_simplified(
                                code_phases_mod[combo],
                                prn[combo],
                                eph[:, combo],
                                (tow[combo] - time_error)*1e3,
                                pos_init,
                                observed_height=observed_heights[snapshot_idx] if observed_heights is not None else None,
                                code_period_ms=code_period_ms,
                                inlier_probability=np.exp(reliability[combo]),
                                min_ransac_iterations=1,
                                max_ransac_iterations=3,
                                min_combo_probability=0.0,
                                inlier_threshold=200,
                                min_inliers=None,
                                hdop=True,
                                no_iterations=3)
                        if np.isnan(hdop):
                            hdop = np.inf
                        # Check plausibility of solution by checking pseudorange
                        # residuals and distance to last plausible solution
                        if (ls_mode == 'ransac' or np.all(np.abs(res) < 200)) \
                            and np.linalg.norm(pos[:3] - pos_init) < max_dist*(1+n_failed) \
                                and np.abs(pos[-1]) < max_time*(1+n_failed):
                            plausible_solution = True
                    except ValueError:
                        print("CTN failed due to value error.")
                        hdop = np.inf

                except StopIteration:
                    # Exhausted all combos
                    iterator_empty = True
                    hdop = np.inf

        if mle and (ls_mode is None or hdop * 20.0 > 200.0 or not plausible_solution):

            try:

                # Coarse-time navigation by gradient-based pseudo-likelihood optimization
                pos, useful = coarse_time_nav_mle(
                    init_pos=pos_init,
                    init_time=[
                        time[gnss][snapshot_idx]-time_error
                        for gnss in gnss_list
                        ],
                    code_phase=[
                        code_phase_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
                        for gnss in gnss_list
                        ],
                    vis=[
                        prn_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
                        for gnss in gnss_list
                        ],
                    eph=[
                        eph_dict[gnss][:, snapshot_idx_dict[gnss] == snapshot_idx]
                        for gnss in gnss_list
                        ],
                    peak_height=[
                        snr_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
                        for gnss in gnss_list
                        ],
                    observed_height=observed_heights[snapshot_idx] if observed_heights is not None else np.nan,
                    code_period=np.array([
                        {'G': 1e-3, 'S': 1e-3, 'E': 4e-3, 'C': 10e-3}[gnss]
                        for gnss in gnss_list
                        ]),
                    search_space_pos=np.array([20.0e3, 20.0e3, 0.2e3]),
                    search_space_time=search_space_time,
                    hard_constraint=False,
                    linear_pr_prediction=True,
                    trop=False, iono=False,
                    time_out=2, optim_opt=0,
                    std=2.0**np.arange(5, -7, -2) * 8.3333e-07
                    )

                # Calculate HDOP
                # Get indices of useful PRNs
                useful_idx = np.concatenate([
                    np.in1d(
                        prn_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx],
                        useful[gnss_idx]
                        ) for gnss_idx, gnss in enumerate(gnss_list)
                    ])

            except ValueError:

                print("MLE failed due to value error.")

                useful_idx = np.array([])

            if np.any(useful_idx):
                # Use only satellites that are considered useful by MLE to
                # calculate HDOP
                hdop = horizontal_dilution_of_precision(
                    prn[useful_idx],
                    eph[:, useful_idx],
                    tow[useful_idx],
                    pos[:3]
                    )
                if np.isnan(hdop):
                    hdop = np.inf
            else:
                hdop = np.inf

            # Check if uncertainty is small
            if hdop * 20.0 <= 200.0:
                plausible_solution = True

        # Check if plausible solution was found
        if plausible_solution:
            # Reset counter how many fixes failed in a row
            n_failed = 0
            # Remember this one as last plausible fix
            last_good_idx = snapshot_idx
            # Extract position estimate and remeber it for next snapshot
            pos_init = pos[:3]
            # Convert Cartesian ECEF XYZ coordinates to geodetic ones
            latitude_init, longitude_init, height_init \
                = pm.ecef2geodetic(pos[0], pos[1], pos[2])
            latitude_vec[snapshot_idx] = latitude_init
            longitude_vec[snapshot_idx] = longitude_init
            # Extract coarse-time error, remember it for next snapshot
            time_error += pos[-1]
            time_error_vec[snapshot_idx] = time_error
            # Estimate uncertainty from measurement uncertainty and HDOP
            uncertainty_vec[snapshot_idx] = 20.0 * hdop
        else:
            # One more consecutive fix failed
            n_failed += 1
            # Convert Cartesian ECEF XYZ coordinates to geodetic ones
            latitude_vec[snapshot_idx], longitude_vec[snapshot_idx], _ \
                = pm.ecef2geodetic(pos[0], pos[1], pos[2])
            # Estimate uncertainty from measurement uncertainty and HDOP
            uncertainty_vec[snapshot_idx] = np.inf  # 20.0 * hdop

    # Subtract coarse-time error from timestamps (millisecond precision)
    utc_vec = utc - (1000*time_error_vec).astype('timedelta64[ms]')

    state = {"n_failed": n_failed,
             "latitude": latitude_init,
             "longitude": longitude_init,
             "height": height_init,
             "time_error": time_error}

    return latitude_vec, longitude_vec, utc_vec, uncertainty_vec
