"""Approximate pseudorange prediction.

Created on Sun Oct 11 2020

@author: Jonas Beuchert
"""
try:
    import autograd.numpy as np
except(ImportError):
    print("""Package 'autograd' not found. 'autograd.numpy' is necessary for
          coarse-time navigation via maximum-likelihood estimation. Falling
          back to 'numpy'.""")
    import numpy as np
import pymap3d as pm
import eph_util as ep


class PseudorangePrediction:
    """Approximate pseudorange prediction.

    Pseudorange prediction in two steps:
        1) initialisation and
        2) prediction via either non-linear or linear approximation.

    Author: Jonas Beuchert
    """

    def __init__(self, sats, eph, coarse_time, rec_pos, common_bias=0.0,
                 trop=False, atm_pressure=1013.0, surf_temp=293.0,
                 humidity=50.0, iono=None, ion_alpha=np.array([]),
                 ion_beta=np.array([]), poly_degree=None):
        """Initialize approximate pseudorange prediction.

        Inputs:
            sats - Indices of satellites (PRNs)
            eph - Ephemeris as matrix
            coarse_time - Coarse GPS time [s]
            rec_pos - Receiver position in ECEF XYZ coordinates [m,m,m]
            common_bias - Common bias in all pseudoranges [m]; default=0.0
            trop - Model for troposheric correction: either None or False
                   for no correction, 'gaod' or True for the model of C. C.
                   Goad and L. Goodman, 'hopfield' for the model of H. S.
                   Hopfield, or 'tsui' for the model of J. B.-Y. Tsui
                   [default=False]
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
            poly_degree - Polynomial degree for non-linear satellite position
                          approximation (should be low, e.g., 2), default=None

            Output:
                self - Initialized pseudorange-prediction object

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
            col = np.array([ep.find_eph(eph, s_i, coarseTimeTOW)
                            for s_i in sats])
            if col.size == 0:
                raise IndexError("Cannot find satellite in navigation data.")
            # Extract matching columns
            eph = eph[:, col]

        # Find satellite positions at coarse transmission time
        txGPS = coarseTimeTOW - ep.get_sat_clk_corr(coarseTimeTOW, sats, eph)
        satPosCoarse = ep.get_sat_pos(txGPS, eph)

        # Find closest one (alternatively, find highest)
        distancesCoarse = np.sqrt(np.sum((rec_pos - satPosCoarse)**2, axis=-1))
        # satByDistance = np.argsort(distancesCoarse)

        # Assign integer ms-part of distances
        # Ns = np.zeros(nSats)
        # Time equivalent to distance [ms]
        # distancesCoarse = distancesCoarse / c / 1e-3
        travel_times_coarse = distancesCoarse / c
        # Index of 1st satellite (reference satellite)
        # N0Inx = satByDistance[0]
        # Initial guess
        # Ns[N0Inx] = np.floor(distancesCoarse[N0Inx])
        # Time error [ms]
        # deltaT = eph[18] * 1e3
        # Update considering time error
        # for i in range(1, nSats):
        #     k = satByDistance[i]
        #     Ns[k] = np.round(Ns[N0Inx] + (distancesCoarse[k] - deltaT[k])
        #                      - (distancesCoarse[N0Inx] - deltaT[N0Inx]))

        # Find integer ms-part difference to reference satellite
        # Ks = Ns - Ns[N0Inx]
        rel_travel_times = travel_times_coarse - np.min(travel_times_coarse)

        # Correct for satellite clock error
        tCorr = np.empty(nSats)
        for i in range(nSats):
            k = np.array([sats[i]])
            # tCorr[i] = ep.get_sat_clk_corr(coarseTimeTOW - Ks[i] * 1e-3, k,
            #                                eph[:, i, np.newaxis])
            tCorr[i] = ep.get_sat_clk_corr(coarseTimeTOW - rel_travel_times[i],
                                           k, eph[:, i, np.newaxis])
        txGPS = coarseTimeTOW - rel_travel_times - tCorr

        # Get satellite position at corrected transmission time
        satPos = ep.get_sat_pos(txGPS, eph)

        # Calculate rough propagation delay
        travelTime = np.linalg.norm(satPos - rec_pos, axis=1) / c

        # Initialize array
        rotSatPos = np.empty((nSats, 3))

        for i in range(nSats):
            # k = satByDistance[i]

            # Rotate satellite ECEF coordinates due to earth rotation during
            # signal travel time
            OmegaEdot = 7.292115147e-5  # Earth's angular velocity [rad/s]
            omegaTau = OmegaEdot * travelTime[i]  # Angle [rad]
            R3 = np.array([[np.cos(omegaTau), np.sin(omegaTau), 0.0],
                           [-np.sin(omegaTau), np.cos(omegaTau), 0.0],
                           [0.0, 0.0, 1.0]])  # Rotation matrix
            rotSatPos[i] = R3 @ satPos[i]  # Apply rotation

        trop = self.__atmospheric_correction(
            trop, atm_pressure, surf_temp, humidity, iono, ion_alpha, ion_beta,
            nSats, rec_pos, rotSatPos, coarseTimeTOW, c)

        # Correct for common bias, satellite clock offset, tropospheric delay
        predictedPR = (np.linalg.norm(rotSatPos - rec_pos, axis=1)
                       + common_bias - tCorr*c + trop)  # [m]

        if (poly_degree is None or
           poly_degree == np.nan or poly_degree < 0 or poly_degree == np.inf):
            poly_degree = None
        if poly_degree is not None:
            # Get points in time -/+ 60 s around coarse time to fit polynom
            t_vec = coarseTimeTOW + np.linspace(-60, 60, poly_degree+1)
            # Get satellite positions at all points in time in interval
            sat_pos_vec = np.array([
                ep.get_sat_pos(t_vec, eph[:, sat_idx])
                for sat_idx in range(nSats)
                ])
            # Fit polynoms for each satellite in all 3 dimensions
            self.poly_coeff = np.array([np.array([
                np.polyfit(t_vec, sat_pos_vec[sat_idx, :, dim_idx],
                           deg=poly_degree)
                for dim_idx in range(3)]) for sat_idx in range(nSats)])

        # Memorize values for later
        self.tCorr = tCorr
        self.recPos = rec_pos
        self.trop = trop
        self.predictedPR = predictedPR
        self.nSats = nSats
        # self.Ks = Ks
        self.rel_travel_times = rel_travel_times
        self.eph = eph
        self.coarseTime = coarse_time
        self.commonBias = common_bias
        self.poly_degree = poly_degree

        # Calculate Jacobian
        dPrT = (self.predict_approx(coarse_time+1.0, rec_pos, common_bias)
                - self.predict_approx(coarse_time-1.0, rec_pos, common_bias)
                ) / 2.0
        dPrX = (self.predict_approx(coarse_time,
                                    np.array([rec_pos[0]+1000.0, rec_pos[1],
                                              rec_pos[2]]),
                                    common_bias)
                - self.predict_approx(coarse_time,
                                      np.array([rec_pos[0]-1000.0,
                                                rec_pos[1], rec_pos[2]]),
                                      common_bias)) / 2000.0
        dPrY = (self.predict_approx(coarse_time,
                                    np.array([rec_pos[0], rec_pos[1]+1000.0,
                                              rec_pos[2]]), common_bias)
                - self.predict_approx(coarse_time,
                                      np.array([rec_pos[0], rec_pos[1]-1000.0,
                                                rec_pos[2]]), common_bias)
                ) / 2000.0
        dPrZ = (self.predict_approx(coarse_time,
                                    np.array([rec_pos[0], rec_pos[1],
                                              rec_pos[2]+1000.0]),
                                    common_bias)
                - self.predict_approx(coarse_time,
                                      np.array([rec_pos[0], rec_pos[1],
                                                rec_pos[2]-1000.0]),
                                      common_bias)) / 2000.0
        dPrB = np.ones(nSats)
        self.grad = np.array([dPrT, dPrX, dPrY, dPrZ, dPrB]).T

    def predict_approx(self, coarse_time, rec_pos, common_bias):
        """Approximately predict pseudoranges to satellites.

        Inputs:
          coarse_time - Coarse GPS time [s]
          rec_pos - Receiver position in ECEF XYZ coordinates [m,m,m]
          common_bias - Common bias in all pseudoranges [m]

        Output:
          predictedPR - Predicted pseudoranges [m]

        Author: Jonas Beuchert
        Algorithm from Chapter 4.4.2 of
        F. van Diggelen, A-GPS: Assisted GPS, GNSS, and SBAS, 2009.
        """
        # Speed of light [m/s]
        c = 299792458.0

        # GPS time since 1980 to time of week (TOW) [s]
        coarseTimeTOW = np.mod(coarse_time, 7 * 24 * 60 * 60)

        # Correct for satellite clock error
        txGPS = coarseTimeTOW - self.rel_travel_times - self.tCorr

        # Get satellite position at corrected transmission time
        if self.poly_degree is None:
            satPos = ep.get_sat_pos(txGPS, self.eph)
        else:
            time = np.array([
                txGPS**deg for deg in range(self.poly_degree, -1, -1)
            ])
            satPos = np.array([np.array([
                 np.matmul(self.poly_coeff[sat_idx, dim_idx], time[:, sat_idx])
                 for dim_idx in range(3)]) for sat_idx in range(self.nSats)])

        # Calculate rough propagation delay
        travelTime = np.linalg.norm(satPos - rec_pos, axis=1) / c

        # Rotate satellite ECEF coordinates due to earth rotation during
        # signal travel time
        OmegaEdot = 7.292115147e-5  # Earth's angular velocity [rad/s]
        omegaTau = OmegaEdot * travelTime  # Angle [rad]
        R3 = np.array(
            [np.array([[np.cos(omegaTau[k]), np.sin(omegaTau[k]), 0.0],
                       [-np.sin(omegaTau[k]), np.cos(omegaTau[k]), 0.0],
                       [0.0, 0.0, 1.0]]) for k in range(self.nSats)]
            )  # Rotation matrix
        rotSatPos = np.array([np.matmul(R3[k], satPos[k])
                              for k in range(self.nSats)])

        # # Initialize array
        # rotSatPos = np.empty((self.nSats, 3))

        # for k in range(self.nSats):

        #     # Rotate satellite ECEF coordinates due to earth rotation during
        #     # signal travel time
        #     OmegaEdot = 7.292115147e-5  # Earth's angular velocity [rad/s]
        #     omegaTau = OmegaEdot * travelTime[k]  # Angle [rad]
        #     R3 = np.array([[np.cos(omegaTau), np.sin(omegaTau), 0.0],
        #                    [-np.sin(omegaTau), np.cos(omegaTau), 0.0],
        #                    [0.0, 0.0, 1.0]])  # Rotation matrix
        #     rotSatPos[k] = np.matmul(R3, satPos[k])  # Apply rotation

        # Correct for common bias, satellite clock offset, tropospheric delay
        return (np.linalg.norm(rotSatPos - rec_pos, axis=1) + common_bias
                - self.tCorr*c + self.trop)  # [m]

    def predict_linear(self, coarse_time, rec_pos, common_bias):
        """Approximately predict pseudoranges using linearization.

        Inputs:
          coarse_time - Coarse GPS time [s]
          rec_pos - Receiver position in ECEF XYZ coordinates [m,m,m]
          common_bias - Common bias in all pseudoranges [m]

        Output:
          predictedPR - Predicted pseudoranges [m]

        Author: Jonas Beuchert
        """
        deltaT = coarse_time - self.coarseTime
        deltaP = rec_pos - self.recPos
        deltaB = common_bias - self.commonBias

        delta = np.array([deltaT, deltaP[0], deltaP[1], deltaP[2], deltaB])
        return self.predictedPR + np.dot(self.grad, delta)

    def __atmospheric_correction(self, tropo, atm_pressure, surf_temp,
                                 humidity, iono, ion_alpha, ion_beta, nSats,
                                 rec_pos, rotSatPos, coarseTimeTOW, c):
        """Correct for tropospheric and ionospheric delay."""
        # Initialize array
        trop = np.zeros(nSats)

        # Check if tropospheric correction shall be applied
        if tropo is True or tropo == 'goad' or tropo == 'hopfield' or tropo == 'tsui':

            for i in range(nSats):

                # Select model for correction
                if tropo is True or tropo == 'goad':
                    # Transform into topocentric coordinate system
                    az, el, dist = ep.topocent(rec_pos, rotSatPos[i] - rec_pos)
                    # Elevation of satellite w.r.t. receiver [deg]
                    # Tropospheric correction
                    trop[i] = ep.tropo(np.sin(el * np.pi/180.0), 0.0,
                                       atm_pressure, surf_temp, humidity, 0.0,
                                       0.0, 0.0)
                elif tropo == 'hopfield':
                    surf_temp_celsius = surf_temp-273.15
                    saturation_vapor_pressure = 6.11*10.0**(
                        7.5*surf_temp_celsius/(237.7+surf_temp_celsius))
                    vapor_pressure = humidity/100.0 * saturation_vapor_pressure
                    trop[i] = ep.tropospheric_hopfield(
                        rec_pos, np.array([rotSatPos[i]]), surf_temp_celsius,
                        atm_pressure/10.0, vapor_pressure/10.0)
                elif tropo == 'tsui':
                    lat, lon, h = pm.ecef2geodetic(rec_pos[0], rec_pos[1],
                                                   rec_pos[2])
                    az, el, srange = pm.ecef2aer(rotSatPos[i][0],
                                                 rotSatPos[i][1],
                                                 rotSatPos[i][2],
                                                 lat, lon, h)
                    trop[i] = ep.tropospheric_tsui(el)

        # Check if ionospheric correction shall be applied
        if iono == 'klobuchar' or iono == 'tsui':

            for i in range(nSats):

                # Select model for ionospheric correction
                if iono == 'klobuchar':
                    trop[i] = trop[i] + ep.ionospheric_klobuchar(
                        rec_pos, np.array([rotSatPos[i]]),
                        np.mod(coarseTimeTOW, 24*60*60),
                        ion_alpha, ion_beta) * c
                elif iono == 'tsui':
                    lat, lon, h = pm.ecef2geodetic(rec_pos[0], rec_pos[1],
                                                   rec_pos[2])
                    az, el, srange = pm.ecef2aer(rotSatPos[i][0],
                                                 rotSatPos[i][1],
                                                 rotSatPos[i][2], lat, lon, h)
                    # Convert degrees to semicircles
                    el = el / 180.0
                    az = az / 180.0
                    lat = lat / 180.0
                    lon = lon / 180.0
                    # Ionospheric delay [s]
                    T_iono = ep.ionospheric_tsui(
                        el, az, lat, lon, coarseTimeTOW, ion_alpha,
                        ion_beta)
                    trop[i] = trop[i] + T_iono * c

        return trop
