# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:40:15 2020

@author: Jonas Beuchert
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as so
import pymap3d as pm
import eph_util as ep


def estimate_frequency_bias_wrapper(
        pos_geo, utc, signal, intermediate_frequency, sampling_frequency,
        gps_file=None, sbas_file=None, galileo_file=None, beidou_file=None,
        plot=False):
    """
    Estimate common frequency bias w.r.t. intermediate frequency.

    Assume intermediate frequency of zero.

    Inputs:
        pos_geo - Coarse receiver position in geodetic coordinates [deg,deg,m]
                  (numpy array)
        utc - Universal coordinated time (numpy datetime64)
        signal - Binary GNSS raw signal (numpy array)
        intermediate_frequency - Uncorrected intermediate frequency [Hz]
        gps_file - Path to GPS RINEX navigation file, default=None
        sbas_file - Path to SBAS RINEX navigation file, default=None
        galileo_file - Path to Galileo RINEX navigation file, default=None
        beidou_file - Path to BeiDou RINEX navigation file, default=None
        plot - Switch to enable likelihood plotting, default=False
    Output:
        frequency_bias - Frequency bias estimate [Hz]

    Author: Jonas Beuchert
    """
    # Subtract mean from signal
    signal = signal.astype(float)
    signal = signal - np.mean(signal)
    # Convert geodetic coordinates to ECEF
    pos_ecef = np.empty(3)
    pos_ecef[0], pos_ecef[1], pos_ecef[2] = pm.geodetic2ecef(
        pos_geo[0], pos_geo[1], pos_geo[2]
        )
    # Convert UTC to GPS time
    gps_time = ep.utc_2_gps_time(utc)
    # Read RINEX navigation files for all systems
    eph = []  # List for navigation data
    time = []  # List for coarse time in different system times
    system = []  # List for GNSS strings
    prn = []  # List for PRNs of satellite constellations
    try:
        print("Read GPS navigation data...")
        eph.append(ep.rinexe(gps_file, 'G'))
        time.append(gps_time)
        system.append('gps')
        prn.append(np.arange(1, 33))
    except:
        print("Could not read GPS navigation data.")
        pass
    try:
        print("Read SBAS navigation data...")
        eph.append(ep.rinexe(sbas_file, 'S'))
        time.append(gps_time)
        system.append('sbas')
        prn.append(np.arange(120, 139))
    except:
        print("Could not read SBAS navigation data.")
        pass
    try:
        print("Read Galileo navigation data...")
        eph.append(ep.rinexe(galileo_file, 'E'))
        time.append(gps_time)
        system.append('galileo')
        prn.append(np.arange(1, 51))
    except:
        print("Could not read Galileo navigation data.")
        pass
    try:
        print("Read BeiDou navigation data...")
        eph.append(ep.rinexe(beidou_file, 'C'))
        beidou_time = ep.gps_time_2_beidou_time(gps_time)
        time.append(beidou_time)
        system.append('beidou')
        prn.append(np.arange(1, 64))
    except:
        print("Could not read BeiDou (BDS) navigation data.")
        pass
    n_gnss = len(eph)
    print("{} navigation satellite systems found.".format(n_gnss))
    # Acqusition
    vis = []
    frequencies = []
    snr = []
    for i_gnss in range(n_gnss):
        # Predict visible satellites
        vis.append(ep.get_visible_sats(time[i_gnss], pos_geo, eph[i_gnss],
                                       elev_mask=10, prn_list=prn[i_gnss]))
        print("{} satellites of system {} expected to be visible.".format(
            len(vis[i_gnss]), i_gnss+1))
        print("Acquire satellites for system {}...".format(i_gnss+1))
        # Acquisition
        _, _, _, _, _, f, _, p = ep.acquisition(
            signal, intermediate_frequency, sampling_frequency, freq_step=20,
            ms_to_process=12, prn_list=vis[i_gnss], fine_freq=False,
            gnss=system[i_gnss]
            )
        vis_idx = (vis[i_gnss]).astype(int) - 1
        frequencies.append(f[vis_idx])
        snr.append(p[vis_idx])
    print("Estimate frequency bias...")
    return estimate_frequency_bias(pos_ecef, time, frequencies, vis, eph, snr,
                                   plot=plot)


def estimate_frequency_bias_minimal_wrapper(
        snapshot_idx_dict,
        prn_dict,
        frequency_dict,
        snr_dict,
        eph_dict,
        utc,
        latitude, longitude, height
        ):
    """Estimate common frequency bias for a batch of snapshots.

    Assume intermediate frequency of zero.

    Inputs:
        snapshot_idx_dict - Dictionary from acquisition_simplified
        prn_dict - Dictionary from acquisition_simplified
        frequency_dict - Dictionary from acquisition_simplified
        snr_dict - Dictionary from acquisition_simplified
        eph_dict - Dictionary from acquisition_simplified
        utc - Timestamps in UTC, one value for each snapshot, 1D NumPy array
              with elements of type numpy.datetime64
        latitude_init - Single latitude for all snapshots [°]
        longitude_init - Single longitude for all snapshots [°]
        height_init -  Single height w.r.t. WGS84 for all snapshots [m]
    Output:
        frequency_bias_vec - Frequency bias estimates [Hz], 1D NumPy array

    Author: Jonas Beuchert
    """
    # Check which GNSS are present
    gnss_list = snapshot_idx_dict.keys()

    # How many snapshots?
    n_snapshots = utc.shape[0]

    # Convert UTC to GPS time
    reference_date = np.datetime64('1980-01-06')  # GPS reference date
    leap_seconds = np.timedelta64(18, 's')  # Hardcoded 18 leap seconds
    time = (utc - reference_date + leap_seconds) / np.timedelta64(1, 's')

    # Convert GPS time to BeiDou time
    time_dict = {gnss: time - 820108814.0 if gnss == 'C' else time
                 for gnss in gnss_list}

    # Convert geodetic coordinates to Cartesian ECEF XYZ coordinates
    # Same position for all snapshots
    pos_ecef = np.array(pm.geodetic2ecef(
       latitude, longitude, height
        ))

    # Loop over all snapshots
    return np.array([
        estimate_frequency_bias(
            init_pos=pos_ecef,
            init_time=[
                time_dict[gnss][snapshot_idx]
                for gnss in gnss_list
                ],
            observed_frequencies=[
                frequency_dict[gnss][snapshot_idx_dict[gnss] == snapshot_idx]
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
            plot=False
            )
        for snapshot_idx in np.arange(n_snapshots)])


def estimate_frequency_bias(init_pos, init_time, observed_frequencies, vis,
                            eph, peak_height, max_frequency_bias=2.5e3,
                            band_width=5.0e3, plot=False):
    """
    Estimate common frequency bias w.r.t. intermediate frequency.

    Assume intermediate frequency of zero.

    Inputs:
        init_pos - Coarse receiver position in ECEF XYZ coordinates [m,m,m]
        init_time - Coarse absolute GNSS times [s], list with one value for
                    each GNSS
        observed_frequencies - Observed carrier frequencies of satellites that
                               are expected to be visible [Hz]; list of numpy
                               arrays, one for each GNSS
        vis - PRNs of satellites that are expected to be visible, list of numpy
              arrays, one for each GNSS
        eph - Matching navigation data, list of 21-row 2D numpy arrays
        peak_height - Reliability measures for satellites that are expected to
                      be visible, e.g., SNR [dB], list of numpy arrays
        max_frequency_bias - Maximum absolute frequency bias [Hz],
                             default=2.5e3
        band_width - Original width of frequency search space during
                     acquistion [Hz], default=5.0e3
        plot - Switch to enable likelihood plotting, default=False
    Output:
        frequency_bias - Frequency bias estimate [Hz]

    Author: Jonas Beuchert
    """
    def bayes_classifier(x, galileo=False):
        """Probability of code phase being invalid or valid based on SNR.

        Inputs:
            x - Signal-to-noise ratio (SNR) array [dB]
            galileo - Type of GNSS, GPS (galileo=False) or Galileo or BeiDou
                      (galileo=True), default=False
        Output:
            p - 2xN array with probabilities for code phases being invalid
                (1st row) or valid (2nd row)

        Author: Jonas Beuchert
        """
        # if not galileo:  # GPS
        # Class means
        mu0 = 7.7
        mu1 = 15.1
        # Class standard deviations
        sigma0 = 1#0.67 * 4
        sigma1 = 1#4.65
        # Class priors
        p0 = 0.5#27
        p1 = 0.5#73
        # else:  # Galileo
        #     # Class means
        #     mu0 = 14.2
        #     mu1 = 19.1
        #     # Class standard deviations
        #     sigma0 = 1.13
        #     sigma1 = 4.12
        #     # Class priors
        #     p0 = 0.62
        #     p1 = 0.38
        px0 = ss.norm(mu0, sigma0).pdf(x) * p0
        px1 = ss.norm(mu1, sigma1).pdf(x) * p1
        return np.array([px0, px1]) / (px0 + px1)

    def neg_log_likelihood(
            bias, std, predicted_frequencies, observed_frequencies, pValid,
            band_width):

        # Initialize log-likelihood
        cd = 0.0

        # Define Gaussian kernel
        kernel = ss.norm(0.0, std)

        for i_gnss in range(n_gnss):
            n_sat = np.int(predicted_frequencies[i_gnss].shape[0] / 2)
            idx = predicted_frequencies[i_gnss] \
                + bias * np.concatenate((-np.ones(n_sat), np.ones(n_sat))) \
                - observed_frequencies[i_gnss]
            # Likelihood of frequency observation given that it is valid
            pPhiValid = kernel.pdf(idx)
            # Likelihood of frequency observation given that it is invalid
            pPhiInvalid = 1.0 / band_width
            # Probability of frequency being valid / invalid given SNR
            pInvalid_sat = 1.0 - pValid[i_gnss]
            pValid_sat = pValid[i_gnss]
            # Likelihood of frequency
            pPhi = pPhiValid * pValid_sat + pPhiInvalid * pInvalid_sat
            # Log-likelihood
            cd = cd + np.sum(np.log(pPhi))
        cd = np.clip(cd, -np.finfo(np.float).max, np.finfo(np.float).max)
        return -cd

    # Constrain search space
    bounds = [(-max_frequency_bias, max_frequency_bias)]

    # Decreasing standard deviation of Gaussian noise on frequencies [Hz]
    std = 2.0**np.arange(9, 1, -2)

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

    # Predict frequencies
    n_gnss = len(vis)
    predicted_frequencies = []
    for i_gnss in range(n_gnss):
        predicted_frequencies.append(np.empty(vis[i_gnss].shape))
        for i_sat in range(vis[i_gnss].shape[0]):
            predicted_frequencies[i_gnss][i_sat] = ep.get_doppler(
                init_time[i_gnss], init_pos, vis[i_gnss][i_sat],
                eph[i_gnss])

    # Double (Doppler sign unknown)
    band_width = band_width * 2.0
    for i_gnss in range(n_gnss):
        predicted_frequencies[i_gnss] = np.concatenate(
            (-predicted_frequencies[i_gnss], predicted_frequencies[i_gnss])
            )
        observed_frequencies[i_gnss] = np.concatenate(
            (-observed_frequencies[i_gnss], observed_frequencies[i_gnss])
            )
        pValid[i_gnss] = np.concatenate(
            (pValid[i_gnss] / 2.0, pValid[i_gnss] / 2.0)
             )

    # Initial hypothesis
    frequency_bias = 0.0

    if plot:
        import matplotlib.pyplot as plt
        b_vec = np.arange(-max_frequency_bias, max_frequency_bias+20, 20)

    for it_idx in range(len(std)):
        frequency_bias = so.minimize(
            neg_log_likelihood, frequency_bias, args=(
                std[it_idx], predicted_frequencies, observed_frequencies,
                pValid, band_width),
            method='L-BFGS-B', bounds=bounds).x

        if plot:
            lik = np.array([neg_log_likelihood(
                b, std[it_idx], predicted_frequencies, observed_frequencies,
                pValid, band_width) for b in b_vec])
            plt.plot(b_vec, lik)
            plt.show()

    return frequency_bias
