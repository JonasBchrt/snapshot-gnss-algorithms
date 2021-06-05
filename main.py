# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 21:49:53 2020

@author: Jonas Beuchert
"""

import numpy as np
import pymap3d as pm
import glob
import eph_util as ep
import coarse_time_navigation as ctn
from direct_position_estimation import DPE
import time as tm
import xml.etree.ElementTree as et
import shapely.geometry as sg
import matplotlib.pyplot as plt
from concurrent import futures
from rinex_preprocessor import preprocess_rinex
import scipy.signal as ss
import os
import getopt
import sys


def worker(data, experiment, mode):
    """Process one snapperGPS dataset for Experiment 1, 2, or 3.

    Inputs:
        data - Uppercase character indicating the SnapperGPS dataset ("A"-"K")
        experiment - Index of experiment (1-3)
                     1: GPS L1, Galileo E1, and BeiDou B1C
                     2: GPS L1 and Galileo E1
                     3: GPS L1
        mode - Algorithm ("ls-single", "ls-linear", "ls-combo", "ls-sac",
               "mle", "ls-sac/mle", "dpe")

    Output:
        results - Dictionary with fields
                  "error" - list of horizontal errors [m]
                  "time" - list of algorithm runtimes per snapshots [s]

    Author: Jonas Beuchert
    """
    print()
    print("Start processing dataset {}.".format(data))

    # Which GNSS to use
    gnss_list = {
        3: ['G'],  # GPS only (fastest, least robust)
        2: ['G', 'E'],  # GPS + Galileo
        1: ['G', 'E', 'C']  # GPS + Galileo + BeiDou (slowest, most robust)
        }
    gnss_list = gnss_list[experiment]

    # List of ground truth positions (static) / initial positions (dynamic)
    init_positions = {
        "A": np.array([50.870492, -1.562298, 100.0]),
        "B": np.array([51.763991, -1.259858, 100.0]),
        "C": np.array([51.751285, -1.246198, 100.0]),
        "D": np.array([51.760732, -1.257458, 100.0]),
        "E": np.array([51.735383, -1.211070, 100.0]),
        "F": np.array([51.735383, -1.211070, 100.0]),
        "G": np.array([51.735383, -1.211070, 100.0]),
        "H": np.array([51.735383, -1.211070, 100.0]),
        "I": np.array([51.755258204127756, -1.2591261135480434, 100.0]),
        "J": np.array([51.755258204127756, -1.2591261135480434, 100.0]),
        "K": np.array([51.755258204127756, -1.2591261135480434, 100.0])
        }
    pos_ref_geo = init_positions[data]

    # Frequency offsets of GNSS front-ends
    frequency_offsets = {
        "A": -864.0,
        "B": -384.0,
        "C": -384.0,
        "D": -768.0 + 900.0,
        "E": -768.0 - 300.0,
        "F": -768.0 - 300.0,
        "G": -768.0 - 300.0,
        "H": -768.0 - 300.0,
        "I": -768.0,
        "J": -768.0,
        "K": -768.0
        }
    # Intermediate frequency [Hz]
    intermediate_frequency = 4092000.0
    # Correct intermediate frequency
    intermediate_frequency = intermediate_frequency + frequency_offsets[data]

    # Diameter of temporal search space [s] (MLE and DPE only)
    search_space_time = {
        "A": 2.0,
        "B": 10.0,
        "C": 2.0,
        "D": 2.0,
        "E": 2.0,
        "F": 2.0,
        "G": 10.0,
        "H": 2.0,
        "I": 2.0,
        "J": 2.0,
        "K": 2.0
        }

    # RINEX navigation data files for different navigation satellite systems
    # You do not need all of them, just use None for those that you do not want
    # Broadcasted ephemeris can be found on
    # https://cddis.nasa.gov/archive/gps/data/daily/2021/brdc/
    rinex_file = glob.glob(os.path.join("data", data,
                                        "BRDC00IGS_R_*_01D_MN.rnx"))[0]
    eph_dict = {}
    eph_dict['G'], eph_dict['E'], eph_dict['C'] = preprocess_rinex(
        rinex_file
        )

    # Ground truth track for dynamic
    gt_files = glob.glob(os.path.join("data", data, "ground_truth*"))
    if len(gt_files) > 0:
        gt_enu = []
        for gt_file in gt_files:

            # Load ground truth
            root = et.parse(gt_file).getroot()
            file_ending = gt_file[-3:]
            print("Open ground truth file of type " + file_ending + ".")
            if file_ending == "gpx":
                # Ground truth position
                gt_geo = [(
                    float(child.attrib['lat']),
                    float(child.attrib['lon'])
                    ) for child in root[-1][-1]]
            elif file_ending == "kml":
                # Get coordinates of path
                try:
                    gt_string = root[-1][-1][-1][-1][-1][-1].text
                except(IndexError):
                    gt_string = root[-1][-1][-1][-1].text
                gt_geo = np.fromstring(
                    gt_string.replace('\n', '').replace('\t', '').replace(
                        ' ', ','), sep=',')
                gt_geo = [(lat, lon)
                        for lat, lon in zip(gt_geo[1::3], gt_geo[::3])]
            else:
                raise ValueError(
                    "Ground truth file format {} not recognized.".format(
                        file_ending))

            # Transform to ENU coordinates with same reference
            gt_enu.append(np.array(pm.geodetic2enu(
                [g[0] for g in gt_geo], [g[1] for g in gt_geo], 0,
                pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2]
                )).T)

        # Concatenate both parts, if there are two
        gt_enu = np.vstack(gt_enu)

        # Convert to line
        gt_enu_line = sg.LineString([(p[0], p[1]) for p in gt_enu])

    else:
        print("No ground truth file.")
        gt_file = None

    # Get all names of data files
    filenames = glob.glob(os.path.join("data", data, "*.bin"))

    modes = ["ls-single", "ls-linear", "ls-combo", "ls-sac", "mle",
             "ls-sac/mle", "dpe"]
    ls_modes = {key: val for key, val in zip(
        modes, ["single", "snr", "combinatorial", "ransac", None, "ransac",
                None]
        )}
    mle_modes = {key: val for key, val in zip(
        modes, [False, False, False, False, True, True, False]
        )}
    # Maximum number of satellites for CTN
    max_sat_count = {key: val for key, val in zip(
        modes, [5, 15, 10, 15, None, 15]
        )}

    if mode == "dpe":
        # Initialize DPE
        dpe_object = DPE()

    all_error = []
    all_time = []

    # Iterate over all files
    for idx, filename in enumerate(filenames):

        print('Snapshot {} of {}'.format(idx+1, len(filenames)))

        # Random error in box
        if gt_file is None:
            init_err_east = np.random.uniform(low=-10.0e3, high=10.0e3)
            init_err_north = np.random.uniform(low=-10.0e3, high=10.0e3)
            init_err_height = np.random.uniform(low=-100.0, high=100.0)
        else:
            init_err_east = np.random.uniform(low=-1.0e3, high=1.0e3)
            init_err_north = np.random.uniform(low=-1.0e3, high=1.0e3)
            init_err_height = np.random.uniform(low=-10.0, high=10.0)
        pos_geo = np.empty(3)
        pos_geo[0], pos_geo[1], pos_geo[2] = pm.enu2geodetic(
            init_err_east, init_err_north, init_err_height,
            pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

        # Ground truth time from filename
        YYYY = filename[-19:-15]
        MM = filename[-15:-13]
        DD = filename[-13:-11]
        hh = filename[-10:-8]
        mm = filename[-8:-6]
        ss = filename[-6:-4]
        utc = np.datetime64(YYYY
                            + "-" + MM
                            + "-" + DD
                            + "T" + hh
                            + ":" + mm
                            + ":" + ss)

        # Read signals from files
        # How many bytes to read
        bytes_per_snapshot = int(4092000.0 * 12e-3 / 8)
        # Read binary raw data from file
        signal_bytes = np.fromfile(filename, dtype='>u1',
                                   count=bytes_per_snapshot)
        # Get bits from bytes
        # Start here if data is passed as byte array
        signal = np.unpackbits(signal_bytes, axis=-1, count=None,
                               bitorder='little')
        # Convert snapshots from {0,1} to {-1,+1}
        signal = -2 * signal + 1

        print('Initial horizontal error: {:.0f} m'.format(np.linalg.norm(
            np.array([init_err_east, init_err_north]))))

        # Measure time spent on positioning
        start_time = tm.time()

        if mode == "dpe":

            ###################################################################
            # Positioning
            ###################################################################

            # Direct position estimation
            pos, time_dpe = dpe_object.run(
                # Subtract mean from signal
                signal=-(signal.astype(float) - np.mean(signal.astype(float))),
                sampling_freq=4092000.0,
                IF=intermediate_frequency,
                sign=+1,
                # Convert geodetic coordinates to ECEF
                init_pos=np.array(pm.geodetic2ecef(pos_geo[0], pos_geo[1],
                                                   pos_geo[2])),
                # Convert UTC to GPS time
                init_time=ep.utc_2_gps_time(utc),
                eph_GPS=eph_dict["G"] if "G" in gnss_list else None,
                eph_Galileo=eph_dict["E"] if "E" in gnss_list else None,
                search_space_pos=np.array([20.0e3, 20.0e3, 0.2e3]),
                search_space_time=search_space_time[data],
                mode="ENU",
                n=16,
                elev_mask=10,
                time_out=60*30,  # 30 min
                exponent=2,
                trop=False,
                time_resolution=40e-3,
                ref=True,
                pr_prediction_mode="approx",
                ms_to_process=int(12),
                multi_ms_mode='single')

        else:

            ###################################################################
            # Acquisition
            ###################################################################

            # Store acquisition results in dictionaries with one element per GNSS
            snapshot_idx_dict = {}
            prn_dict = {}
            code_phase_dict = {}
            snr_dict = {}
            eph_dict_curr = {}
            # Loop over all GNSS
            for gnss in gnss_list:
                # Acquisition
                snapshot_idx_dict[gnss], prn_dict[gnss], code_phase_dict[gnss], \
                    snr_dict[gnss], eph_idx, _, _ = ep.acquisition_simplified(
                        np.array([signal]),
                        np.array([utc]),
                        pos_geo,
                        eph=eph_dict[gnss],
                        system_identifier=gnss,
                        intermediate_frequency=intermediate_frequency,
                        frequency_bins=np.linspace(-0, 0, 1),
                        # Elevation mask for predicting satellites [deg]
                        elev_mask=10
                        )
                # Keep only navigation data that matches the satellites
                eph_dict_curr[gnss] = eph_dict[gnss][:, eph_idx]

            ###################################################################
            # Positioning
            ###################################################################

            # Estimate all positions with a single function call
            # Correct timestamps, too
            # Finally, estimate the horizontal one-sigma uncertainty
            latitude_estimates, longitude_estimates, time_utc_estimates, \
                uncertainty_estimates \
                = ctn.positioning_simplified(
                        snapshot_idx_dict,
                        prn_dict,
                        code_phase_dict,
                        snr_dict,
                        eph_dict_curr,
                        np.array([utc]),
                        # Initial position goes here or
                        # if data is processed in mini-batches, last plausible position
                        pos_geo[0], pos_geo[1], pos_geo[2],
                        # If we could measure the height, it would go here (WGS84)
                        observed_heights=None,
                        # If we measure pressure & temperature, we can estimate the height
                        pressures=None, temperatures=None,
                        # There are 5 different modes, 'snr' is fast, but inaccurate
                        # In the future, 'ransac' might be the preferred option
                        ls_mode=ls_modes[mode],
                        # Turn mle on to get a 2nd run if least-squares fails (recommended)
                        mle=mle_modes[mode],
                        # This parameter is crucial for speed vs. accuracy/robustness
                        # 10-15 is good for 'snr', 10 for 'combinatorial', 15 for 'ransac'
                        max_sat_count=max_sat_count[mode],
                        # These parameters determine the max. spatial & temporal distance
                        # between consecutive snapshots to be plausible
                        # Shall depend on the application scenario
                        max_dist=15.0e3, max_time=30.0,
                        # If we would know an initial offset of the timestamps
                        # If data is processed in mini-batches, the error from previous one
                        time_error=0.0,
                        search_space_time=search_space_time[data])

        # Measure time spent on positioning
        all_time.append(tm.time() - start_time)

        if mode == "dpe":

            # Calculate positioning error in ENU coordinates [m,m,m]
            err_east, err_north, err_height \
                = pm.ecef2enu(pos[0], pos[1], pos[2],
                              pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

        else:

            # Calculate positioning error in ENU coordinates [m,m,m]
            err_east, err_north, err_height \
                = pm.geodetic2enu(latitude_estimates[0], longitude_estimates[0], pos_ref_geo[2],
                                  pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

        if gt_file is not None:
            pos_enu = np.array([err_east, err_north, err_height])
            # Get nearest point on line for all estimated points
            nearest_point = gt_enu_line.interpolate(gt_enu_line.project(
                sg.Point((pos_enu[0], pos_enu[1]))
                ))

            # Calculate horizontal error
            err = np.linalg.norm(nearest_point.coords[0] - pos_enu[:2])
        else:
            err = np.linalg.norm(np.array([err_east, err_north]))

        if np.isnan(err):
            err = np.inf

        print('Resulting horizontal error: {:.0f} m'.format(err))

        all_error.append(err)

    return {"error": all_error, "time": all_time}


def worker_watson(mode):
    """Process data from Watson et al. for Experiment 4.

    Input:
        mode - Algorithm ("ls-single", "ls-linear", "ls-combo", "ls-sac")

    Output:
        results - Dictionary with fields
                  "error" - list of horizontal errors [m]
                  "time" - list of algorithm runtimes per snapshots [s]

    Author: Jonas Beuchert
    """
    # Downsampling factor
    scaling = 4

    # I or IQ
    iq = False

    # Which GNSS to use
    gnss_list = ['G']  # GPS only (fastest, least robust)

    # Elevation mask for predicting satellites [deg]
    elev_mask = 10
    frequency_bins = np.linspace(-0, 0, 1)

    # Snapshot parameters
    snapshot_duration = 12e-3
    snapshot_interval = 10.0

    all_error = {}
    all_time = {}
    modes = ["ls-single", "ls-linear", "ls-combo", "ls-sac"]
    ls_modes = {key: val for key, val in zip(
        modes, ["single", "snr", "combinatorial", "ransac"]
        )}
    mle_modes = {key: val for key, val in zip(
        modes, [False, False, False, False]
        )}
    # Maximum number of satellites for CTN
    max_sat_count = {key: val for key, val in zip(
        modes, [5, 15, 10, 15]
        )}

    all_error = []
    all_time = []

    # RINEX navigation data files for different navigation satellite systems
    # You do not need all of them, just use None for those that you do not want
    # Broadcasted ephemeris can be found on
    # https://cddis.nasa.gov/archive/gps/data/daily/2021/brdc/
    rinex_file = os.path.join("data", "Enabling_Robust_State_Estimation_through_Measurement_Error", "shared_data", "BRDC00IGS_R_20183550000_01D_MN.rnx")
    eph_dict = {}
    eph_dict['G'], eph_dict['E'], eph_dict['C'] = preprocess_rinex(
        rinex_file
        )

    # Index of drive
    for drive in np.arange(2, 4):

        print()
        print("Start processing drive {}.".format(drive))

        # Ground truth position (latitude, longitude, WGS84 height) [deg,deg,m]
        truth_file = os.path.join("data", "Enabling_Robust_State_Estimation_through_Measurement_Error", "truth", "drive_{}.xyz".format(drive))
        truth = np.genfromtxt(truth_file)
        truth[:, 0] = truth[:, 0] + 2032 * 7 * 24 * 60 * 60
        init = truth[0]
        pos_ref_geo = np.array(pm.ecef2geodetic(init[1], init[2], init[3]))
        time_ref = init[0] - 31.3
        if drive == 2:
            time_ref = time_ref - 48.3

        # Transform to ENU coordinates with same reference
        gt_enu = np.array(pm.ecef2enu(
            [g[1] for g in truth], [g[2] for g in truth], [g[3] for g in truth],
            pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2]
            )).T

        # Convert to line
        gt_enu_line = sg.LineString([(p[0], p[1]) for p in gt_enu])

        # Path to binary files
        data_directory = os.path.join("data", "Enabling_Robust_State_Estimation_through_Measurement_Error", "iq", "drive_{}".format(drive), "*.LS3")

        # Get all names of data files
        filenames = glob.glob(data_directory)

        # Intermediate frequency [Hz]
        intermediate_frequency = 16368000.0 / scaling - 283.8

        # Pre-process raw data
        sampling_frequency = 16368000.0
        # Calculate the minimax optimal filter using the Remez exchange algorithm
        taps = ss.remez(numtaps=5,
                        bands=np.array([0.0, 0.45, 0.55, 1.0])*0.5*sampling_frequency,
                        desired=[1.0, 0.0],
                        weight=[1.0, 1.0],
                        type='bandpass', grid_density=16, fs=sampling_frequency)

        # Iterate over all files
        idx = 0
        array_idx = 0
        samples_per_snapshot = int(sampling_frequency * snapshot_duration)
        samples_per_interval = int(sampling_frequency * snapshot_interval)

        int_per_snapshot = int(samples_per_snapshot / 8)
        int_per_interval = int(samples_per_interval / 8)

        file_idx = 0

        # Figure out how many snapshots are there
        n_snapshots = 0
        snapshots_per_byte = 4.0
        for filename in filenames:
            n_snapshots = n_snapshots + os.stat(
                filename
                ).st_size*snapshots_per_byte/sampling_frequency/snapshot_interval
        n_snapshots = int(np.ceil(n_snapshots))

        def bitget(data, bit):
            """Get value of bit at certian position."""
            return 1 if data & 2**bit else 0

        while file_idx < len(filenames):

            print('Snapshot {} of {} ({}).'.format(
                idx+1, n_snapshots, filenames[file_idx].split("/")[-1].split("\\")[-1]))

            raw_snapshot = np.fromfile(filenames[file_idx],
                                       dtype='int16',
                                       count=int_per_snapshot,
                                       sep='',
                                       offset=int(array_idx*2))
            # Check for end of file
            if raw_snapshot.shape[0] < int_per_snapshot:
                # Jump to next file
                file_idx = file_idx + 1
                array_idx = 0

            else:

                array_idx = array_idx + int_per_interval

                data = np.empty(samples_per_snapshot, dtype=np.csingle)

                for int_idx, input_short in enumerate(raw_snapshot):
                    for bit_idx in np.arange(8):
                        sample_idx = int_idx*8 + bit_idx
                        data[sample_idx] = float(bitget(input_short, 15-2*bit_idx)) \
                            + float(bitget(input_short, 14-2*bit_idx)) * 1.0j
                        data[sample_idx] = data[sample_idx] * 2.0 - (1.0 + 1.0j)

                data = ss.lfilter(b=taps, a=1, x=data, axis=- 1, zi=None)

                if iq:
                    signal = data[::scaling]
                else:
                    signal = np.real(data[::scaling])

                ref_idx = (np.abs(truth[:, 0] - time_ref)).argmin()
                pos_ref_ecef = truth[ref_idx, 1:]
                pos_ref_geo = np.array(pm.ecef2geodetic(pos_ref_ecef[0],
                                                        pos_ref_ecef[1],
                                                        pos_ref_ecef[2]))

                # Random error in box
                init_err_east = np.random.uniform(low=-1.0e3, high=1.0e3)
                init_err_north = np.random.uniform(low=-1.0e3, high=1.0e3)
                init_err_height = np.random.uniform(low=-10.0, high=10.0)
                pos_geo = np.empty(3)
                pos_geo[0], pos_geo[1], pos_geo[2] = pm.enu2geodetic(
                    init_err_east, init_err_north, init_err_height,
                    pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

                # Ground truth time
                utc = ep.gps_time_2_utc(time_ref + snapshot_interval * idx)

                # Truncate signal to 12 ms
                signal = signal[:int(
                    snapshot_duration*sampling_frequency/scaling)]
                if snapshot_duration < 12e-3:
                    factor = int(12e-3/snapshot_duration)
                    print("Repeat snapshot {} times.".format(factor))
                    signal = np.tile(signal, factor)

                print('Initial horizontal error: {:.0f} m'.format(np.linalg.norm(
                    np.array([init_err_east, init_err_north]))))

                # Measure time spent on positioning
                start_time = tm.time()

                ###############################################################################
                # Acquisition
                ###############################################################################

                # Store acquisition results in dictionaries with one element per GNSS
                snapshot_idx_dict = {}
                prn_dict = {}
                code_phase_dict = {}
                snr_dict = {}
                eph_dict_curr = {}
                # Loop over all GNSS
                for gnss in gnss_list:
                    # Acquisition
                    snapshot_idx_dict[gnss], prn_dict[gnss], code_phase_dict[gnss], \
                        snr_dict[gnss], eph_idx, _, _ = ep.acquisition_simplified(
                            np.array([signal]),
                            np.array([utc]),
                            pos_geo,
                            eph=eph_dict[gnss],
                            system_identifier=gnss,
                            intermediate_frequency=intermediate_frequency,
                            frequency_bins=frequency_bins,
                            elev_mask=elev_mask
                            )
                    # Keep only navigation data that matches the satellites
                    eph_dict_curr[gnss] = eph_dict[gnss][:, eph_idx]
                # Measure time spent on acquisition

                ###############################################################
                # Positioning
                ###############################################################

                # Estimate all positions with a single function call
                # Correct timestamps, too
                # Finally, estimate the horizontal one-sigma uncertainty
                latitude_estimates, longitude_estimates, time_utc_estimates, \
                    uncertainty_estimates \
                    = ctn.positioning_simplified(
                            snapshot_idx_dict,
                            prn_dict,
                            code_phase_dict,
                            snr_dict,
                            eph_dict_curr,
                            np.array([utc]),
                            # Initial position goes here or
                            # if data is processed in mini-batches, last plausible position
                            pos_geo[0], pos_geo[1], pos_geo[2],
                            # If we could measure the height, it would go here (WGS84)
                            observed_heights=None,
                            # If we measure pressure & temperature, we can estimate the height
                            pressures=None, temperatures=None,
                            # There are 5 different modes, 'snr' is fast, but inaccurate
                            # In the future, 'ransac' might be the preferred option
                            ls_mode=ls_modes[mode],
                            # Turn mle on to get a 2nd run if least-squares fails (recommended)
                            mle=mle_modes[mode],
                            # This parameter is crucial for speed vs. accuracy/robustness
                            # 10-15 is good for 'snr', 10 for 'combinatorial', 15 for 'ransac'
                            max_sat_count=max_sat_count[mode],
                            # These parameters determine the max. spatial & temporal distance
                            # between consecutive snapshots to be plausible
                            # Shall depend on the application scenario
                            max_dist=15.0e3, max_time=30.0,
                            # If we would know an initial offset of the timestamps
                            # If data is processed in mini-batches, the error from previous one
                            time_error=0.0)

                # Measure time spent on positioning
                all_time.append(tm.time() - start_time)

                # Calculate positioning error in ENU coordinates [m,m,m]
                err_east, err_north, err_height \
                    = pm.geodetic2enu(latitude_estimates[0], longitude_estimates[0], pos_ref_geo[2],
                                      pos_ref_geo[0], pos_ref_geo[1], pos_ref_geo[2])

                pos_enu = np.array([err_east, err_north, err_height])
                # Get nearest point on line for all estimated points
                nearest_point = gt_enu_line.interpolate(gt_enu_line.project(
                    sg.Point((pos_enu[0], pos_enu[1]))
                    ))

                # Calculate horizontal error
                err = np.linalg.norm(nearest_point.coords[0] - pos_enu[:2])

                if np.isnan(err):
                    err = np.inf

                print(mode)
                print('Resulting horizontal error: {:.0f} m'.format(err))

                all_error.append(err)

                idx = idx + 1

    return {"error": all_error, "time": all_time}


if __name__ == '__main__':

    usage = f"""
Usage: python {sys.argv[0]} [-h] | [-e <experiment>] [-m <mode>]

Example: python {sys.argv[0]} -e 1 -m "ls-sac"

Valid arguments for <experiment>: 1, 2, 3, 4
Valid arguments for <mode> if experiment=1:
    "ls-single", "ls-linear", "ls-combo", "ls-sac", "mle", "ls-sac/mle"
Valid arguments for <mode> if experiment=2:
    "ls-single", "ls-linear", "ls-combo", "ls-sac", "mle", "ls-sac/mle", "dpe"
Valid arguments for <mode> if experiment=3:
    "ls-single", "ls-linear", "ls-combo", "ls-sac", "mle", "ls-sac/mle", "dpe"
Valid arguments for <mode> if experiment=4:
    "ls-single", "ls-linear", "ls-combo", "ls-sac"
"""

    argv = sys.argv[1:]
    options, arguments = getopt.getopt(
        argv,
        "he:m:",
        ["help", "experiment=", "mode="])
    experiment = None
    mode = None
    for opt, arg in options:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        elif opt in ("-e", "--experiment"):
            experiment = arg
        elif opt in ("-m", "--mode"):
            mode = arg
    try:
        experiment = int(experiment)
    except TypeError:
        raise SystemExit(usage)
    if not (experiment == 1 and mode in ("ls-single",
                                         "ls-linear",
                                         "ls-combo",
                                         "ls-sac",
                                         "mle",
                                         "ls-sac/mle")
            or experiment == 2 and mode in ("ls-single",
                                            "ls-linear",
                                            "ls-combo",
                                            "ls-sac",
                                            "mle",
                                            "ls-sac/mle",
                                            "dpe")
            or experiment == 3 and mode in ("ls-single",
                                            "ls-linear",
                                            "ls-combo",
                                            "ls-sac",
                                            "mle",
                                            "ls-sac/mle",
                                            "dpe")
            or experiment == 4 and mode in ("ls-single",
                                            "ls-linear",
                                            "ls-combo",
                                            "ls-sac")):
        raise SystemExit(usage)
    print("Experiment {}".format(experiment))
    print("Algorithm {}".format(mode))

    np.random.seed(0)

    if experiment == 4:
        results = [worker_watson(mode)]
    else:
        # List of folders
        data = list(map(chr, range(ord('A'), ord('K')+1)))

        with futures.ProcessPoolExecutor() as pool:
            results = pool.map(worker, data,
                               [experiment]*len(data), [mode]*len(data))

        # results = []
        # for d in data:
        #     results.append(worker(d, experiment, mode))

    all_error = []
    all_time = []
    for result in results:
        all_error += result["error"]
        all_time += result["time"]

    def cdf(x, plot=True, *args, **kwargs):
        """Plot cumulative error."""
        x, y = sorted(x), np.arange(len(x)) / len(x)
        return plt.plot(x, y, *args, **kwargs) if plot else (x, y)

    def reliable(errors):
        """Portion of horizontal errors below 200 m."""
        return (np.array(errors) < 200).sum(axis=0) / len(errors)

    print()
    print("Median horizontal error: {:.1f} m".format(np.median(all_error)))
    print("Error < 200 m: {:.0%}".format(reliable(all_error)))
    print("Mean runtime: {:.2f} s".format(np.mean(all_time)))
    print()

    # Plot CDF
    cdf(all_error)

    plt.xlim(0, 200)
    plt.ylim(0, 1)
    plt.grid()
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("horizontal error [m]")
    plt.title(f"Experiment {experiment}, {mode}")
    plt.grid(True)

    plt.show()
