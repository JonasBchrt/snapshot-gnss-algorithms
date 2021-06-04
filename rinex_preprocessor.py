# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:19:45 2021

@author: Jonas Beuchert
"""

import numpy as np
import os


def preprocess_rinex(rinex_file, target_directory=None):
    """Read a RINEX Navigation Message file and reformat the data.

    Read a file with name "BRDC00IGS_R_yyyyddd0000_01D_MN.rnx" that was
    downloaded from https://cddis.nasa.gov/archive/gnss/data/daily/yyyy/brdc/
    where yyyy is the year and ddd is the day of the year. The file must be
    unpacked first such that it is a plain text file in RINEX 3 format. It is
    expected to contain data for GPS (G), Galileo (E), and BeiDou (C) in this
    order.

    If target_directory=None, then return a 2D navigation data NumPy array with
    21 rows for GPS, Galileo, and BeiDou, respectively, else attempt to write
    the three arrays in the '.npy' format to the directory specified in
    target_directory:
    yyyy_ddd_G.npy for GPS,
    yyyy_ddd_E.npy for Galileo, and
    yyyy_ddd_C.npy for BeiDou.

    Units are either seconds, meters, or radians.

    Typical call: preprocess_rinex("BRDC00IGS_R_20203410000_01D_MN.rnx")

    Inputs:
        rinex_file - Path to unpacked RINEX Navigation Message file ending with
                     "BRDC00IGS_R_yyyyddd0000_01D_MN.rnx"
        target_directory - Directory to store the navigation data matrices or
                           None if they shall be returned, default=None

    Outputs:
        eph_G - GPS navigation data matrix, 2D NumPy array with 21 rows
        eph_E - Galileo navigation data matrix, 2D NumPy array with 21 rows
        eph_D - BeiDou navigation data matrix, 2D NumPy array with 21 rows

    Author: Jonas Beuchert
    """
    with open(rinex_file, "r") as fide:

        # Skip header
        line = fide.readline()
        while not line == "" and not line.find("END OF HEADER") > -1:
            line = fide.readline()
        if line == "":
            raise Exception(
                "Invalid RINEX navigation data file."
                )

        # Expected maximum number of columns for a single GNSS (Galileo)
        max_col = 20000
        # Set aside memory for the input
        svprn = np.zeros(max_col)
        toe = np.zeros(max_col)
        af2 = np.zeros(max_col)
        af1 = np.zeros(max_col)
        af0 = np.zeros(max_col)
        deltan = np.zeros(max_col)
        M0 = np.zeros(max_col)
        ecc = np.zeros(max_col)
        roota = np.zeros(max_col)
        toe = np.zeros(max_col)
        cic = np.zeros(max_col)
        crc = np.zeros(max_col)
        cis = np.zeros(max_col)
        crs = np.zeros(max_col)
        cuc = np.zeros(max_col)
        cus = np.zeros(max_col)
        Omega0 = np.zeros(max_col)
        omega = np.zeros(max_col)
        i0 = np.zeros(max_col)
        Omegadot = np.zeros(max_col)
        idot = np.zeros(max_col)
        # Create list for returned matrices
        eph = []

        # Loop over all three GNSS (expected order: GPS, Galileo, BeiDou)
        line = fide.readline()
        gnss_list = ["G", "E", "C"]
        while not line == "" and len(gnss_list) > 0:
            gnss = gnss_list.pop(0)

            # Loop until next desired GNSS is found
            while not line == "" and line[0] != gnss:
                line = fide.readline()
            if line == "":
                raise Exception(
                    "RINEX navigation data file does not contain data for "
                    + "all desired GNSS or they are not in the desired order "
                    + "(G - E - C)."
                    )

            # reset index
            i = 0
            # Loop over all entries for this GNSS
            while line[0] == gnss:

                try:
                    # Read one entry
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
                    line = fide.readline()
                except:
                    Exception(
                        "Found corrupted entry for GNSS {}.".format(gnss)
                    )

                # Read first line of next entry
                line = fide.readline()
                i += 1

            # Reformat data into array with 21 rows
            eph.append(np.array(
                [
                    svprn[:i],
                    af2[:i],
                    M0[:i],
                    roota[:i],
                    deltan[:i],
                    ecc[:i],
                    omega[:i],
                    cuc[:i],
                    cus[:i],
                    crc[:i],
                    crs[:i],
                    i0[:i],
                    idot[:i],
                    cic[:i],
                    cis[:i],
                    Omega0[:i],
                    Omegadot[:i],
                    toe[:i],
                    af0[:i],
                    af1[:i],
                    toe[:i]
                ]
            ))

    if len(gnss_list) > 0:
        raise Exception(
            "RINEX navigation data file does not contain data for "
            + "all desired GNSS or they are not in the desired order "
            + "(G - E - C)."
            )

    if target_directory is None:
        return tuple(eph)

    # Extract year and day of year
    yyyy = rinex_file[-22:-18]
    ddd = rinex_file[-18:-15]
    # Save three .npy files
    for gnss, eph_gnss in zip(["G", "E", "C"], eph):
        np.save(os.path.join(target_directory, yyyy + "_" + ddd + "_" + gnss),
                eph_gnss)
