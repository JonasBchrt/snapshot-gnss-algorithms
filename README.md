# Snapshot GNSS

Author: *Jonas Beuchert*

This repository contains a Python script and additional open-source code to reproduce the results that are presented in

> Jonas Beuchert and Alex Rogers. 2021. SnapperGPS: Algorithms for Energy-Efficient Low-Cost Location Estimation Using GNSS Signal Snapshots. In SenSys ’21: ACM Conference on Embedded Networked Sensor Systems, November, 2021, Coimbra, Portugal. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3485730.3485931.

The script takes raw twelve-millisecond-long time-stamped low-quality GNSS signal snapshots, estimates the locations where they were recorded, and compares the results with the ground truth to obtain performance measures for the proposed algorithms. The first section of this readme describes the setup that is necessary to run the experiments.

Besides that, the repository provides a large number of GNSS utility functions that go beyond what is needed to reproduce the results presented in this paper. They are listed in the second section of this readme.

**Table of Contents**
1. [Setup](#setup)
2. [Exploring the Algorithms](#alg)
3. [GNSS Utility Functions](#gnss)

## Setup

### 1. Clone the repository to your local machine.

For example, via HTTPS

```shell
git clone https://github.com/JonasBchrt/snapshot-gnss-algorithms.git
cd snapshot-gnss-algorithms
```

### 2. Get the input data.

Experiments 1-3 are based on our data collection

> Jonas Beuchert and Alex Rogers. 2021. SnapperGPS: Collection of GNSS Signal Snapshots. University of Oxford, Oxford, UK. https://doi.org/10.5287/bodleian:eXrp1xydM.

Download the data into a sub-directory called `data` such that the file structure is

```
+-- main.py
+-- ...
+-- data
|   +-- A
    |   +-- 20201206_150000.bin
    |   +-- 20201206_150020.bin
    |   +-- ...
    |   +-- BRDC00IGS_R_20203410000_01D_MN.rnx
|   +-- ...
|   +-- K
    |   +-- 20210327_165641.bin
    |   +-- ...
    |   +-- BRDC00IGS_R_20210860000_01D_MN.rnx
    |   +-- ground_truth.gpx
```

Experiment 4 is based on the data collection that was published alongside

> Ryan M. Watson, Jason N. Gross, Clark N. Taylor, and Robert C. Leishman. 2019. Enabling robust state estimation through measurement error covariance adaptation. IEEE Trans. Aerospace Electron. Systems 56, 3 (2019), 2026–2040.

Download the data from [https://bit.ly/2vybpgA](https://bit.ly/2vybpgA) and add it to the `data` directory such that the file structure is

```
+-- main.py
+-- ...
+-- data
|   +-- Enabling_Robust_State_Estimation_through_Measurement_Error
    |   +-- iq
        |   +-- drive_2
            |   +-- GPS_002_0000.LS3
            |   +-- GPS_002_0001.LS3
            |   +-- GPS_002_0002.LS3
            |   +-- GPS_002_0003.LS3
        |   +-- drive_3
            |   +-- GPS_003_0000.LS3
            |   +-- GPS_002_0001.LS3
            |   +-- GPS_002_0002.LS3
            |   +-- GPS_002_0003.LS3
    |   +-- shared_data
        |   +-- BRDC00IGS_R_20183550000_01D_MN.rnx
    |   +-- truth
        |   +-- drive_2.xyz
        |   +-- drive_3.xyz
```

### 3. Install the dependencies.

The code was tested with Python 3.7.2 on Ubuntu 16.04, with Python 3.7.7 on Windows 10, and with Python 3.7.10 on Ubuntu 18.04 and macOS Big Sur.

Reproducing the results for the algorithms *LS-single*, *LS-linear*, *LS-combo*, *LS-SAC*, and *DPE* requires the packages `numpy`, `scipy`, `pymap3d`, `sklearn`, `shapely`, and `matplotlib`. You can install them via `pip`

```shell
python -m pip install -r requirements.txt
```

Algorithm *MLE* requires in addition `autograd` and `autoptim`, which you can add with

```shell
python -m pip install -r requirements_mle.txt
```

When using *MLE*, you **may** have to replace the first part
of the import section in `pymap3d/ecef.py`

```python
from math import radians, sin, cos, tan, atan, hypot, degrees, atan2, sqrt, pi
```

with

```python
from autograd.numpy import radians, sin, cos, tan, arctan as atan, hypot, degrees, arctan2 as atan2, sqrt, pi, vectorize
```

Optionally, *LS-single*, *LS-linear*, *LS-combo*, *LS-SAC*, and *MLE* run faster with `mkl_fft`, which you can install with

```shell
conda install -c intel mkl_fft
```

or

```shell
python -m pip install -r requirements_mkl.txt
```

### 3. Run the experiments.

Execute the script `main.py`, e.g., open a terminal and type

```shell
python main.py -e 1 -m "ls-sac"
```

to run Experiment 1 with algorithm *LS-SAC*. After the execution is completed, you should see the values in Table 1 of the paper printed in the terminal as well as a window with a CDF plot of the localisation error like in Figure 1

```shell
Median horizontal error: 11.7 m
Error < 200 m: 96%
Mean runtime: 0.52 s
```

![Expected output](http://users.ox.ac.uk/~kell5462/cdf_github_readme.png)

For a complete list of options, type

```shell
python main.py -h
```

## Exploring the Algorithms<a name="alg" />

If you want to explore the algorithm implementations beyond simply reproducing the results in the paper, then keep reading. However, I will not explain the theory behind the algorithms in detail here and will instead assume that you had a look on the paper.

Basically, the paper breaks the various approaches to snapshot GNSS into two groups: one that contains two-step algorithms with satellite acquisition followed by coarse-time navigation (CTN) and one that contains one-step algorithms that directly estimate a position from a raw snapshot and are called direct positioning, direct position estimation (DPE), or collective detection (CD). I will follow a similar structure here and will start with acquisition.

The satellite acquisition stage takes a raw GNSS signal snapshot and estimates the code phases of all potentially visible satellites as well as measures for the reliabilities of the estimates. The `main.py` script that you might have executed in the previous section calls an acquisition function `eph_util.acquisition_simplified` that is tailored to raw signal snapshots recorded with a SnapperGPS receiver. It in includes the prediction of the set of visible satellites and their expected Doppler shifts for a given coarse initial position and time. Knowledge of both helps to narrow down the search space and, therefore, speed up calculations. The functions supports different GNSS with their different code structures and lengths, but exposes only a small number of parameters to the user. However, there is also another `eph_util.acquisition` function that offers a lot more options and can be applied to custom raw data with arbitrary snapshot durations and sampling frequencies. For example, you can choose the way how code-phase interpolation is done, if you want to use non-coherent or coherent integration, and how you want to deal with the individual channels of the more sophisticated L1C, E1, and B1C signals. On the other hand, this function does not predict the visible satellites and their Doppler shifts. You may want to use `eph_util.get_visible_sats` and `ep.get_doppler` for this at first.

A raw implementation of least-squares CTN according to [van Diggelen](https://us.artechhouse.com/A-GPS-Assisted-GPS-GNSS-and-SBAS-P1729.aspx) is `coarse_time_navigation.coarse_time_nav`, which also has a (slightly) simplified counter-part `coarse_time_navigation.coarse_time_nav_simplified`. A core contribution of the paper is the introduction and analysis of different strategies to select the satellites that are used for CTN. They are implemented on top of the two aforementioned functions in `coarse_time_navigation.selective_coarse_time_nav` and `coarse_time_navigation.positioning_simplified`.

The latter function provides a wrapper for another core contribution of the paper, robust CTN via maximum-likelihood estimation (MLE) rather than least-squares. It is implemented in `coarse_time_navigation.coarse_time_nav_mle`.

Finally, the class `direct_position_estimation.DPE` contains our implementation of [Bissig et al.](https://doi.org/10.1145/3055031.3055083)'s algorithm for efficient direct positioning using branch-and-bound optimisation. Among others, it offers parameters to switch between the original version of the algorithm (`n=81`, `elev_mask=5`, `exponent=1`, `ref=False`, and `multi-ms-mode='multiple'`) and our version that improves robustness and runtime for signals that are longer than one millisecond (`n=16`, `elev_mask=10`, `exponent=2`, `ref=True`, and `multi-ms-mode='single'`). Note that the fundamentally different performance results between the two versions that you can see in the paper are achieved by changing `multi-ms-mode`. The original mode processes the individual milliseconds of a longer snapshot separately and averages over the results while our version integrates all individual millisecond-chunks and runs the optimisation just ones.

## GNSS Utility Functions<a name="gnss" />

The base for the algorithms mentioned in the preceding section is our Python library with GNSS utility functions, which might offer some useful pieces of code for other projects. If you employ the code in this repository for your own research, please consider citing the SnapperGPS paper.

### Additional Dependencies

Other functions in this repository rely on the following packages, which are not required to reproduce the results in the paper:

*    `nlopt` (only for `coarse_time_navigation.coarse_time_nav_mle` with non-default option)
*    `pygeodesy` (only for `eph_util.get_elevation`; requires external geoid models only one of which is included in the repository, others [can be downloaded](https://geographiclib.sourceforge.io/html/geoid.html#geoidinst) and unpacked into the `snapshot-gnss-algorithms` directory)
*    `rockhound` (only for `eph_util.get_elevation`)
*    `python-srtm` (only for `eph_util.get_elevation` with non-default option; requires external digital elevation models and only those close to Oxford are included in the repository, others [can be downloaded](https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/) and unpacked into the `snapshot-gnss-algorithms/digital_elevation_models` directory; `pip`-installation for Python <3.8 requires flag `--ignore-requires-python`)
*    `folium` (only for map plotting in `position_estimation_script`)
*    `sympy` (only for `eph_util.generate_b1c_code` and `eph_util.generate_l1c_code`)
*    `pandas` (only for `eph_util.get_sat_pos_sp3` and `eph_util.read_sp3`)
*    `gnsspy` (only for `eph_util.read_sp3`)

### Function Listing

*TODO*

### Usage Examples

Here are just a bunch of code snippets that demonstrate what some of the utility functions can do.

To localise a receiver on the Earth, you might first want to know where the GNSS satellites are. You can find information such as the satellite orbits, the errors of their on-board clocks, or the state of the ionosphere - which is a layer of the atmosphere that has a significant effect of the GNSS signal propagation speed - in so-called RINEX files. The NASA archives them [here](https://cddis.nasa.gov/archive/), another public source is the [BKG](https://igs.bkg.bund.de/). Internally, this code library represents the navigation data as a 2D NumPy array with 21 rows and columns for different satellites as well as different points in time. There are different ways to turn RINEX files into this representation:

```python
import eph_util as ep
# Read navigation data for Galileo satellites from a RINEX 3 file that contains only Galileo data
eph = ep.rinexe('BRUX00BEL_R_20201360000_01D_EN.rnx')
# Read navigation data for GPS satellites from a RINEX 2 file
eph = ep.rinexe('brdc1360.20n')
# Read navigation data for GPS satellites from a RINEX 3 file that contains data for multiple GNSS
eph = ep.rinexe("BRDC00IGS_R_20203410000_01D_MN.rnx", "G")
# Read navigation data for SBAS satellites from a RINEX 3 file that contains data for multiple GNSS
eph = ep.rinexe("BRDC00IGS_R_20203410000_01D_MN.rnx", "S")
# Read navigation data for Galileo satellites from a RINEX 3 file that contains data for multiple GNSS
eph = ep.rinexe("BRDC00IGS_R_20203410000_01D_MN.rnx", "E")
# Read navigation data for BeiDou satellites from a RINEX 3 file that contains data for multiple GNSS
eph = ep.rinexe("BRDC00IGS_R_20203410000_01D_MN.rnx", "C")

from rinex_preprocessor import preprocess_rinex
# Read navigation data for GPS, Galileo, and BeiDou satellites from a RINEX 3 file
eph_G, eph_E, eph_C = preprocess_rinex("BRDC00IGS_R_20203410000_01D_MN.rnx")
# Save the same data in .npy files in the working directory
preprocess_rinex("BRDC00IGS_R_20203410000_01D_MN.rnx", target_directory="")
```

You feed the navigation data into function such as `ep.get_sat_clk_corr` (satellite clock error), `ep.get_sat_pos_vel_acc` (satellite position, velocity, and acceleration), `ep.get_visible_sats` (satellites that are visible from the receiver location), and `ep.get_doppler` (expected Doppler shift of a GNSS signal).

Instead of RINEX data, you can also use `.sp3` files to calculate satellite orbits and positions using `ep.read_sp3` and `ep.get_sat_pos_sp3`.

Global Navigation Satellite Systems use their own time references. Basically, they count the number of seconds since a certain reference date. You can convert GPS and BeiDou time into UTC and vice-versa.

```python
gps_time = ep.utc_2_gps_time(np.datetime64('2020-05-15T00:00:00'))
beidou_time = ep.gps_time_2_beidou_time(gps_time)
utc = ep.gps_time_2_utc(gps_time)
```

The troposphere and the ionosphere are two layers of the atmosphere that have a considerable effect on the signal propagation speed. You can read the GPS ionosphere parameters from a RINEX file with `eph_util.gps_ionosphere_parameters_from_rinex`. Then you can use `eph_util.ionospheric_klobuchar` or `eop_util.ionospheric_tsui` to approximate the delay in the ionosphere. Furthermore, `eph_util.tropo`, `eph_util.troposheric_hopfield`, and `eph_util.troposheric_tsui` implement different models for correction of the tropospheric delay.

Each GNSS satellite broadcasts a unique code that allows a receiver to identify it in the captured data. (Except for GLONASS satellites, which all broadcast the same code in the L1 frequency band, but with slightly different carrier frequencies.) There are functions that generate the codes that are broadcasted in the L1/E1/B1C/L1C bands, namely `eph_util.generate_ca_code` for GPS L1 and SBAS, `eph_util.generate_e1_code` for Galileo E1, where `eph_util.e1b` and `eph_util.e1c` are responsible for the individual channels, `eph_util.generate_b1c_code` for BeiDou B1C with `eph_util.b1c_data` and `b1c_pilot` for the channels, `eph_util.generate_l1c_code` for GPS L1C based on `eph_util.l1c_data` and `eph_util.l1c_pilot`, and `eph_util.generate_ca_code_glonass` for GLONASS L1. ALternatively, you can read the already generated from files using the class `CodeDB` or even codes that are already pre-sampled at SnapperGPS' sampling frequency 4.092 MHz from `.npy` files: `codes_G.npy` for GPS L1, `codes_E.npy` for Galileo E1, and `codes_C.npy` for BeiDou B1C.

Finally, `eph_util.get_elevation` is there to work with different elevation models and geoids and `eph_util.get_relative_height_from_pressure` helps you to include pressure measurements into position estimation.

## Funding statment

SnapperGPS was supported by an EPSRC IAA Technology Fund.

Additionally, Jonas Beuchert is supported by the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems.
