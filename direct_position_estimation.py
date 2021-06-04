"""Fast and Robust GPS Fix Using One Millisecond of Data.

Created on Tue Jun  9 11:40:21 2020

@author: Jonas Beuchert
"""
import pymap3d as pm
import numpy as np
import time
import copy
import eph_util as ep
from cd_metric import CDMetric
from hypothesis import Hypothesis
from hypotheses_queue import Queue
from hypotheses_set import Set
from pseudorange_prediction import PseudorangePrediction


class DPE:
    """Maximum-pseudo-likelihood estimation for GPS and Galileo snapshot.

    Branch-and-bound algorithm implemented according to:
    Bissig, Pascal, et al. “Fast and Robust GPS Fix Using One Millisecond of
    Data.” Proceedings of the 16th ACM/IEEE International Conference on
    Information Processing in Sensor Networks, 2017, pp. 223–233.
    https://tik-old.ee.ethz.ch/file/f65e5d021e6daee3344591d433b49e83/paper.pdf
    Author: Jonas Beuchert
    """

    def run(self, signal, sampling_freq, IF, sign, init_pos, init_time,
            eph_GPS=None, eph_Galileo=None,
            search_space_pos=np.array([20.0e3, 20.0e3, 200.0]),
            search_space_time=2.0,
            mode="ENU", n=16, elev_mask=5, time_out=120, exponent=2,
            trop=False, time_resolution=40e-3, ref=True,
            pr_prediction_mode='accurate', ms_to_process=1,
            multi_ms_mode='single'):
        """Direct position estimation (DPE).

        Inputs:
          signal - Raw GNSS signal snapshot (1 ms)
          sampling_freq - Sampling frequency [Hz]
          IF - Intermediate frequency [Hz]
          sign - Sign of Doppler shifts (+/-1)
          init_pos - Initial position hypothesis in ECEF coordinates (3x1 vec.)
          init_time - Initial GPS time hypothesis [s]
          eph_GPS - GPS ephemerides as matrix or None if Galileo only
          eph_Galileo - Galileo ephemerides as matrix or None if GPS only
          search_space_pos - Diameters of spatial uncertainty intervals
                             in ECEF or ENU coordinates (3x1 vector)
          search_space_time = Diameter of temporal uncertainty interval
          mode - Search space in "ECEF" or "ENU" coordinates, default="ENU"
          n - Number of most likely points to average over, default=16,
              Bissig et al. use 81
          elev_mask - Elevation mask for estimating visible satellites [deg],
                      default=5, cf. Bissig et al.
          time_out - Maximum run time [s], default=120,  Bissig et al. use Inf
          exponent - Exponent of pseudo-likelihodd function, default=2
          trop - Flag for tropospheric correction, default=False
          time_resolution - Resolution of temporal search space, default=40e-3
          ref - Switch for type of code-phase prediction, True for algorithm
                according to van Diggelen based on one reference satellite,
                False for algorithm according to Bissig et al. with two
                iterations for each satellite independently, default=True
          pr_prediction_mode - If ref=True, type of pseudorange approximation,
                               'accurate' (no approximation), 'approx' (non-
                               linear approximation), or 'linear'
                               (linearisation), default='accurate'
          ms_to_process - Number of milliseconds to use, default=1
          multi_ms_mode - Way to process multiple milliseconds, either
                          'multiple' for independently processing each
                          millisecond and averaging resulting positions (ETHZ)
                          or 'single' for non-coherant integration (summation
                          of the correlograms of all milliseconds) and one
                          joint optimisation, default = 'single'
        Outputs:
          pos - Position estimate in ECEF coordinates (3x1 vector)
          time - GPS time estimate [s]
        """
        # Initialize properties
        self.samplingFreq = sampling_freq
        self.mode = mode
        self.initPosGeo = np.empty(3)
        self.initPosGeo[0], self.initPosGeo[1], self.initPosGeo[2] \
            = pm.ecef2geodetic(init_pos[0], init_pos[1], init_pos[2])
        self.time_out = time_out
        self.trop = trop
        self.time_resolution = time_resolution
        self.ref = ref

        if multi_ms_mode == 'multiple':
            ms_to_process_avg = ms_to_process
            ms_to_process_integration = 1
        elif multi_ms_mode == 'single':
            ms_to_process_avg = 1
            ms_to_process_integration = ms_to_process
        else:
            raise Exception(
                "Chosen multi-ms processing mode not supported, \
                select 'multiple' or 'single'.")

        # Find number of samples per millisecond
        samplesPerMs = np.int((np.round(sampling_freq * 1e-3)))

        # Initialise arrays for results (position and time)
        pos = np.zeros((ms_to_process_avg, 3))
        time = np.zeros(ms_to_process_avg)

        # Loop over all 1-ms snapshots
        for current_ms in range(ms_to_process_avg):

            # Estimate visible satellites and identify matching columns in
            # ephemeris matrix, closest column in time for each satellite
            tow = np.mod(init_time, 7 * 24 * 60 * 60)
            if eph_GPS is not None:
                self.visGPS = ep.get_visible_sats(init_time, self.initPosGeo,
                                                  eph_GPS, elev_mask)
                colEphGPS = np.array([ep.find_eph(eph_GPS, sat, tow) for sat in
                                      self.visGPS])
                self.ephGPS = eph_GPS[:, colEphGPS]
            else:
                self.visGPS = np.array([])
                self.ephGPS = np.array([])
            if eph_Galileo is not None:
                self.visGalileo = ep.get_visible_sats(init_time,
                                                      self.initPosGeo,
                                                      eph_Galileo, elev_mask)
                colEphGalileo = np.array([ep.find_eph(eph_Galileo, sat, tow)
                                          for sat in self.visGalileo])
                self.ephGalileo = eph_Galileo[:, colEphGalileo]
            else:
                self.visGalileo = np.array([])
                self.ephGalileo = np.array([])

            # Estimate Doppler shifts
            dopplerGPS = np.empty(self.visGPS.shape)
            dopplerGalileo = np.empty(self.visGalileo.shape)
            for i in range(self.visGPS.shape[0]):
                dopplerGPS[i] = sign * ep.get_doppler(init_time, init_pos,
                                                      self.visGPS[i],
                                                      self.ephGPS)
            for i in range(self.visGalileo.shape[0]):
                dopplerGalileo[i] = sign * ep.get_doppler(init_time, init_pos,
                                                          self.visGalileo[i],
                                                          self.ephGalileo)

            # Extract current ms
            current_signal = signal[
                (current_ms*samplesPerMs*ms_to_process_integration):
                    ((current_ms+1)*samplesPerMs*ms_to_process_integration)]

            # Initialize metric
            self.metric = CDMetric(current_signal, sampling_freq, IF,
                                   self.visGPS, dopplerGPS,
                                   self.visGalileo, dopplerGalileo,
                                   exponent,
                                   ms_to_process=ms_to_process_integration)

            if (pr_prediction_mode == "approx"
               or pr_prediction_mode == "linear"):
                # Initialize pseudorange prediction
                if self.visGPS.shape[0] > 0:
                    self.pr_prediction_obj_gps = PseudorangePrediction(
                        self.visGPS, self.ephGPS, init_time, init_pos, 0, trop)
                if self.visGalileo.shape[0] > 0:
                    self.pr_prediction_obj_galileo = PseudorangePrediction(
                        self.visGalileo, self.ephGalileo, init_time, init_pos,
                        0, trop)
                self.pr_prediction_mode = pr_prediction_mode
            else:
                self.pr_prediction_mode = 'accurate'

            # Check coordinate system
            if mode == "ENU":
                init_pos = np.zeros(3)

            # Create initial hypothesis
            h = Hypothesis(init_pos, init_time, np.array([]), np.array([]),
                           search_space_pos, search_space_time)

            # Run MLE
            h = self.getMostLikelyPoints(n, h)

            # # Average over n most likely points
            L = np.array([_.lmin for _ in h])
            L = L / np.sum(L)
            pos[current_ms] = L @ np.array([_.p for _ in h])
            time[current_ms] = L @ np.array([_.t for _ in h])

            # stdPosENU = sqrt(var([h.p]-pos,L,2))

            # Check coordinate system, convert to ECEF if necessary
            if mode == "ENU":
                pos[current_ms, 0], pos[current_ms, 1], pos[current_ms, 2] = (
                    pm.enu2ecef(pos[current_ms, 0], pos[current_ms, 1],
                                pos[current_ms, 2], self.initPosGeo[0],
                                self.initPosGeo[1], self.initPosGeo[2]))

            # Increase initial time for next snapshot by 1 ms
            init_time = init_time + 1e-3

        return np.mean(pos, axis=0), np.mean(time)

    def getMostLikelyPoints(self, n, h):
        """Find n most likely points given search space defined by hypothesis.

        Implemented according to:
        Bissig, Pascal, et al. “Fast and Robust GPS Fix Using One Millisecond
        of Data.” Proceedings of the 16th ACM/IEEE International Conference on
        Information Processing in Sensor Networks, 2017, pp. 223–233.
        https://tik-old.ee.ethz.ch/file/f65e5d021e6daee3344591d433b49e83/paper.pdf
        Inputs:
          n - The number of likely points contained in S
          h - The initial hypothesis defining the search space
        Output:
          S - Array of n most likely hypotheses
        """
        h.lmax = self.maxLikelihood(h)
        queue = Queue()
        queue.add(h)
        S = Set(n)
        start = time.time()
        while queue.hasElement() and time.time() - start < self.time_out:
            # Pop most likely element
            h = queue.popMostLikely()
            if h.lmax <= S.minLmin:
                # Terminate because min likelihood in set is larger
                continue
            # Keep iterating because min likelihood in set is smaller
            # Likelihood
            h.lmin = self.likelihood(h)
            h.lmax = self.maxLikelihood(h)
            S.add(h)
            hSplit = self.splitHypothesis(h)
            for hSplit_i in hSplit:
                hSplit_i.lmax = h.lmax
                queue.add(hSplit_i)
        return S.data

    def likelihood(self, h):
        """Calculate pseudo-likelihood of given hypothesis."""
        # Check coordinate system, convert to ECEF if necessary
        if self.mode == "ENU":
            x, y, z = pm.enu2ecef(h.p[0], h.p[1], h.p[2], self.initPosGeo[0],
                                  self.initPosGeo[1], self.initPosGeo[2])
        else:
            x = h.p[0]
            y = h.p[1]
            z = h.p[2]

        # Predict code phases
        if self.visGPS.shape[0] > 0:
            codePhaseGPS = self.predict_code_phases(h, x, y, z, gps=True)
        else:
            codePhaseGPS = np.array([])
        if self.visGalileo.shape[0] > 0:
            codePhaseGalileo = self.predict_code_phases(h, x, y, z, gps=False)
        else:
            codePhaseGalileo = np.array([])

        # Calculate pseudo-likelihood
        return self.metric.likelihood(np.concatenate((codePhaseGPS,
                                                      codePhaseGalileo)))

    def maxLikelihood(self, h):
        """Estimate upper-bound of pseudo-likelihoods in uncertainty space."""
        # Check coordinate system, convert to ECEF if necessary
        if self.mode == "ENU":
            x, y, z = pm.enu2ecef(h.p[0], h.p[1], h.p[2], self.initPosGeo[0],
                                  self.initPosGeo[1], self.initPosGeo[2])
        else:
            x = h.p[0]
            y = h.p[1]
            z = h.p[2]

        # Predict code phases
        if self.visGPS.shape[0] > 0:
            codePhaseGPS = self.predict_code_phases(h, x, y, z, gps=True)
        else:
            codePhaseGPS = np.array([])
        if self.visGalileo.shape[0] > 0:
            codePhaseGalileo = self.predict_code_phases(h, x, y, z, gps=False)
        else:
            codePhaseGalileo = np.array([])

        # Estimate upper bound
        return self.metric.max_likelihood(
            np.concatenate((codePhaseGPS, codePhaseGalileo)),
            np.linalg.norm(h.searchSpaceP))

    def splitHypothesis(self, h):
        """Split hypotheses in all dimensions.

        Split hypotheses in all dimensions in which the minimum resolution of
        the search space discretization has not been reached.
        Input:
          h - Parent hypothesis
        Output:
          hSplit - Array of child hypotheses
        """
        # Split recursively, start with time dimension
        hSplit = self.splitTime(h)

        # Return nothing if no split was done, i.e., just initial hypothesis is
        # remaining
        if hSplit.shape[0] <= 1:
            return np.array([])
        else:
            return hSplit

    def splitTime(self, h):
        """Split hypothesis recursively in all dimensions, start with time.

        Split hypothesis recursively in all dimensions in which the minimum
        resolution of the search space discretization has not been reached.
        Start with time dimension.
        Input:
          h - Parent hypothesis
        Output:
          hSplit - Array of child hypotheses
        """
        # Check if minimum time resolution has not been reached
        if h.searchSpaceT > self.time_resolution:

            # Half time search space
            h.searchSpaceT = h.searchSpaceT / 2.0

            # Split hypothesis in time domain
            h2 = copy.deepcopy(h)
            h.t = h.t - h.searchSpaceT / 2.0
            h2.t = h2.t + h2.searchSpaceT / 2.0

            # Split recursively in spatial domain starting with first
            # position dimension (x or east)
            return np.concatenate((self.splitPosition(h, 0),
                                   self.splitPosition(h2, 0)))

        else:  # Minimum resolution reached

            # Split recursively in spatial domain starting with
            # first position dimension (x or east)
            return self.splitPosition(h, 0)

    def splitPosition(self, h, ind):
        """Split hypothesis recursively in all spatial dimensions.

        Split hypothesis recursively in all spatial dimensions in which the
        minimum resolution of the search space discretization has not been
        reached.
        Input:
          h - Parent hypothesis
          ind - Index of spatial dimension (0, 1, 2)
        Output:
          hSplit - Array of child hypotheses
        """
        c = 299792458.0  # Speed of light [m/s]

        # Check if minimum resolution has not been reached
        if h.searchSpaceP[ind] > c / self.samplingFreq:

            # Half search space
            h.searchSpaceP[ind] = h.searchSpaceP[ind] / 2.0

            # Split hypothesis
            h2 = copy.deepcopy(h)
            h.p[ind] = h.p[ind] - h.searchSpaceP[ind] / 2.0
            h2.p[ind] = h2.p[ind] + h2.searchSpaceP[ind] / 2.0

            # Check if spatial dimensions are left to continue
            # recursion otherwise terminate recursion
            if ind < 2:
                return np.concatenate((self.splitPosition(h, ind + 1),
                                       self.splitPosition(h2, ind + 1)))
            else:
                return np.array([h, h2])

        else:  # Minimum spatial resolution reached

            # Check if spatial dimensions are left to continue
            # recursion otherwise terminate recursion
            if ind < 2:
                return self.splitPosition(h, ind + 1)
            else:
                return np.array([h])

    def predict_code_phases(self, h, x, y, z, gps):
        """Predict inverted code phases using one of two methods.

        Inputs:
            h - Hypothesis
            x - x-coordinate (ECEF)
            y - y-coordinate (ECEF)
            z - z-coordinate (ECEF)
            gps - Switch for GNSS, True for GPS, False for Galileo
        Output:
            code_phases - Array with inverted code phases [s]
        """
        c = 299792458.0  # Speed of light [m/s]
        if gps:
            vis = self.visGPS
            eph = self.ephGPS
            code_duration = 1e-3
            if (self.pr_prediction_mode == 'approx'
               or self.pr_prediction_mode == 'linear'):
                pr_prediction_obj = self.pr_prediction_obj_gps
        else:
            vis = self.visGalileo
            eph = self.ephGalileo
            code_duration = 4e-3
            if (self.pr_prediction_mode == 'approx'
               or self.pr_prediction_mode == 'linear'):
                pr_prediction_obj = self.pr_prediction_obj_galileo

        if self.ref:
            # Use van Diggelen's algorithm to predict pseudoranges
            if self.pr_prediction_mode == 'approx':
                # Non-linear approximation
                pr = pr_prediction_obj.predict_approx(h.t, np.array([x, y, z]),
                                                      0.0)
            elif self.pr_prediction_mode == 'linear':
                # Linear approximation
                pr = pr_prediction_obj.predict_linear(h.t, np.array([x, y, z]),
                                                      0.0)
            else:
                # No approximation
                pr = ep.predict_pseudoranges(vis, eph, h.t,
                                             np.array([x, y, z]), 0.0,
                                             self.trop)
            # Get code phases from travel times
            codePhase = np.mod(pr / c, code_duration) / code_duration
        else:
            # Use Bissig et al.'s algorithm to predict code phases
            codePhase = ep.get_code_phase(h.t, np.array([x, y, z]), 0.0, vis,
                                          eph, code_duration, self.trop
                                          ) / code_duration

        # Invert, convert to seconds
        return (1.0 - codePhase) * code_duration
