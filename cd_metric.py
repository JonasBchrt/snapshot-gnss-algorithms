"""Pseudo-likelihood inspired by collective detection / direct positioning."""


class CDMetric:
    """Calculate approximate pseudo-likelihood and its upper bound.

    Simplified pseudo-likelihood (CD metric) to speed up calculations.
    CD = collective detection.
    Based on a 4D hypothesis,i.e., 3D position and coarse time.
    Automatic omptimization of the 5th variable, the common bias.
    Doppler shifts taken into account.
    Speed-up relies on proper intialization when the constructor is called.
    Methods for the pseudo-likelihood of a hypothesis and for an estimated
    upper bound of the pseudo-likelihoods in a cube with a given diameter.

    Author: Jonas Beuchert
    """

    def __init__(self, signal, sampling_freq, IF, vis_sat_GPS=None,
                 doppler_GPS=None, vis_sat_Galileo=None, doppler_Galileo=None,
                 exponent=2, ms_to_process=1):
        """Initialise likelihood calculation.

        Convert signal from intermediate frequency (IF) to baseband considering
        Doppler shifts. Create local replicas of C/A codes of visible
        satellites. Correlate baseband signals with C/A codes. Store results to
        speed-up likelihood calculations.
        Consider GPS only, Galileo only, or GPS and Galileo.

        Inputs:
          signal - GNSS snapshot
          sampling_freq - Sampling frequency [Hz]
          IF - Intermediate frequency [Hz]
          vis_sat_GPS - Indices of GPS satellites (PRNs), which are expected to
                        be visible, or None if Galileo only
          doppler_GPS - Doppler shifts of the visible GPS satellites [Hz], or
                        None if Galileo only
          vis_sat_Galileo - Indices of Galileo satellites (PRNs), which are
                            expected to be visible, or None if GPS only
          doppler_Galileo - Doppler shifts of the visible Galileo satellites
                            [Hz] or None if Galileo only
          exponent - Exponent of pseudo-likelihodd function, default=2
          ms_to_process - Number of milliseconds to use (non-coherent
                          integration), default=1
        """
        import eph_util as ep
        import numpy as np

        # Number of visible satellites
        if vis_sat_GPS is not None and doppler_GPS is not None:
            nSatGPS = vis_sat_GPS.shape[0]
        else:
            nSatGPS = 0
        if vis_sat_Galileo is not None and doppler_Galileo is not None:
            nSatGalileo = vis_sat_Galileo.shape[0]
        else:
            nSatGalileo = 0
        if nSatGPS == 0 and nSatGalileo == 0:
            raise Exception("Missing input data.")
        nSat = nSatGPS + nSatGalileo

        # Store sampling frequency
        self.samplingFreq = sampling_freq
        # C/A code frequency [Hz]
        codeFreqBasis = 1023000.0
        # C/A code length [samples]
        if nSatGalileo > 0:
            codeLength = 4092.0
        else:
            codeLength = 1023.0
        # Find number of samples per spreading code
        samplesPerCode = np.int(np.round((sampling_freq / (codeFreqBasis /
                                                           codeLength))))
        # Find number of samples per millisecond
        samplesPerMs = np.int((np.round(sampling_freq * 1e-3)))

        # Find number of samples in signal snapshot
        samplesInSignal = samplesPerMs * ms_to_process

        # Create vector of data with correct length (1 ms)
        signal = signal[0: samplesInSignal]
        signal = np.reshape(signal, (ms_to_process, samplesPerMs))

        # Find time constants
        ts = 1.0 / self.samplingFreq  # Sampling period [s]
        tc = 1.0 / codeFreqBasis  # C/A chip period [s]

        # Generate all C/A codes and sample them according to sampling freq.
        # For Galileo, seperately generate data and pilot channel
        # Prepare output matrix to speed up function
        caCodesTable = np.empty((nSat+nSatGalileo, samplesPerCode))
        for k in range(nSatGPS):
            PRN = vis_sat_GPS[k]  # Index (PRN) of GPS satellite
            # Generate CA code for given PRN
            caCode = ep.generate_ca_code(PRN)
            if nSatGalileo > 0:
                # Adjust length to Galileo code (4 ms)
                caCode = np.tile(caCode, 4)
            # Digitizing
            # Make index array to read C/A code values
            codeValueIndex = np.ceil(ts * np.arange(1, samplesPerCode + 1) / tc
                                     ).astype(int) - 1
            # Correct the last index (due to number rounding issues)
            codeValueIndex[-1] = codeLength - 1
            # Make the digitized version of the C/A code
            caCodesTable[k] = caCode[codeValueIndex]
        for k in range(nSatGalileo):
            PRN = vis_sat_Galileo[k]  # Index (PRN) of Galileo sat
            # Make the digitized version of the E1B code (data)
            caCodesTable[nSatGPS + k] = ep.generate_e1_code(PRN,
                                                            sampling_freq)
            # Make the digitized version of the E1C code (pilot)
            caCodesTable[nSat + k] = ep.generate_e1_code(PRN, sampling_freq,
                                                         pilot=True)

        # Repeat C/A code table to match number of milliseconds in snapshot
        caCodesTable = np.tile(caCodesTable, (ms_to_process, 1))

        # Find phase points of the local carrier wave
        phasePoints = np.arange(samplesPerMs) * 2.0 * np.pi * ts

        # Shift raw signal to baseband
        # Prepare the output matrix to speed up function
        # (Zero padding if samplesPerCode > 1 ms)
        baseBandSignal = np.zeros(((nSat+nSatGalileo)*ms_to_process,
                                   samplesPerCode), dtype=np.complex64)
        for ms_idx in range(ms_to_process):
            for k in range(nSat):
                # Generate carrier wave frequency
                if k < nSatGPS:
                    frq = IF + doppler_GPS[k]  # Do not ignore Doppler shift
                else:
                    frq = IF + doppler_Galileo[k - nSatGPS]
                # Generate local sine and cosine
                carrier = np.exp(1j * frq * phasePoints)
                # "Remove carrier" from the signal
                baseband_signal_curr = carrier * signal[ms_idx]
                if k < nSatGPS:
                    # GPS L1
                    baseBandSignal[ms_idx*(nSat+nSatGalileo)+k,
                                   :samplesPerMs] = baseband_signal_curr
                else:
                    chunk_idx = np.mod(ms_idx, 4)
                    # Galileo E1B (data)
                    baseBandSignal[ms_idx*(nSat+nSatGalileo)+k,
                                   samplesPerMs*chunk_idx:
                                   samplesPerMs*(chunk_idx+1)] \
                        = baseband_signal_curr
                    # Galileo E1C (pilot)
                    baseBandSignal[ms_idx*(nSat+nSatGalileo)+k+nSatGalileo,
                                   samplesPerMs*chunk_idx:
                                   samplesPerMs*(chunk_idx+1)] \
                        = baseband_signal_curr

        # Correlate signals (to square or not to square?)
        corrTable = np.abs(np.fft.ifft(
            np.fft.fft(caCodesTable) * np.conj(np.fft.fft(baseBandSignal)))
            )**exponent

        # Sum correlograms for each satellite
        self.corrTable = np.zeros((nSat+nSatGalileo, samplesPerCode))
        for ms_idx in range(ms_to_process):
            self.corrTable = (self.corrTable
                              + corrTable[(ms_idx*(nSat+nSatGalileo)):
                                          ((ms_idx+1)*(nSat+nSatGalileo))])

        # Sum over Galileo channels
        self.corrTable = np.vstack((self.corrTable[:nSatGPS],
                                    self.corrTable[nSatGPS:nSat]
                                    + self.corrTable[nSat:nSat+nSatGalileo]))

        self.corrTableUncert = []
        self.uncertList = []

    def likelihood(self, code_phase):
        """Pseudo-likelihood of given set of C/A code phases.

        Input:
          code_phase - Expected C/A code phases of visible sats [s]
        Output:
          cd - Likelihood for observing signal given C/A code phases
        """
        import numpy as np

        # Pseudo-likelihood initialization
        cd = 0.0

        # Pseudo-likelihood calculation
        # For all listed PRN numbers ...
        for k in range(code_phase.shape[0]):

            # Shift correlation of C/A code replica with signal by code
            # phase and sum correlograms
            cd = cd + np.roll(self.corrTable[k], shift=-np.int(np.round(
                code_phase[k] * self.samplingFreq)))

        # Account for (unknown) common bias
        return np.max(cd)

    def max_likelihood(self, code_phase, diagonal):
        """Pseudo-likelihood of given set of C/A code phases.

        Input:
          code_phase - Expected C/A code phases of visible sats [s]
          diagonal - Diameter of 3D spatial cube that covers uncertainty [m]
        Output:
          cd - Approximated upper bound of pseudo-likelihood in uncertainty
               cube
        """
        import numpy as np

        # Max. pseudo-likelihood initialization
        cd = 0.0

        # Speed of light [m/s]
        c = 299792458.0

        # Maximum absolute deviation of C/A code phase in search space
        # with respect to center [samples]
        n = np.int(np.round(diagonal / c * self.samplingFreq / 2.0))

        # If uncertainty level is evaluated for the 1st time, create
        # table for later use
        if not self.uncertList or n < self.uncertList[-1]:
            # Remember that this uncertainty level has been evaluated
            self.uncertList.append(n)
            uncertInd = len(self.uncertList) - 1
            # Max-filter correlograms
            nSats = code_phase.shape[0]
            temp_table = np.empty_like(self.corrTable)
            for k in range(nSats):
                temp_table[k] = self.max_filter(self.corrTable[k], n)
            self.corrTableUncert.append(temp_table)
        else:
            # If uncertainty level has been evaluated already, find
            # index in table with max-filtered correlograms
            uncertInd = self.uncertList.index(n)

        for k in range(code_phase.shape[0]):

            # Shift max-filtered correlogram by code phase and sum
            cd = cd + np.roll(self.corrTableUncert[uncertInd][k], - np.int(
                np.round(code_phase[k] * self.samplingFreq)))

        # cd = np.array([np.roll(table_row, shift)
        #                for table_row, shift in
        #                zip(self.corrTableUncert[uncertInd],
        #                    - np.round(code_phase * self.samplingFreq).astype(
        #                        int))
        #                ])

        # cd = np.sum(cd, axis=0)

        # Account for (unknown) common bias
        return np.max(cd)

    def max_filter(self, x, n):
        """1D max-filter with wrapping.

        Inputs:
          x - Raw signal (row vector)
          n - Number of strides in each direction
        Output:
          filteredX - Maximum-filtered signal
        """
        import scipy.ndimage as si
        return si.maximum_filter(x, 2 * n + 1, mode="wrap")
