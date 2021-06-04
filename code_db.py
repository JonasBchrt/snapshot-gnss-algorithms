# -*- coding: utf-8 -*-
"""
Create or query a database with GNSS signal codes.

Author: Jonas Beuchert
"""
import numpy as np
import pickle
from eph_util import generate_ca_code, generate_e1_code, generate_b1c_code, \
    generate_l1c_code


class CodeDB:
    """
    Create or query a database with GNSS signal codes.

    Author: Jonas Beuchert
    """

    def create_db(self, filename='code_db.dat'):
        """Create new database and store codes L1, E1, and B1C codes.

        Generate and store pre-sampled codes. Use 4.092 MHz as sampling
        frequency. Include L1 C/A codes of GPS and SBAS, E1 codes of Galileo,
        and B1C codes of BeiDou.

        Input:
            filename - Filename of the database, default='code_db.dat'

        Author: Jonas Beuchert
        """
        # Fix sampling frequency
        sampling_frequency = 4092000.0

        # Create instance of database
        self.db = {}
        self.db['gps'] = {}
        self.db['gps'][None] = {}
        self.db['sbas'] = {}
        self.db['sbas'][None] = {}
        self.db['galileo'] = {}
        self.db['galileo'][False] = {}
        self.db['galileo'][True] = {}
        self.db['beidou'] = {}
        self.db['beidou'][False] = {}
        self.db['beidou'][True] = {}

        # GPS/SBAS L1 C/A codes

        code_duration = 1.0e-3
        # Samples per C/A code sequence
        sample = np.int(np.ceil(sampling_frequency * code_duration))
        sampleindex = np.arange(1, sample + 1)
        # C/A code frequency
        codeFreqBasis = 1.023e6

        # Loop over all potential GPS satellites
        for idx in np.arange(start=1, stop=32+1):
            print("GPS L1: {}".format(idx))
            # Generate C/A code
            ocode = generate_ca_code(idx)
            ocode = np.concatenate((ocode, ocode))
            # Sample code
            scode = ocode[np.ceil(sampleindex * codeFreqBasis
                                  / sampling_frequency).astype(int) - 1]
            # Insert into database
            self.db['gps'][None][idx.item()] = scode

        # Loop over all potential SBAS satellites
        for idx in np.arange(start=120, stop=138+1):
            print("SBAS L1: {}".format(idx))
            # Generate C/A code
            ocode = generate_ca_code(idx)
            ocode = np.concatenate((ocode, ocode))
            # Sample code
            scode = ocode[np.ceil(sampleindex * codeFreqBasis
                                  / sampling_frequency).astype(int) - 1]
            # Insert into database
            self.db['sbas'][None][idx.item()] = scode

        # Galileo E1 codes

        # Loop over all potential Galileo satellites
        for idx in np.arange(start=1, stop=50+1):
            print("Galileo E1: {}".format(idx))
            # Insert sampled E1 data signal
            self.db['galileo'][False][idx.item()] = generate_e1_code(
                idx, sampling_frequency, pilot=False)
            # Insert sampled E1 pilot signal
            self.db['galileo'][True][idx.item()] = generate_e1_code(
                idx, sampling_frequency, pilot=True)

        # BeiDou B1C codes

        # Loop over all potential BeiDou satellites
        for idx in np.arange(start=1, stop=63+1):
            print("BeiDou B1C: {}".format(idx))
            # Insert sampled B1C data signal
            self.db['beidou'][False][idx.item()] = generate_b1c_code(
                idx, sampling_frequency, pilot=False)
            # Insert sampled B1C pilot signal
            self.db['beidou'][True][idx.item()] = generate_b1c_code(
                idx, sampling_frequency, pilot=True)

        # Save dictionary into pickle file
        pickle.dump(self.db, open(filename, "wb"))

    def connect_db(self, filename='code_db.dat'):
        """Connect to existing database with pre-sampled GNSS codes.

        Input:
            filename - Filename of the database, default='code_db.dat'

        Author: Jonas Beuchert
        """
        # Load dictionary back from the pickle file
        self.db = pickle.load(open(filename, "rb"))

    def query_db(self, gnss, idx, pilot=None):
        """Extract pre-sampled GNSS code replica from database.

        Inputs:
            gnss - Type of satellite system: 'gps', 'sbas', 'galileo', or
                   'beidou'
            idx - Satellite index (PRN): 1-32 for GPS, 120-138 for SBAS,
                  1-50 for Galileo, 1-63 for Beidou
            pilot - Signal type: None for C/A code, False for data component,
                    or True for pilot component, default=None

        Output:
            replica - Pre-sampled satellite signal replica at 4.092 MHz

        Author: Jonas Beuchert
        """
        if self.db is None:
            raise Exception("No database loaded. Load database first using "
                            + "load_db().")
        # Search database
        # Return signal replica
        return self.db[gnss][pilot][idx]


# db = CodeDB()
# db.create_db()
