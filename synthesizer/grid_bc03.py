"""
Read BC03 ASCII files in to a single numpy array (age, Z, lambda)

adapted from...

https://bitbucket.org/rteyssie/ramses/src/ea56da02dc3c52ba2b92650a14c206a1547aee52/trunk/ramses/utils/py/sed_utils.py

"""

import numpy as np
import sys
import re

from utils import write_data_h5py, write_attribute

# TODO: read in latest bc03-2016 *binary* files
# example:
# https://github.com/cmancone/easyGalaxy/blob/0608b17d84d00c2bdc069ebfb83024bf8d15e309/ezgal/utils.py#L382

def readBC03Array(file, lastLineFloat=None):
    """Read a record from bc03 ascii file. The record starts with the
       number of elements N and is followed by N numbers. The record may
       or may not start within a line, i.e. a line need not necessarily
       start with a record.
    Parameters:
    ----------------------------------------------------------------------
    file: handle on open bc03 ascii file
    lastLineFloat: still open line from last line read, in case of a
                   record starting mid-line.
    Returns array, lastLine, where:
    ----------------------------------------------------------------------
    array = The array values read from the file
    lastLine = The remainder of the last line read (in floating format),
               for continued reading of the file
    """

    if lastLineFloat is None or len(lastLineFloat) == 0:
        # Nothing in last line, so read next line
        line = file.readline()
        lineStr = line.split()
        lastLineFloat = [float(x) for x in lineStr]
    # Read array 'header' (i.e. number of elements)
    arrayCount = int(lastLineFloat[0])    # Length of returned array
    array = np.empty(arrayCount)          # Initialise the array
    lastLineFloat = lastLineFloat[1:len(lastLineFloat)]
    iA = 0                                # Running array index
    while True:                           # Read numbers until array is full
        for iL in range(0, len(lastLineFloat)):  # Loop numbers in line
            array[iA] = lastLineFloat[iL]
            iA = iA+1
            if iA >= arrayCount:                # Array is full so return
                return array, lastLineFloat[iL+1:]
        line = file.readline()   # Went through the line so get the next one
        lineStr = line.split()
        lastLineFloat = [float(x) for x in lineStr]


def convertBC03(files=None):
    """Convert BC03 outputs

    Parameters (user will be prompted for those if not present):
    ----------------------------------------------------------------------
    files: list of each BC03 SED ascii file, typically named
           bc2003_xr_mxx_xxxx_ssp.ised_ASCII
    """

    # Prompt user for files if not provided--------------------
    if files is None:
        print('Please write the model to read',)
        files = []
        while True:
            filename = input('filename >')
            if filename == '':
                break
            files.append(filename)
        print('ok checking now')
        if not len(files):
            print('No BC03 files given, nothing do to')
            return

    # Initialise ---------------------------------------------------------
    ageBins = None
    lambdaBins = None
    metalBins = [None] * len(files)
    seds = np.array([[[None]]])

    print('Reading BC03 files and converting...')
    # Loop SED tables for different metallicities
    for iFile, fileName in enumerate(files):
        print('Converting file ', fileName)
        file = open(fileName, 'r')
        ages, lastLine = readBC03Array(file)  # Read age bins
        nAge = len(ages)
        print("Number of ages: %s" % nAge)
        if ageBins is None:
            ageBins = ages
            seds.resize((seds.shape[0], len(ageBins),
                         seds.shape[2]), refcheck=False)
        if not np.array_equal(ages, ageBins):  # check for consistency
            print('Age bins are not identical everywhere!!!')
            print('CANCELLING CONVERSION!!!')
            return
        # Read four (five ?) useless lines
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()
        line = file.readline()
        # These last three lines are identical and contain the metallicity
        ZZ, = re.search('Z=([0-9]+\.?[0-9]*)', line).groups()
        metalBins[iFile] = eval(ZZ)
        seds.resize(
            (len(metalBins), seds.shape[1], seds.shape[2]), refcheck=False)
        # Read wavelength bins
        lambdas, lastLine = readBC03Array(file, lastLineFloat=lastLine)
        if lambdaBins is None:  # Write wavelengths to sed file
            lambdaBins = lambdas
            seds.resize((seds.shape[0], seds.shape[1],
                         len(lambdaBins)), refcheck=False)
        if not np.array_equal(lambdas, lambdaBins):  # check for consistency
            print('Wavelength bins are not identical everywhere!!!')
            print('CANCELLING CONVERSION!!!')
            return
        # Read luminosities
        for iAge in range(0, nAge):
            lums, lastLine = readBC03Array(file, lastLineFloat=lastLine)
            if len(lums) != len(lambdaBins):
                print('Inconsistent number of wavelength bins in BC03')
                print('STOPPING!!')
                return
            # Read useless array
            tmp, lastLine = readBC03Array(file, lastLineFloat=lastLine)
            seds[iFile, iAge] = lums
            progress = (iAge + 1) / nAge
            sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format(
                '#' * int(progress * 50), progress * 100))
        print(' ')
        lastLine = None


    return (np.array(seds, dtype=np.float64),
            np.array(metalBins, dtype=np.float64),
            np.array(ageBins, dtype=np.float64),
            np.array(lambdaBins, dtype=np.float64))





def main():
    """ Main function to convert BC03 grids and
        produce grids used by synthesizer """
    
    # Define base path
    basepath = "input_files/bc03/bc03/models/Padova2000/"

    # Define files
    files = ['bc2003_hr_m122_chab_ssp.ised_ASCII',
             'bc2003_hr_m132_chab_ssp.ised_ASCII',
             'bc2003_hr_m142_chab_ssp.ised_ASCII',
             'bc2003_hr_m152_chab_ssp.ised_ASCII',
             'bc2003_hr_m162_chab_ssp.ised_ASCII',
             'bc2003_hr_m172_chab_ssp.ised_ASCII']

    out = convertBC03([basepath + s for s in files])

    zsol = 0.0127

    spec = out[0]                   # Lsol / AA
    metals = np.log(out[1])         # log(Z)
    ages = out[2]                   # yr
    wl = out[3]                     # ?? (AA)

    # print("sed shape", spec.shape)
    # ignore zero age model
    # ages = ages[1:]
    # sed = sed[:,1:,:]

    spec *= (3.826e33 / 1.1964952e40)  # erg s^-1 cm^-2 AA^-1

    fname = 'output/bc03.h5'
    
    write_data_h5py(fname, 'spectra', data=spec, overwrite=True)
    write_attribute(fname, 'spectra', 'Description',
                    'Three-dimensional spectra grid, [Z,Age,wavelength]')
    write_attribute(fname, 'spectra', 'Units', 'erg s^-1 cm^2 AA^-1')

    write_data_h5py(fname, 'ages', data=ages, overwrite=True)
    write_attribute(fname, 'ages', 'Description', 
            'Stellar population ages in log10 years')
    write_attribute(fname, 'ages', 'Units', 'log10(yr)')

    write_data_h5py(fname, 'metallicities', data=metals, overwrite=True)
    write_attribute(fname, 'metallicities', 'Description', 
            'raw abundances in log10')
    write_attribute(fname, 'metallicities', 'Units', 'dimensionless [log10(Z)]')

    write_data_h5py(fname, 'wavelength', data=wl, overwrite=True)
    write_attribute(fname, 'wavelength', 'Description', 
            'Wavelength of the spectra grid')
    write_attribute(fname, 'wavelength', 'Units', 'AA')


# Lets include a way to call this script not via an entry point
if __name__ == "__main__":
    main()
