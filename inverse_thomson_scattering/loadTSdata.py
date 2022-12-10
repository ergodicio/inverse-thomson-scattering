from os.path import join

from pyhdf.SD import SD, SDC
import numpy as np
from inverse_thomson_scattering.warpcorr import warpCorrection


def loadData(sNum, sDay, specType, magE, loadspecs):
    """
    This function loads the appropriate data based off the provided shot number (sNum) and the type of data specified
    using specType. The flag sDay changes the default path to the temporary archive on the redwood server and will
    only work if connected to the LLE redwood server.

    If the data is time resolved an attempt will be made to locate t=0 based off the fiducials. The relationship
    between the fiducials and t=0 changes from one shot day to the next.

    Args:
        sNum:
        sDay:
        specType:
        magE:
        loadspecs:

    Returns:

    """

    if specType == 1:
        # shot day locations and nomenclature for ARTS needs to be checked
        if sDay:
            folder = r"\\redwood\archive\tmp\thomson"
            hdfnameE = join(folder, "ats", "s" + str(sNum) + ".hdf")
        else:
            folder = "data"
            hdfnameE = join(folder, "ATS-s" + str(sNum) + ".hdf")

        xlab = "Scattering angle (degrees)"
        zero = 0
        loadspecs["load_ion_spec"] = False
        loadspecs["load_ele_spec"] = True

    elif specType == 2:
        if sDay:
            folder = r"\\redwood\archive\tmp\thomson"
            hdfnameE = join(folder, "epw", "s" + str(sNum) + ".hdf")
            hdfnameI = join(folder, "iaw", "s" + str(sNum) + ".hdf")

        else:
            folder = "data"
            hdfnameE = join(folder, "EPW-s" + str(sNum) + ".hdf")
            hdfnameI = join(folder, "IAW-s" + str(sNum) + ".hdf")

        if loadspecs["load_ion_spec"]:
            try:
                iDatfile = SD(hdfnameI, SDC.READ)
                sds_obj = iDatfile.select("Streak_array")  # select sds
                iDat = sds_obj.get()  # get sds data
                iDat = iDat.astype("float64")
                iDat = iDat[0, :, :] - iDat[1, :, :]
            except BaseException:
                print("Unable to find Streaked IAW")
                iDat = []
                loadspecs["load_ion_spec"] = False
        else:
            iDat = []

        if loadspecs["load_ele_spec"]:
            try:
                eDatfile = SD(hdfnameE, SDC.READ)
                sds_obj = eDatfile.select("Streak_array")  # select sds
                eDat = sds_obj.get()  # get sds data
                eDat = eDat.astype("float64")
                eDat = eDat[0, :, :] - eDat[1, :, :]
                eDat = warpCorrection(eDat) #correction file need to be updated
                #eDat = eDat[16:1039,16:1039]
            except BaseException:
                print("Unable to find Streaked EPW")
                eDat = []
                loadspecs["load_ele_spec"] = False
        else:
            eDat = []

        xlab = "Time (ps)"

        # Depreciated matlab code for finding t=0 will be added back later 8/8/22
        #     % Attempt to find t=0 from the fiducials
        #     %for Aug-26-2021 data this is roughly 130 pixels before the first upper
        #     %fiducial
        #     %shift_zero = 1590; %currently no T-0
        #     fidu=sum(eDat(50:100,:));
        #     [~,fiduLocs]=findpeaks(fidu,'MinPeakHeight',0.5*max(fidu));
        #     zero= magE*(fiduLocs(1)-130);
        #     %find the zero time location in the iaw then offset is
        #     %zeroi=zeroe+offset
        #     fidui=sum(iDat(100:200,:));
        #     [~,fiduiLocs]=findpeaks(fidui,'MinPeakHeight',0.5*max(fidui));
        #     zeroi= magE*(fiduiLocs(1)-154);
        #     ioff=zeroi-zero;
        zero = 0

    else:
        if sDay:
            folder = r"\\redwood\archive\tmp\thomson"
            hdfnameE = join(folder, "epw_ccd", "s" + str(sNum) + ".hdf")
            hdfnameI = join(folder, "iaw_ccd", "s" + str(sNum) + ".hdf")

        else:
            folder = "data"
            hdfnameE = join(folder, "EPW_CCD-s" + str(sNum) + ".hdf")
            hdfnameI = join(folder, "IAW_CCD-s" + str(sNum) + ".hdf")

        if loadspecs["load_ion_spec"]:
            try:
                iDatfile = SD(hdfnameI, SDC.READ)
                sds_obj = iDatfile.select("Streak_array")  # select sds
                iDat = sds_obj.get()  # get sds data
                iDat = iDat.astype("float64")
                iDat = np.squeeze(iDat[0, :, :] - iDat[1, :, :])
                iDat = np.rot90(iDat, 3)
            except BaseException:
                print("Unable to find Imaging IAW")
                iDat = []
                loadspecs["load_ion_spec"] = False
        else:
            iDat = []

        if loadspecs["load_ele_spec"]:
            try:
                eDatfile = SD(hdfnameE, SDC.READ)
                sds_obj = eDatfile.select("Streak_array")  # select sds
                eDat = sds_obj.get()  # get sds data
                eDat = eDat.astype("float64")
                eDat = np.squeeze(eDat[0, :, :] - eDat[1, :, :])
                eDat = np.rot90(eDat, 3)
            except BaseException:
                print("Unable to find Imaging EPW")
                eDat = []
                loadspecs["load_ele_spec"] = False
        else:
            eDat = []

        xlab = "Radius (\mum)"
        zero = 0

    return eDat, iDat, xlab, zero
