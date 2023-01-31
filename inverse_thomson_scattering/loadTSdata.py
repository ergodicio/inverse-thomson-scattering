from os.path import join
from os import listdir

from pyhdf.SD import SD, SDC
import numpy as np
from inverse_thomson_scattering.warpcorr import warpCorrection


def loadData(sNum, sDay, loadspecs):
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
    if sDay:
        folder = r"\\redwood\archive\tmp\thomson"
    else:
        folder = "data"
        
    file_list = listdir(folder)
    files = [name for name in file_list if str(sNum) in name]
    
    for fl in files:
        if "epw" in fl or "EPW" in fl:
            hdfnameE = join(folder, fl)
            if "ccd" in fl or "CCD" in fl:
                xlab = "Radius (\mum)"
                specType = "imaging"
            else:
                xlab = "Time (ps)"
                specType = "temporal"
        if "iaw" in fl or "IAW" in fl:
            hdfnameI = join(folder, fl)
            if "ccd" in fl or "CCD" in fl:
                xlab = "Radius (\mum)"
                specType = "imaging"
            else:
                xlab = "Time (ps)"
                specType = "temporal"
        if "ats" in fl or "ATS" in fl:
            hdfnameE = join(folder, fl)
            specType = "angular"
            xlab = "Scattering angle (degrees)"
            
    if loadspecs["load_ion_spec"]:
        try:
            iDatfile = SD(hdfnameI, SDC.READ)
            sds_obj = iDatfile.select("Streak_array")  # select sds
            iDat = sds_obj.get()  # get sds data
            iDat = iDat.astype("float64")
            iDat = iDat[0, :, :] - iDat[1, :, :]
        except BaseException:
            print("Unable to find IAW")
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
            eDat = np.fliplr(eDat[0, :, :] - eDat[1, :, :])
        except BaseException:
            print("Unable to find EPW")
            eDat = []
            loadspecs["load_ele_spec"] = False
    else:
        eDat = []
            
    return eDat, iDat, xlab, specType