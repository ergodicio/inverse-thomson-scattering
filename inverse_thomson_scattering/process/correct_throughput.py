import xlrd
import numpy as np
from numpy.matlib import repmat
import scipy.interpolate as sp
import scipy.io as sio
from os.path import join, exists
import matplotlib.pyplot as plt


def correctThroughput(data, tstype, axisy, shotNum):
    if tstype == "angular":
        imp = sio.loadmat(join('files', 'spectral_sensitivity.mat'), variable_names='speccal')
        speccal = imp['speccal']
        # not sure if this transpose is correct, need to check once plotting is working
        speccal = np.transpose(speccal)
        if shotNum < 95000:
            vq1 = 1. / speccal
        else:
            specax = np.arange(0, 1023) * .214116 + 449.5272
            speccalshift = sp.interp1d(specax, speccal, "linear", bounds_error=False, fill_value=speccal[0])
            vq1 = 1. / speccalshift(axisy)
        # print(np.shape(vq1))

    elif tstype == "temporal":
        wb = xlrd.open_workbook(join('files', 'Copy of MeasuredSensitivity_9.21.15.xls'))
        sheet = wb.sheet_by_index(0)
        sens = np.zeros([301, 2])

        for i in range(2, 303):
            sens[i - 2, 0] = sheet.cell_value(i, 0)
            sens[i - 2, 1] = sheet.cell_value(i, 1)

        sens[:, 1] = 1. / sens[:, 1]
        sens[0:17, 1] = sens[18, 1]  # the sensitivity goes to zero in this location and is not usable

        speccalshift = sp.interp1d(sens[:, 0], sens[:, 1], "linear", bounds_error=False, fill_value=sens[0, 1])
        vq1 = speccalshift(axisy)

    else:
        imp = sio.loadmat(join("files", "MeasuredSensitivity_11_30_21.mat"), variable_names="sens")
        sens = imp["sens"]

        sens[:, 1] = 1.0 / sens[:, 1]
        sens[0:17, 1] = sens[18, 1]  # the sensitivity goes to zero in this location and is not usable

        speccalshift = sp.interp1d(sens[:, 0], sens[:, 1], "linear", bounds_error=False, fill_value=sens[0, 1])
        vq1 = speccalshift(axisy)

    # Note that C has NaN in it.
    C = np.transpose(repmat(vq1, 1024, 1))  # expand my wavelength corrections vector into a matrix
    C[np.isnan(C)] = 0
    cdata = data * C
    # Correct each wavelength/Row of the matrix
    return cdata
