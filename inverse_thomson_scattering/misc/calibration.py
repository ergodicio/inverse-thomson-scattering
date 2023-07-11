# Supplies wavelength, space, time, and throughput calibrations based off of shot numbers including historical values
# new calibration values should be added here as they are calculated
import numpy as np
import scipy.io as sio
from os.path import join, exists

def get_calibrations(shotNum, tstype, CCDsize):
    stddev = dict()
    # Dispersions and calibrations
    if tstype == "angular":
        if shotNum < 95000:
            EPWDisp = 0.214116
            # EPWoff = 449.5272
            EPWoff = 449.5272
        elif shotNum < 105000:
            EPWDisp = 0.2129
            EPWoff = 439.8
        else:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.2129
            EPWoff = 439.8

        IAWDisp = 1  # dummy since ARTS does not measure ion spectra
        IAWoff = 1  # dummy
        stddev["spect_stddev_ion"] = 1  # dummy
        magE = 1  # dummy
        stddev["spect_FWHM_ele"] = 0.9  # nominally this is ~.8 or .9 for h2
        stddev["spect_stddev_ele"] = stddev["spect_FWHM_ele"] / 2.3548  # dummy
        stddev["ang_FWHM_ele"] = 1  # see Joe's FDR slides ~1-1.2
        IAWtime = 0  # means nothing here just kept to allow one code to be used for both

    elif tstype == "temporal":
        if shotNum < 105000:
            # These are valid for the 8-26-21 shot day, not sure how far back they are valid
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 523.1  # 522.90
            stddev["spect_stddev_ion"] = 0.02262  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        if shotNum <109000:
            #these are calibrations for shot 108135
            EPWDisp = 0.4104
            IAWDisp = 0.005749
            EPWoff = 319.3
            IAWoff = 523.3438  # 522.90
            stddev["spect_stddev_ion"] = 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        else:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 522.90
            stddev["spect_stddev_ion"] = 0.02262  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddev["spect_stddev_ele"] = 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21

            # Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5  # (ps / px) this is just a rough guess
            magE = 5  # (ps / px) this is just a rough guess

        IAWtime = 0  # temporal offset between EPW ross and IAW ross (varies shot to shot, can potentially add a fix based off the fiducials)

    else:
        if shotNum < 105000:
            EPWDisp = 0.27093
            IAWDisp = 0.00438
            EPWoff = 396.256  # needs to be checked
            IAWoff = 524.275

            stddev["spect_stddev_ion"] = 0.028  # needs to be checked
            stddev["spect_stddev_ele"] = 1.4365  # needs to be checked

            magI = 2.87  # um / px
            magE = 5.10  # um / px

            EPWtcc = 1024 - 456.1  # 562;
            IAWtcc = 1024 - 519  # 469;

        else:
            # needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.27093
            IAWDisp = 0.00438
            EPWoff = 396.256  # needs to be checked
            IAWoff = 524.275

            stddev["spect_stddev_ion"] = 0.028  # needs to be checked
            stddev["spect_stddev_ele"] = 1.4365  # needs to be checked

            magI = 2.87  # um / px
            magE = 5.10  # um / px

            EPWtcc = 1024 - 456.1  # 562;
            IAWtcc = 1024 - 519  # 469;

        IAWtime = 0  # means nothing here just kept to allow one code to be used for both

    ## Apply calibrations
    axisy = np.arange(1, CCDsize[0] + 1)
    axisyE = axisy * EPWDisp + EPWoff  # (nm)
    axisyI = axisy * IAWDisp + IAWoff  # (nm)

    if tstype != "angular":
        axisx = np.arange(1, CCDsize[1] + 1)
        axisxE = axisx * magE  # ps,um
        axisxI = axisx * magI  # ps,um
        if tstype == "imaging":
            axisxE = axisxE - EPWtcc * magE
            axisxI = axisxI - IAWtcc * magI
            axisxI = axisxI + 200
    else:
        imp = sio.loadmat(join("files", "angsFRED.mat"), variable_names="angsFRED")
        axisxE = imp["angsFRED"][0, :]
        # axisxE = np.vstack(np.loadtxt("files/angsFRED.txt"))
        axisxI = np.arange(1, CCDsize[1] + 1)

    return axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev


def get_scattering_angles(config):
    if config["other"]["extraoptions"]["spectype"] != "angular":
        if config["data"]["probe_beam"] == "P9":
            # Scattering angle in degrees for OMEGA TIM6 TS
            sa = dict(
                sa=np.linspace(53.637560, 66.1191, 10),
                weights=np.array(
                    [
                        0.00702671050853565,
                        0.0391423809738300,
                        0.0917976667717670,
                        0.150308544660150,
                        0.189541011666141,
                        0.195351560740507,
                        0.164271879645061,
                        0.106526733030044,
                        0.0474753389486960,
                        0.00855817305526778,
                    ]
                ),
            )
        if config["data"]["probe_beam"] == "B15":
            # Scattering angle in degrees for OMEGA TIM6 TS
            sa = dict(
                sa=np.linspace(12.0404, 24.0132, 10),
                weights=np.array(
                    [
                        0.0093239,
                        0.04189,
                        0.0912121,
                        0.145579,
                        0.182019,
                        0.188055,
                        0.163506,
                        0.1104,
                        0.0546822,
                        0.0133327,
                    ]
                ),
            )
        if config["data"]["probe_beam"] == "B23":
            # Scattering angle in degrees for OMEGA TIM6 TS
            sa = dict(
                sa=np.linspace(72.281, 84.3307, 10),
                weights=np.array(
                    [
                        0.00945903,
                        0.0430611,
                        0.0925634,
                        0.146705,
                        0.182694,
                        0.1881,
                        0.162876,
                        0.109319,
                        0.0530607,
                        0.0121616,
                    ]
                ),
            )
        if config["data"]["probe_beam"] == "B26":
            # Scattering angle in degrees for OMEGA TIM6 TS
            sa = dict(
                sa=np.linspace(55.5636, 68.1058, 10),
                weights=np.array(
                    [
                        0.00648619,
                        0.0386019,
                        0.0913923,
                        0.150489,
                        0.190622,
                        0.195171,
                        0.166389,
                        0.105671,
                        0.0470249,
                        0.00815279,
                    ]
                ),
            )
        if config["data"]["probe_beam"] == "B58":
            # Scattering angle in degrees for OMEGA TIM6 TS
            sa = dict(
                sa=np.linspace(119.093, 131.666, 10),
                weights=np.array(
                    [
                        0.00549525,
                        0.0337372,
                        0.0819783,
                        0.140084,
                        0.186388,
                        0.19855,
                        0.174136,
                        0.117517,
                        0.0527003,
                        0.00941399,
                    ]
                ),
            )
        else:
            raise NotImplmentedError("Other probe geometrries are not yet supported")
    else:
        # Scattering angle in degrees for Artemis
        imp = sio.loadmat(join("files", "angleWghtsFredfine.mat"), variable_names="weightMatrix")
        weights = imp["weightMatrix"]
        sa = dict(sa=np.arange(19, 139.5, 0.5), weights=weights)
    return sa
