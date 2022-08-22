#Supplies wavelength, space, time, and throughput calibrations based off of shot numbers including historical values
#new calibration values should be added here as they are calculated
import numpy as np

def getCalibrations(shotNum,tstype,CCDsize):

    # Dispersions and calibrations
    if tstype == 1:
        if shotNum <9500:
            EPWDisp = .214116
            EPWoff = 449.5272
        elif shotNum <105000:
            EPWDisp = .2129
            EPWoff = 439.8
        else:
            #needs to be updated with the calibrations from 7-26-22
            EPWDisp = .2129
            EPWoff = 439.8
        
        
        IAWDisp = 1 #dummy since ARTS does not measure ion spectra
        IAWoff = 1 #dummy
        stddevI = 1 #dummy
        stddevE = .9  # nominally this is ~.8 or .9 for h2
        angularFWHM = 1  # see Joe's FDR slides ~1-1.2
        IAWtime = 0 # means nothing here just kept to allow one code to be used for both

    elif tstype == 2:
        if shotNum <105000:
            #These are valid for the 8-26-21 shot day, not sure how far back they are valid
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 522.90
            stddevI = .02262 # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddevE = 1.4294 # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
            
            #Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5 # (ps / px) this is just a rough guess
            magE = 5 # (ps / px) this is just a rough guess
        
        else:
            #needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.4104
            IAWDisp = 0.00678
            EPWoff = 319.3
            IAWoff = 522.90
            stddevI = .02262 # spectral IAW IRF for 8 / 26 / 21(grating was masked)
            stddevE = 1.4294 # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
            
            #Sweep speed calculated from 5 Ghz comb (should be updated, date unknown)
            magI = 5 # (ps / px) this is just a rough guess
            magE = 5 # (ps / px) this is just a rough guess

        IAWtime = 0 # temporal offset between EPW ross and IAW ross (varies shot to shot, can potentially add a fix based off the fiducials)

    else:
        if shotNum <105000:
            EPWDisp = 0.27093
            IAWDisp = 0.00438
            EPWoff = 396.256 # needs to be checked
            IAWoff = 524.275

            stddevI = .028 # needs to be checked
            stddevE = 1.4365 # needs to be checked

            magI = 2.87 # um / px
            magE = 5.10 # um / px

            EPWtcc = 1024 - 456.1 # 562;
            IAWtcc = 1024 - 519 # 469;
        
        else:
            #needs to be updated with the calibrations from 7-26-22
            EPWDisp = 0.27093
            IAWDisp = 0.00438
            EPWoff = 396.256 # needs to be checked
            IAWoff = 524.275

            stddevI = .028 # needs to be checked
            stddevE = 1.4365 # needs to be checked

            magI = 2.87 # um / px
            magE = 5.10 # um / px

            EPWtcc = 1024 - 456.1 # 562;
            IAWtcc = 1024 - 519 # 469;
            
        IAWtime = 0 # means nothing here just kept to allow one code to be used for both
        
    ## Apply calibrations
    axisy = np.arange(1,CCDsize[0]+1)
    axisyE = axisy * EPWDisp + EPWoff #(nm)
    axisyI = axisy * IAWDisp + IAWoff #(nm)

    if tstype > 1:
        axisx = np.arange(1,CCDsize[1]+1)
        axisxE = axisx * magE #ps,um
        axisxI = axisx * magI #ps,um
        if tstype==3:
            axisxE = axisxE - EPWtcc * magE
            axisxI = axisxI - IAWtcc * magI
            axisxI = axisxI + 200
    else:
        axisxE = np.vstack(np.loadtxt("files/angsFRED.txt"))
        axisxI = np.arange(1,CCDsize[1]+1)
        
    stddev=[stddevE, stddevI]
        
    return axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev