import matplotlib.pyplot as plt
import numpy as np
from inverse_thomson_scattering.v0.plotters import LinePlots

def plotState(x, TSinputs, xie, sas, data, fitModel2):
    [modlE, modlI, lamAxisE, lamAxisI] = fitModel2(x)
    
    lam = TSinputs["lam"]["val"]
    amp1 = TSinputs["amp1"]["val"]
    amp2 = TSinputs["amp2"]["val"]
    amp3 = TSinputs["amp3"]["val"]

    stddev = TSinputs['D']["PhysParams"]["widIRF"]
    
    if TSinputs['D']["extraoptions"]["load_ion_spec"]:
        originI = (max(lamAxisI) + min(lamAxisI)) / 2 #Conceptual_origin so the convolution donsn't shift the signal
        inst_funcI = np.squeeze(
        (1 / (stddev[1] * np.sqrt(2 * np.pi))) * np.exp(-((lamAxisI - originI) ** 2) / (2 * (stddev[1]) ** 2))
        )#Gaussian
        ThryI = np.convolve(modlI, inst_funcI,'same')
        ThryI = (max(modlI) / max(ThryI)) * ThryI
        ThryI = np.average(ThryI.reshape(1024, -1), axis=1)
        
        if TSinputs['D']["PhysParams"]["norm"] == 0:
            lamAxisI = np.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = amp3 * TSinputs['D']["PhysParams"]["amps"][1] * ThryI / max(ThryI)

        
    if TSinputs['D']["extraoptions"]["load_ele_spec"]:
        originE = (max(lamAxisE) + min(lamAxisE)) / 2  # Conceptual_origin so the convolution donsn't shift the signal
        inst_funcE = np.squeeze(
        (1 / (stddev[0] * np.sqrt(2 * np.pi))) * np.exp(-((lamAxisE - originE) ** 2) / (2 * (stddev[0]) ** 2))
        )  # Gaussian
        ThryE = np.convolve(modlE, inst_funcE, "same")
        ThryE = (max(modlE) / max(ThryE)) * ThryE
        
        if TSinputs['D']["PhysParams"]["norm"] > 0:
            ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam] / max(ThryE[lamAxisE < lam]))
            ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam] / max(ThryE[lamAxisE > lam]))
            
        ThryE = np.average(ThryE.reshape(1024, -1), axis=1)
        if TSinputs['D']["PhysParams"]["norm"] == 0:
            lamAxisE = np.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = TSinputs['D']["PhysParams"]["amps"][0] * ThryE / max(ThryE)
            ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam])
            ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam])


    if TSinputs['D']["extraoptions"]["spectype"] == 1:
        print("colorplot still needs to be written")
        # Write Colorplot
        # Thryinit=ArtemisModel(TSinputs,xie,scaterangs,x0,weightMatrix,...
    #    spectralFWHM,angularFWHM,lamAxis,xax,D,norm2B);
    # if ~norm2B
    #    Thryinit=Thryinit./max(Thryinit(470:900,:));
    #    Thryinit=Thryinit.*max(data(470:900,:));
    #    Thryinit=TSinputs.amp1.Value*Thryinit;
    # end
    # chisq = sum(sum((data([40:330 470:900],90:1015)-Thryinit([40:330 470:900],90:1015)).^2));
    # Thryinit(330:470,:)=0;
    #
    # ColorPlots(yax,xax,rot90(Thryinit),'Kaxis',[TSinputs.ne.Value*1E20,TSinputs.Te.Value,526.5],...
    #    'Title','Starting point','Name','Initial Spectrum');
    # ColorPlots(yax,xax,rot90(data-Thryinit),'Title',...
    #    ['Initial difference: \chi^2 =' num2str(chisq)],'Name','Initial Difference');
    # load('diffcmap.mat','diffcmap');
    # colormap(diffcmap);

    # if norm2B
    #    caxis([-1 1]);
    # else
    #    caxis([-8000 8000]);
    # end
    else:
        if TSinputs['D']["extraoptions"]["load_ion_spec"]:
            LinePlots(lamAxisI, np.vstack((data[1, :], ThryI)), CurveNames=["Data", "Fit"], XLabel="Wavelength (nm)")
            plt.xlim([525, 528])
            
        if TSinputs['D']["extraoptions"]["load_ele_spec"]:
            LinePlots(lamAxisE, np.vstack((data[0, :], ThryE)), CurveNames=["Data", "Fit"], XLabel="Wavelength (nm)")
            plt.xlim([450, 630])

    #chisq = float("nan")
    #redchi = float("nan")
    
    chisq = 0
    if TSinputs['D']["extraoptions"]['fit_IAW']:
        #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
        chisq = chisq + sum((data[1, :] - ThryI) ** 2)

    if TSinputs['D']["extraoptions"]['fit_EPWb']:
        chisq = chisq + sum(
            (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
        )

    if TSinputs['D']["extraoptions"]['fit_EPWr']:
        chisq = chisq + sum(
            (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
        )