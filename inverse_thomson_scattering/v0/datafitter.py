## function definition
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.optimize as spopt
from jax import numpy as jnp
from jax import jit
from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.v0.loadTSdata import loadData
from inverse_thomson_scattering.v0.correctThroughput import correctThroughput
from inverse_thomson_scattering.v0.getCalibrations import getCalibrations
from inverse_thomson_scattering.v0.plotters import LinePlots
from inverse_thomson_scattering.v0.numDistFunc import NumDistFunc
from inverse_thomson_scattering.v0.form_factor import nonMaxwThomson
from inverse_thomson_scattering.jax.form_factor import get_form_factor_fn


def dattafitter(shotNum, bgShot, lineoutloc, bgloc, bgscale, dpixel, TSinputs, extraoptions):

    ## function description [from Ang](update once complete)
    # This function takes the inputs from the ANGTSDATAFITTERGUI and preforms
    # the data corrections then fits the data returning the fit result
    #
    # The inputs from the GUI are Shot number, lineout locations, background shot
    # number, probe wavelength, electron temperature, electron density, m, amp1
    # and amp2, ionization state, starting distribution function
    # type and the number of distribution function points to use in numerical
    # distribution function fitting.

    # Summary of additional needs:
    #       A wrapper to allow for multiple lineouts or shots to be analyzed and gradients to be handled
    #       A way to store shot data from one call to the next (this code is frequently used on the same shot repeatedly)
    #       Better way to handle data finding since the location may change with computer or on a shot day
    #       Better way to hadnle shots with multiple types of data
    #       Way to handle calibrations which change from one to shot day to the next and have to be recalculated frequently (adding a new function to attempt this 8/8/22)
    #       Potentially move the default values, especially the calibration into the input file
    #       A way to handle the expanded ion calculation when colapsing the spectrum to pixel resolution
    #       A way to handle different numbers of points

    # Depreciated functions that need to be restored:
    #    Streaked EPW warp correction
    #    Time axis alignment with fiducials
    #    persistents
    #    persistends in numDistFunc
    #    interactive confirmation of new table creation
    #    ability to generate different table names without the default values

    ## Persistents
    # used to prevent reloading and one time analysis (not sure if there is a way to do this in python, omitted for now)
    # persistent prevShot

    ## Hard code toggles
    # collection of the toggles that are often changed and should eventually be moved out into an input deck
    tstype = extraoptions["spectype"]
    # tstype = 2 #1 for ARTS, 2 for TRTS, 3 for SRTS

    ## Minimizer options
    # minimizer options needs to be completely redone for the new minimizer
    # options = optimoptions( @ fmincon, 'Display', 'iter', 'PlotFcn', [], 'UseParallel', true, 'MaxIter', 300,\
    #          'MaxFunEval', 10000, 'TolX', 1e-10)
    # options = optimoptions( @ fmincon, 'Display', 'iter', 'PlotFcn', [], \
    #         'UseParallel', false, 'MaxIter', 1, 'MaxFunEval', 10000, 'TolX', 1e-10)
    # options = optimoptions( @ fmincon, 'Display', 'off', 'PlotFcn', [], ...
    # 'UseParallel', true, 'MaxIter', 300, 'MaxFunEval', 10000, 'TolX', 1e-10);

    ## Hard code locations and values
    # these should only be changed if something changes in the experimental setup or data is moved around
    # These are the detector info and fitting options
    D = dict([])
    D["Detector"] = "ideal"
    D["BinWidth"] = 10
    D["NumBinInRng"] = 0
    D["TotalNumBin"] = 1023
    PhysParams = dict([])
    D["expandedions"] = False

    D["extraoptions"] = extraoptions
    if "loadspecs" not in D["extraoptions"].keys():
        # This defines the spectra to be loaded and plotted [IAW, EPW]
        D["extraoptions"]["loadspecs"] = [1, 1]
    if "fitspecs" in D["extraoptions"].keys():
        if D["extraoptions"]["loadspecs"][0] == 0 and D["extraoptions"]["fitspecs"][0] == 1:
            D["extraoptions"]["fitspecs"][0] = 0
            print("IAW data not loaded, omitting IAW fit")
        if D["extraoptions"]["loadspecs"][1] == 0 and (
            D["extraoptions"]["fitspecs"][1] == 1 or D["extraoptions"]["fitspecs"][1] == 1
        ):
            D["extraoptions"]["fitspecs"][1] = 0
            D["extraoptions"]["fitspecs"][2] = 0
            print("EPW data not loaded, omitting EPW fit")
    else:
        # This defines the spectrum to be fit [IAW, EPWb, EPWr]
        D["extraoptions"]["fitspecs"] = [0, 1, 1]

    norm2B = 0  # 0 no normalization
    # 1 norm to blue
    # 2 norm to red
    # not sure if norm2B is ever changed, might be worth removing

    feDecreaseStrict = 1  # forces the result to have a decreasing distribution function(no bumps)

    TSinputs["fe"]["Length"] = 3999
    CCDsize = [1024, 1024]  # dimensions of the CCD chip as read
    shift_zero = 0

    # need a better way to do this
    shotDay = 0  # turn on to switch file retrieval to shot day location

    gain = 1
    bgscalingE = bgscale  # multiplicitive factor on the EPW BG lineout
    bgscalingI = 0.1  # multiplicitive factor on the IAW BG lineout
    bgshotmult = 1
    flatbg = 0

    # Scattering angle in degrees for TIM6 TS
    if tstype > 1:
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
    else:
        sa = dict(sa=np.arange(19, 139, 0.5), weights=np.vstack(np.loadtxt("files/angleWghtsFredfine.txt")))

    # Define jet colormap with 0=white (this might be moved and just loaded here)
    upper = mpl.cm.jet(np.arange(256))
    lower = np.ones((int(256 / 16), 4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

    # combine parts of colormap
    cmap = np.vstack((lower, upper))

    # convert to matplotlib colormap
    cmap = mpl.colors.ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])

    # Retrieve calibrated axes
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = getCalibrations(shotNum, tstype, CCDsize)

    ## Data loading and corrections
    # Open data stored from the previous run (inactivated for now)
    # if isfield(prevShot, 'shotNum') & & prevShot.shotNum == shotNum
    #    elecData = prevShot.elecData;
    #    ionData = prevShot.ionData;
    #    xlab = prevShot.xlab;
    #    shift_zero = prevShot.shift_zero;

    [elecData, ionData, xlab, shift_zero] = loadData(shotNum, shotDay, tstype, magE, D["extraoptions"]["loadspecs"])

    if D["extraoptions"]["loadspecs"][1]:
        elecData = correctThroughput(elecData, tstype, axisyE)
    # prevShot.shotNum = shotNum;
    # prevShot.elecData = elecData;
    # prevShot.ionData = ionData;
    # prevShot.xlab = xlab;
    # prevShot.shift_zero = shift_zero;

    ## Background Shot subtraction
    if bgShot["type"] == "Shot":
        [BGele, BGion, _, _] = loadData(bgShot["val"], shotDay, specType, magE, D["extraoptions"]["loadspecs"])
        if D["extraoptions"]["loadspecs"][0]:
            ionData_bsub = ionData - conv2(BGion, np.ones([5, 3]) / 15, mode="same")
        if D["extraoptions"]["loadspecs"][1]:
            BGele = correctThroughput(BGele, tstype, axisyE)
            if specType == 1:
                elecData_bsub = elecData - bgshotmult * conv2(BGele, np.ones([5, 5]) / 25, mode="same")
            else:
                elecData_bsub = elecData - bgshotmult * conv2(BGele, np.ones([5, 3]) / 15, mode="same")

    else:
        elecData_bsub = elecData
        ionData_bsub = ionData

    ## Assign lineout locations
    if lineoutloc["type"] == "ps":
        LineoutPixelE = np.argmin(abs(axisxE - lineoutloc["val"] - shift_zero))
        LineoutPixelI = LineoutPixelE

    elif lineoutloc["type"] == "um":  # [char(hex2dec('03bc')) 'm']:
        LineoutPixelE = np.argmin(abs(axisxE - lineoutloc["val"]))
        LineoutPixelI = np.argmin(abs(axisxI - lineoutloc["val"]))

    elif lineoutloc["type"] == "pixel":
        LineoutPixelE = lineoutloc["val"]
        LineoutPixelI = LineoutPixelE

    if bgloc["type"] == "ps":
        BackgroundPixel = np.argmin(abs(axisxE - bgloc["val"]))

    elif bgloc["type"] == "pixel":
        BackgroundPixel = bgloc["val"]

    elif bgloc["type"] == "auto":
        BackgroundPixel = LineoutPixelE + 100

    span = 2 * dpixel + 1
    # (span must be odd)

    if D["extraoptions"]["loadspecs"][1]:
        LineoutTSE = np.mean(elecData_bsub[:, LineoutPixelE - dpixel : LineoutPixelE + dpixel], axis=1)
        LineoutTSE_smooth = np.convolve(LineoutTSE, np.ones(span) / span, "same")

    if D["extraoptions"]["loadspecs"][0]:
        LineoutTSI = np.mean(
            ionData_bsub[:, LineoutPixelI - IAWtime - dpixel : LineoutPixelI - IAWtime + dpixel], axis=1
        )
        LineoutTSI_smooth = np.convolve(
            LineoutTSI, np.ones(span) / span, "same"
        )  # was divided by 10 for some reason (removed 8-9-22)

    if bgShot["type"] == "Fit":
        if D["extraoptions"]["loadspecs"][1]:
            if specType == 1:
                [BGele, _, _, _] = loadData(bgShot["val"], shotDay, specType, magE, D["extraoptions"]["loadspecs"])
                xx = np.arange(1024)

                def qaudbg(x):
                    np.sum(
                        (elecData[1000, :] - ((x[0] * (xx - x[3]) ** 2 + x[1] * (xx - x[3]) + x[2]) * BGele[1000, :]))
                        ** 2
                    )

                corrfactor = spopt.minimize(quadbg, [0.1, 0.1, 1.15, 300])
                newBG = (
                    corrfactor.x[0] * (xx - corrfactor.x[3]) ** 2
                    + corrfactor.x[1] * (xx - corrfactor.x[3])
                    + corrfactor[2]
                ) * BGele
                elecData_bsub = elecData - newBG
            else:
                # exp2 bg seems to be the best for some imaging data while rat11 is better in other cases but should be checked in more situations
                bgfitx = np.hstack([np.arange(100, 200), np.arange(800, 1024)])

                def exp2(x, a, b, c, d):
                    return a * np.exp(b * x) + c * np.exp(d * x)

                # [expbg, _] = spopt.curve_fit(exp2,bgfitx,LineoutTSE_smooth[bgfitx])

                def power2(x, a, b, c):
                    return a * x**b + c

                # [pwerbg, _] = spopt.curve_fit(power2,bgfitx,LineoutTSE_smooth[bgfitx])

                def rat21(x, a, b, c, d):
                    return (a * x**2 + b * x + c) / (x + d)

                # [ratbg, _] = spopt.curve_fit(rat21,bgfitx,LineoutTSE_smooth[bgfitx])

                def rat11(x, a, b, c):
                    return (a * x + b) / (x + c)

                [rat1bg, _] = spopt.curve_fit(rat11, bgfitx, LineoutTSE_smooth[bgfitx])

                LineoutTSE_smooth = LineoutTSE_smooth - rat11(np.arange(1024), rat1bg)

        #### loadspecs added up to this line
    # Attempt to quantify any residual background
    # this has been switched from mean of elecData to mean of elecData_bsub 8-9-22
    noiseE = np.mean(elecData_bsub[:, BackgroundPixel - dpixel : BackgroundPixel + dpixel], 1)
    noiseE = np.convolve(noiseE, np.ones(span) / span, "same")

    def exp2(x, a, b, c, d):
        return a * np.exp(-b * x) + c * np.exp(-d * x)

    bgfitx = np.hstack(
        [np.arange(200, 480), np.arange(540, 900)]
    )  # this is specificaly targeted at streaked data, removes the fiducials at top and bottom and notch filter
    [expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[1000, 0.001, 1000, 0.001])
    noiseE = bgscalingE * exp2(np.arange(1024), *expbg)

    noiseI = np.mean(ionData_bsub[:, BackgroundPixel - dpixel : BackgroundPixel + dpixel], 1)
    noiseI = np.convolve(noiseI, np.ones(span) / span, "same")
    bgfitx = np.hstack([np.arange(200, 400), np.arange(700, 850)])
    noiseI = np.mean(noiseI[bgfitx])
    noiseI = np.ones(1024) * bgscalingI * noiseI

    # temporary constant addition to the background
    noiseE = noiseE + flatbg

    ## Plot data
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    imE = ax[0].imshow(
        conv2(elecData_bsub, np.ones([5, 3]) / 15, mode="same"),
        cmap,
        interpolation="none",
        extent=[axisxE[0] - shift_zero, axisxE[-1] - shift_zero, axisyE[-1], axisyE[0]],
        aspect="auto",
        vmin=0,
    )
    ax[0].set_title(
        "Shot : " + str(shotNum) + " : " + "TS : Thruput corrected", fontdict={"fontsize": 10, "fontweight": "bold"}
    )
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel("Wavelength (nm)")
    plt.colorbar(imE, ax=ax[0])
    ax[0].plot([axisxE[LineoutPixelE] - shift_zero, axisxE[LineoutPixelE] - shift_zero], [axisyE[0], axisyE[-1]], "r")
    ax[0].plot(
        [axisxE[BackgroundPixel] - shift_zero, axisxE[BackgroundPixel] - shift_zero], [axisyE[0], axisyE[-1]], "k"
    )

    imI = ax[1].imshow(
        conv2(ionData_bsub, np.ones([5, 3]) / 15, mode="same"),
        cmap,
        interpolation="none",
        extent=[axisxI[0] - shift_zero, axisxI[-1] - shift_zero, axisyI[-1], axisyI[0]],
        aspect="auto",
        vmin=0,
    )
    ax[1].set_title(
        "Shot : " + str(shotNum) + " : " + "TS : Thruput corrected", fontdict={"fontsize": 10, "fontweight": "bold"}
    )
    ax[1].set_xlabel(xlab)
    ax[1].set_ylabel("Wavelength (nm)")
    plt.colorbar(imI, ax=ax[1])
    ax[1].plot([axisxI[LineoutPixelI] - shift_zero, axisxI[LineoutPixelI] - shift_zero], [axisyI[0], axisyI[-1]], "r")
    ax[1].plot(
        [axisxI[BackgroundPixel] - shift_zero, axisxI[BackgroundPixel] - shift_zero], [axisyI[0], axisyI[-1]], "k"
    )

    ## Normalize Data before fitting

    noiseE = noiseE / gain
    LineoutTSE_norm = LineoutTSE_smooth / gain
    LineoutTSE_norm = LineoutTSE_norm - noiseE  # new 6-29-20
    ampE = np.max(LineoutTSE_norm[100:-1])  # attempts to ignore 3w comtamination

    noiseI = noiseI / gain
    LineoutTSI_norm = LineoutTSI_smooth / gain
    LineoutTSI_norm = LineoutTSI_norm - noiseI  # new 6-29-20
    ampI = np.max(LineoutTSI_norm)
    PhysParams = {
        "widIRF": stddev,
        "background": [0, 0],
        "amps": [ampE, ampI],
        "norm": norm2B,
    }  # {width of IRF,background , amplitude ,Normalization of peaks} new 6-29-20

    ## Setup x0
    D["lamrangE"] = [axisyE[0], axisyE[-2]]
    D["lamrangI"] = [axisyI[0], axisyI[-2]]
    D["iawoff"] = 0
    D["iawfilter"] = [1, 4, 24, 528]
    D["npts"] = (len(LineoutTSE_norm) - 1) * 20
    D["PhysParams"] = PhysParams
    data = np.vstack((LineoutTSE_norm, LineoutTSI_norm))

    xie = np.linspace(-7, 7, TSinputs["fe"]["Length"])
    initFe(TSinputs, xie)

    TSinputs["fe"]["lb"] = np.multiply(TSinputs["fe"]["lb"], np.ones(TSinputs["fe"]["length"]))
    TSinputs["fe"]["ub"] = np.multiply(TSinputs["fe"]["ub"], np.ones(TSinputs["fe"]["length"]))

    x0 = []
    lb = []
    ub = []
    for key in TSinputs.keys():
        if TSinputs[key]["active"]:
            x0.append(TSinputs[key]["val"])
            lb.append(TSinputs[key]["lb"])
            ub.append(TSinputs[key]["ub"])

    ## Plot initial guess
    fitmodel2 = get_fitModel2(TSinputs, xie, sa, D)
    plotState(x0, TSinputs, xie, sa, D, data, fitModel2=fitmodel2)
    chiSq2 = get_chisq2(TSinputs, xie, sa, D, data)
    chiinit = chiSq2(x0)
    print(chiinit)

    ## Perform fit
    if np.shape(x0)[0] != 0:
        # fun = lambda x: chiSq2(x, TSinputs, xie, sa, D, data)
        res = spopt.minimize(chiSq2, np.array(x0), bounds=zip(lb, ub))
        # print(res)
        # [x,~,~,~,~,grad,hess]=fmincon(@(x)chiSq2(x,TSinputs,xie,sa,D,data),x0,[],[],[],[],lb,ub,[],options)

        # chisq=@(x)chiSq2(x,TSinputs,xie,sa,D,data);
    else:
        x = x0

    ## Plot Result
    plotState(res.x, TSinputs, xie, sa, D, data, fitModel2=fitmodel2)
    chifin = chiSq2(res.x)  # , TSinputs, xie, sa, D, data)
    print(chifin)

    result = TSinputs
    xiter = iter(res.x)
    for key in result.keys():
        if result[key]["active"]:
            result[key]["val"] = next(xiter)
            print(key, ": ", result[key]["val"])
    if result["fe"]["active"]:
        result["fe"]["val"] = res.x[-result["fe"]["length"] : :]
    elif result["m"]["active"]:
        initFe(result, xie)

    return result


def plotState(x, TSinputs, xie, sas, D, data, fitModel2):
    # all ion terms are commented out for testing
    [modlE, lamAxisE] = fitModel2(x)

    lam = TSinputs["lam"]["val"]
    amp1 = TSinputs["amp1"]["val"]
    amp2 = TSinputs["amp2"]["val"]
    # [_,_,lamAxisE,_]=lamParse(D['lamrangE'],lam,D['npts'])
    # [omgL,omgsI,lamAxisI,_]=lamParse(D['lamrangI'],lam,D['npts'])

    # this needs to be updated
    # modlI=fitModel(Te,Ti,Z,D.A,D.fract,ne,Va,ud,omgsI,omgL,D.sa,curDist,D.distTable,0,{0},D.lamrangI,lam,lamAxisI);

    originE = (max(lamAxisE) + min(lamAxisE)) / 2  # Conceptual_origin so the convolution donsn't shift the signal
    # originI=(max(lamAxisI)+min(lamAxisI))/2 #Conceptual_origin so the convolution donsn't shift the signal

    stddev = D["PhysParams"]["widIRF"]

    inst_funcE = np.squeeze(
        (1 / (stddev[0] * np.sqrt(2 * np.pi))) * np.exp(-((lamAxisE - originE) ** 2) / (2 * (stddev[0]) ** 2))
    )  # Gaussian
    # inst_funcI = (1/(stddev[1]*np.sqrt(2*np.pi)))*np.exp(-(lamAxisI-originI)**2/(2*(stddev[1])**2)) #Gaussian

    ThryE = np.convolve(modlE, inst_funcE, "same")
    ThryE = (max(modlE) / max(ThryE)) * ThryE
    # ThryI = np.convolve(modlI, inst_funcI,'same')
    # ThryI=(max(modlI)/max(ThryI))*ThryI

    if D["PhysParams"]["norm"] > 0:
        ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam] / max(ThryE[lamAxisE < lam]))
        ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam] / max(ThryE[lamAxisE > lam]))

    # n=np.floor(len(ThryE)/len(data))
    # ThryE = np.average(ThryE.reshape(-1, n), axis=1)
    ThryE = np.average(ThryE.reshape(1024, -1), axis=1)
    # ThryE= [np.mean(ThryE[i:i+n-1]) for i in np.arange(0,len(ThryE),n)]
    # arrayfun(@(i) mean(ThryE(i:i+n-1)),1:n:length(ThryE)-n+1);
    # n=floor(length(ThryI)/length(data));
    # ThryI=arrayfun(@(i) mean(ThryI(i:i+n-1)),1:n:length(ThryI)-n+1);

    if D["PhysParams"]["norm"] == 0:
        lamAxisE = np.average(lamAxisE.reshape(1024, -1), axis=1)
        # lamAxisE=arrayfun(@(i) mean(lamAxisE(i:i+n-1)),1:n:length(lamAxisE)-n+1);
        ThryE = D["PhysParams"]["amps"][0] * ThryE / max(ThryE)
        # lamAxisI=arrayfun(@(i) mean(lamAxisI(i:i+n-1)),1:n:length(lamAxisI)-n+1);
        # ThryI = amp3*D.PhysParams{3}(2)*ThryI/max(ThryI);
        ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam])
        ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam])

    if D["extraoptions"]["spectype"] == 1:
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
        LinePlots(lamAxisE, np.vstack((data[0, :], ThryE)), CurveNames=["Data", "Fit"], XLabel="Wavelength (nm)")
        plt.xlim([450, 630])

        # LinePlots(lamAxisI,[data(2,:); ThryI],'CurveNames',{'Data','Fit'},'XLabel','Wavelength (nm)')
        # xlim([525 528])

    chisq = float("nan")
    redchi = float("nan")

    if "fitspecs" in D["extraoptions"].keys():
        chisq = 0
        # if D.extraoptions.fitspecs(1)
        #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21

        if D["extraoptions"]["fitspecs"][1]:
            # chisq=chisq+sum((data(1,lamAxisE<lam)-ThryE(lamAxisE<lam)).^2);
            chisq = chisq + sum(
                (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
            )

        if D["extraoptions"]["fitspecs"][2]:
            # chisq=chisq+sum((data(1,lamAxisE>lam)-ThryE(lamAxisE>lam)).^2);
            chisq = chisq + sum(
                (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
            )


def get_fitModel2(TSins, xie, sa, D):
    nonMaxwThomson_jax, _ = get_form_factor_fn(D["lamrangE"])

    def fitModel2(x):
        i = 0
        for key in TSins.keys():
            if TSins[key]["active"]:
                TSins[key]["val"] = x[i]
                i = i + 1
        if TSins["fe"]["active"]:
            TSins["fe"]["val"] = x[-TSins["fe"]["length"] : :]
        elif TSins["m"]["active"]:
            initFe(TSins, xie)

        # [Te,ne]=TSins.genGradients(Te,ne,7)
        fecur = jnp.exp(TSins["fe"]["val"])
        lam = TSins["lam"]["val"]

        # Thry, lamAxisE = nonMaxwThomson(
        #     TSins["Te"]["val"],
        #     TSins["Te"]["val"],
        #     1,
        #     1,
        #     1,
        #     TSins["ne"]["val"] * 1e20,
        #     0,
        #     0,
        #     D["lamrangE"],
        #     lam,
        #     sa["sa"],
        #     fecur,
        #     xie,
        #     expion=D["expandedions"],
        # )
        # Te, Ti, Z, A, fract, ne, Va, ud, sa, fe, lamrang, lam
        Thry, lamAxisE = nonMaxwThomson_jax(
            TSins["Te"]["val"],
            TSins["Te"]["val"],
            1,
            1,
            1,
            TSins["ne"]["val"] * 1e20,
            0,
            0,
            sa["sa"],
            (fecur, xie),
            lam,
            # ,
            # expion=D["expandedions"],
        )

        # if TSins.fe['Type']=='MYDLM':
        #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,TSins.fe['thetaphi'])
        # elif TSins.fe['Type']=='Numeric':
        #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,[2*np.pi/3,0])
        # else:
        #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,expion=D['expandedions'])
        # nonMaxwThomson,_ =get_form_factor_fn(D['lamrangE'],lam)
        # [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,sa['sa'], [fecur,xie])

        # remove extra dimensions and rescale to nm
        lamAxisE = jnp.squeeze(lamAxisE) * 1e7

        Thry = jnp.real(Thry)
        Thry = jnp.mean(Thry, axis=0)
        modlE = jnp.sum(Thry * sa["weights"], axis=1)

        # [modl,lamAx]=S2Signal(Thry,lamAxis,D);
        # [_,_,lamAxisE,_]=lamParse(D['lamrangE'],lam,D['npts'])
        if D["iawoff"] and (D["lamrangE"][0] < lam and D["lamrangE"][1] > lam):
            # set the ion feature to 0 #should be switched to a range about lam
            lamloc = jnp.argmin(jnp.abs(lamAxisE - lam))
            # lamloc=find(abs(lamAxisE-lam)<(lamAxisE(2)-lamAxisE(1)));
            # modlE[lamloc - 2000 : lamloc + 2000] = 0
            modlE = jnp.concatenate([modlE[: lamloc - 2000], jnp.zeros(4000), modlE[lamloc + 2000 :]])

        if D["iawfilter"][0]:
            filterb = D["iawfilter"][3] - D["iawfilter"][2] / 2
            filterr = D["iawfilter"][3] + D["iawfilter"][2] / 2
            if D["lamrangE"][0] < filterr and D["lamrangE"][1] > filterb:
                if D["lamrangE"][0] < filterb:
                    lamleft = jnp.argmin(jnp.abs(lamAxisE - filterb))
                else:
                    lamleft = 0

                if D["lamrangE"][1] > filterr:
                    lamright = jnp.argmin(jnp.abs(lamAxisE - filterr))
                else:
                    lamright = lamAxisE.size
                    # lamright=[0 length(lamAxisE)];

                # modlE[lamleft:lamright] = modlE[lamleft:lamright] * 10 ** (-D["iawfilter"][1])
                modlE = jnp.concatenate(
                    [modlE[:lamleft], modlE[lamleft:lamright] * 10 ** (-D["iawfilter"][1]), modlE[lamright:]]
                )

        return modlE, lamAxisE

    return fitModel2


def get_chisq2(TSinputs, xie, sas, D, data):

    fitModel2 = get_fitModel2(TSinputs, xie, sas, D)

    def chiSq2(x):

        # all ion terms are commented out for testing
        modlE, lamAxisE = fitModel2(x)

        lam = TSinputs["lam"]["val"]
        amp1 = TSinputs["amp1"]["val"]
        amp2 = TSinputs["amp2"]["val"]
        # [_,_,lamAxisE,_]=lamParse(D['lamrangE'],lam,D['npts'])
        # [omgL,omgsI,lamAxisI,_]=lamParse(D['lamrangI'],lam,D['npts'])

        # this needs to be updated
        # modlI=fitModel(Te,Ti,Z,D.A,D.fract,ne,Va,ud,omgsI,omgL,D.sa,curDist,D.distTable,0,{0},D.lamrangI,lam,lamAxisI);

        # Conceptual_origin so the convolution donsn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        # originI=(max(lamAxisI)+min(lamAxisI))/2 #Conceptual_origin so the convolution donsn't shift the signal

        stddev = D["PhysParams"]["widIRF"]

        inst_funcE = jnp.squeeze(
            (1.0 / (stddev[0] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddev[0]) ** 2.0))
        )  # Gaussian
        # inst_funcI = (1/(stddev[1]*jnp.sqrt(2*jnp.pi)))*jnp.exp(-(lamAxisI-originI)**2/(2*(stddev[1])**2)) #Gaussian

        ThryE = jnp.convolve(modlE, inst_funcE, "same")
        ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE
        # ThryI = jnp.convolve(modlI, inst_funcI,'same')
        # ThryI=(max(modlI)/max(ThryI))*ThryI

        if D["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                amp1 * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                amp2 * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )
            # ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam] / jnp.amax(ThryE[lamAxisE < lam]))
            # ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam] / jnp.amax(ThryE[lamAxisE > lam]))

        # n=jnp.floor(len(ThryE)/len(data))
        # ThryE = jnp.average(ThryE.reshape(-1, n), axis=1)
        ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
        # ThryE= [jnp.mean(ThryE[i:i+n-1]) for i in jnp.arange(0,len(ThryE),n)]
        # arrayfun(@(i) mean(ThryE(i:i+n-1)),1:n:length(ThryE)-n+1);
        # n=floor(length(ThryI)/length(data));
        # ThryI=arrayfun(@(i) mean(ThryI(i:i+n-1)),1:n:length(ThryI)-n+1);

        if D["PhysParams"]["norm"] == 0:
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
            # lamAxisE=arrayfun(@(i) mean(lamAxisE(i:i+n-1)),1:n:length(lamAxisE)-n+1);
            ThryE = D["PhysParams"]["amps"][0] * ThryE / jnp.amax(ThryE)
            # lamAxisI=arrayfun(@(i) mean(lamAxisI(i:i+n-1)),1:n:length(lamAxisI)-n+1);
            # ThryI = amp3*D.PhysParams{3}(2)*ThryI/max(ThryI);
            # ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam])
            # ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam])

            ThryE = jnp.where(
                lamAxisE < lam,
                amp1 * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                amp2 * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        chisq = jnp.nan
        redchi = jnp.nan

        if "fitspecs" in D["extraoptions"].keys():
            chisq = 0
            # if D.extraoptions.fitspecs(1)
            #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21

            if D["extraoptions"]["fitspecs"][1]:
                # chisq=chisq+sum((data(1,lamAxisE<lam)-ThryE(lamAxisE<lam)).^2);
                chisq = chisq + jnp.sum(
                    (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
                )

            if D["extraoptions"]["fitspecs"][2]:
                # chisq=chisq+sum((data(1,lamAxisE>lam)-ThryE(lamAxisE>lam)).^2);
                chisq = chisq + jnp.sum(
                    (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
                )

        return chisq

    return chiSq2


def initFe(TSinputs, xie):
    # generate fe from inputs or keep numerical fe
    if TSinputs["fe"]["type"] == "DLM":
        TSinputs["fe"]["val"] = np.log(
            NumDistFunc([TSinputs["fe"]["type"], TSinputs["m"]["val"]], xie, TSinputs["fe"]["type"])
        )

    elif TSinputs["fe"]["type"] == "Fourkal":
        TSinputs["fe"]["val"] = np.log(
            NumDistFunc(
                [TSinputs["fe"]["type"], TSinputs["m"]["val"], TSinputs["Z"]["val"]], xie, TSinputs["fe"]["type"]
            )
        )

    elif TSinputs["fe"]["type"] == "SpitzerDLM":
        TSinputs["fe"]["val"] = np.log(
            NumDistFunc(
                [TSinputs["fe"]["type"], TSinputs["m"]["val"], TSinputs["fe"]["theta"], TSinputs["fe"]["delT"]],
                xie,
                TSinputs["fe"]["type"],
            )
        )

    elif TSinputs["fe"]["type"] == "MYDLM":  # This will eventually need another parameter for density gradient
        TSinputs["fe"]["val"] = np.log(
            NumDistFunc(
                [TSinputs["fe"]["type"], TSinputs["m"]["val"], TSinputs["fe"]["theta"], TSinputs["fe"]["delT"]],
                xie,
                TSinputs["fe"]["type"],
            )
        )

    else:
        raise NameError("Unrecognized distribtuion function type")

    TSinputs["fe"]["val"][TSinputs["fe"]["val"] <= -100] = -99
