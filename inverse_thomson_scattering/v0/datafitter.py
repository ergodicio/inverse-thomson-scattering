## function definition
def dattafitter(shotNum, bgShot, lineoutloc, bgloc, bgscale, dpixel, TSinputs, extraoptions):

    import numpy as np
    ## function description [from Ang](update once complete)
    # This function takes the inputs from the ANGTSDATAFITTERGUI and preforms
    # the data corrections then fits the data retunring the fit result
    #
    #The inputs from the GUI are Shot number, linout locations, background shot
    #number, probe wavelength, electron temperature, electron density, m, amp1
    #and amp2, ionization state, starting distribution function
    #type and the number of distribution function points to use in numerical
    #distribution function fitting.

    ## Persistents [from Omg]
    # used to prevent reloading and one time analysis (not sure if there is a way to do this in python)
    persistent prevShot

    ## Hard code locations and values [from Ang]
    # these should only be changed if something changes in the experimental setup or data is moved around
    # These are the detector info and fitting options
    # make the structure D into a dictionary?
    D.Detector = 'ideal'
    D.BinWidth = 10
    D.NumBinInRng = 0
    D.TotalNumBin = 1023

    spectralFWHM = .9 # nominaly this is ~.8 or .9 for h2
    angularFWHM = 1 # see Joe's FDR slides ~1-1.2

    norm2B = 0 # 0 no normalization
                # 1 norm to blue
                # 2 norm to red
    #not sure if norm2B is ever changed, might be worth removing

    #minimizer options needs to be completely redone for the new minimizer
    options = optimoptions( @ fmincon, 'Display', 'iter', 'PlotFcn', [], 'UseParallel', true, 'MaxIter', 300,\
              'MaxFunEval', 10000, 'TolX', 1e-10)
    # options = optimoptions( @ fmincon, 'Display', 'iter', 'PlotFcn', [], \
            # 'UseParallel', false, 'MaxIter', 1, 'MaxFunEval', 10000, 'TolX', 1e-10)

    scaterangs = np.arange(19,139,.5)

    #[from Omg]
    feDecreaseStrict = 1 # forces the result to have a decreasing distribution function(no bumps)

    TSinputs.fe.Length = 3999
    CCDsize = [1024, 1024] # dimensions of the CCD chip as read
    shift_zero = 0

    # temporary hard code of the gradients (these are from the RCI campaign and this section needs to be reworked)
tegrads = [13.4    12.1    15.5    12.6    5.3    8.7    11    7.7    5.7    7.7    11    8.7    5.3    12.6    15.5
           12.1    13.4];
negrads = [8.3    2.7    5.1    1.9    5.8    2.7    5.4    2.9    2.3    2.9    5.4    2.7    5.8    1.9    5.1    2.7
           8.3];
% tegrads = [24.57335013    22.73347057    20.90494825    19.09119797    17.29471889    15.51695032    13.75912199
             12.02528956    10.33099953    8.728459334    7.397509239    7.096308139    11.64464733    7.427516264
             7.184179079    8.390366998 9.955503333];
% negrads = [7.997358404    7.531129172    7.098102129    6.74884386    6.538674831    6.43573586    6.276988137
             5.902481943    5.309826692    4.63467171    4.031271811    3.694117652    4.621808592    3.711504395
             3.91991231    4.486187408    5.157712433];
ii = lineoutloc.val / 100 + 9;
% TSinputs = TSinputs.addGradients(1, tegrads(ii), 1, negrads(ii));
% TSinputs = TSinputs.addGradients(1, 12.8, 1, 8.5);

shotDay = 0; % run
on
to
switch
file
retrieval
to
shot
day
location

load('MyColormaps_TS', 'TS_mycmap');

gain = 1;
bgscalingE = bgscale; % multiplicitive
factor
on
the
EPW
BG
lineout
bgscalingI = .1; % multiplicitive
factor
on
the
IAW
BG
lineout
bgshotmult = 1;
flatbg = 0;

options = optimoptions( @ fmincon, 'Display', 'off', 'PlotFcn', [], ...
'UseParallel', true, 'MaxIter', 300, 'MaxFunEval', 10000, 'TolX', 1e-10);

% Scattering
angle in degrees
sa.sa = linspace(53.637560, 66.1191, 10); % need
the
exact
for P9 and the f / numbers
sa.weights = [0.00702671050853565;
0.0391423809738300;
0.0917976667717670;
0.150308544660150;
0.189541011666141;
0.195351560740507;
0.164271879645061;
0.106526733030044;
0.0474753389486960;
0.00855817305526778];

% Dispersions and calibrations
if strcmp(extraoptions.spectype, 'Streaked')
    EPWDisp = 0.4104;
    IAWDisp = 0.00678;
    EPWoff = 319.3;
    IAWoff = 522.90;
    stddevI = .02262; % spectral
    IAW
    IRF
    for 8 / 26 / 21(grating was masked)
    stddevE = 1.4294; % spectral
    EPW
    IRF
    for 200um pinhole used on 8 / 26 / 21

    IAWtime = 0; % temporal
    offset
    between
    EPW
    ross and IAW
    ross(does
    not appear
    to
    be
    consistent)
    % Sweep
    speed
    calculated
    from

    5
    Ghz
    comb
    magI = 5; % (ps / px)
    this is just
    a
    rough
    guess
    magE = 5; % (ps / px)
    this is just
    a
    rough
    guess
else
    EPWDisp = 0.27093;
    IAWDisp = 0.00438;
    EPWoff = 396.256; % needs
    to
    be
    checked
    IAWoff = 524.275;

    stddevI = .028; % needs
    to
    be
    checked
    stddevE = 1.4365; % needs
    to
    be
    checked

    IAWtime = 0; % means
    nothing
    here
    just
    kept
    to
    allow
    one
    code
    to
    be
    used
    for both
        magI = 2.87; % um / px
    magE = 5.10; % um / px

    EPWtcc = 1024 - 456.1; % 562;
    IAWtcc = 1024 - 519; % 469;
end