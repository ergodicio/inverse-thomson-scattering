function result=OmegaDataFitter(shotNum,bgShot,lineoutloc,bgloc,bgscale,dpixel,TSinputs,extraoptions)
tic
%% Persistents 
%used to prevent reloading and one time analysis
persistent prevShot


%% Hard code locations and values
% these should only be changed if something changes in the experimental
% setup or data is moved around
TSinputs.fe.Length=3999;
CCDsize=[1024 1024]; %dimensions of the CCD chip as read
shift_zero=0;

%temporary hard code of the gradients
tegrads=[13.4	12.1	15.5	12.6	5.3	8.7	11	7.7	5.7	7.7	11	8.7	5.3	12.6	15.5	12.1	13.4];
negrads=[8.3	2.7	5.1	1.9	5.8	2.7	5.4	2.9	2.3	2.9	5.4	2.7	5.8	1.9	5.1	2.7	8.3];
%tegrads=[24.57335013	22.73347057	20.90494825	19.09119797	17.29471889	15.51695032	13.75912199	12.02528956	10.33099953	8.728459334	7.397509239	7.096308139	11.64464733	7.427516264	7.184179079	8.390366998 9.955503333];
%negrads=[7.997358404	7.531129172	7.098102129	6.74884386	6.538674831	6.43573586	6.276988137	5.902481943	5.309826692	4.63467171	4.031271811	3.694117652	4.621808592	3.711504395	3.91991231	4.486187408	5.157712433];
ii=lineoutloc.val/100+9;
%TSinputs=TSinputs.addGradients(1,tegrads(ii),1,negrads(ii));
%TSinputs=TSinputs.addGradients(1,12.8,1,8.5);

shotDay=0; %run on to switch file retrieval to shot day location

load ('MyColormaps_TS','TS_mycmap');

gain=1;
bgscalingE=bgscale; %multiplicitive factor on the EPW BG lineout
bgscalingI=.1; %multiplicitive factor on the IAW BG lineout
bgshotmult=1;
flatbg=0;

options = optimoptions(@fmincon,'Display','off','PlotFcn',[],...
    'UseParallel',true,'MaxIter',300,'MaxFunEval',10000,'TolX',1e-10);

% Scattering angle in degrees
sa.sa=linspace(53.637560,66.1191,10);%need the exact for P9 and the f/numbers
sa.weights=[0.00702671050853565;0.0391423809738300;0.0917976667717670;0.150308544660150;0.189541011666141;0.195351560740507;0.164271879645061;0.106526733030044;0.0474753389486960;0.00855817305526778];

%Dispersions and calibrations
if strcmp(extraoptions.spectype,'Streaked')
    EPWDisp = 0.4104; 
    IAWDisp = 0.00678;
    EPWoff = 319.3;
    IAWoff = 522.90;
    stddevI=.02262; %spectral IAW IRF for 8/26/21 (grating was masked)
    stddevE=1.4294; %spectral EPW IRF for 200um pinhole used on 8/26/21

    IAWtime=0; %temporal offset between EPW ross and IAW ross (does not appear to be consistent)
    %Sweep speed calculated from 5Ghz comb
    magI = 5; %(ps/px) this is just a rough guess
    magE = 5; %(ps/px) this is just a rough guess
else
    EPWDisp = 0.27093;
    IAWDisp = 0.00438;
    EPWoff = 396.256; %needs to be checked
    IAWoff = 524.275;
    
    stddevI = .028; %needs to be checked
    stddevE = 1.4365; %needs to be checked
    
    IAWtime=0; %means nothing here just kept to allow one code to be used for both
    magI = 2.87; %um/px
    magE = 5.10; %um/px
    
    EPWtcc=1024-456.1;%562;
    IAWtcc=1024-519;%469;
end

%% Apply calibrations
axisy = 1:CCDsize(1);
axisyE = axisy .* EPWDisp + EPWoff; %(nm)
axisyI = axisy .* IAWDisp + IAWoff; %(nm)

axisx = 1:CCDsize(2);
axisxE = axisx .* magE;
axisxI = axisx .* magI;

if strcmp(extraoptions.spectype,'Imaging')
    axisxE= axisxE-EPWtcc*magE;
    axisxI= axisxI-IAWtcc*magI;
    axisxI= axisxI+200;
end

%% One time dataloading and corrections
%Load Data and correct throughput
if isfield(prevShot,'shotNum') && prevShot.shotNum==shotNum
    elecData=prevShot.elecData;
    ionData=prevShot.ionData;
    xlab=prevShot.xlab;
    shift_zero=prevShot.shift_zero;
else
    [elecData,ionData,xlab,shift_zero]=loadData(shotNum,shotDay,extraoptions.spectype,magE);
    elecData=correctThroughput(elecData,extraoptions.spectype,axisyE);
    prevShot.shotNum=shotNum;
    prevShot.elecData=elecData;
    prevShot.ionData=ionData;
    prevShot.xlab=xlab;
    prevShot.shift_zero=shift_zero;
end

%performbackgroud shot subtraction
if strcmp(bgShot.type,'Shot')
    if shotDay
        folder = '\\redwood\archive\tmp\thomson\';
        hdfnameE=strcat(strcat(folder,'epw_ccd\s',num2str(bgShot.val),'.hdf')); %for use with archive/tmp
        hdfnameI=strcat(strcat(folder,'iaw_ccd\s',num2str(bgShot.val),'.hdf'));
    else
        hdfnameE=strcat(strcat('EPW_CCD-s',num2str(bgShot.val),'.hdf'));
        hdfnameI=strcat(strcat('IAW_CCD-s',num2str(bgShot.val),'.hdf'));
    end

    BGion=double(hdfread(hdfnameI,'Streak_array'));
    BGion = squeeze(BGion(1,:,:) - BGion(2,:,:));
    BGion = rot90(BGion,3);
    ionData_bsub=ionData-filter2(ones(5,3)/15,BGion);

    BGele=double(hdfread(hdfnameE,'Streak_array'));
    BGele = squeeze(BGele(1,:,:) - BGele(2,:,:));
    BGele = rot90(BGele,3);
    BGele=correctThroughput(BGele,extraoptions.spectype,axisyE);
    elecData_bsub=elecData-bgshotmult*filter2(ones(5,3)/15,BGele);
else
    elecData_bsub=elecData;
    ionData_bsub=ionData;
end

%% Assign linout locations
if strcmp(lineoutloc.type,'ps')
    [~,LineoutPixelE]=min(abs(axisxE-lineoutloc.val-shift_zero));
    LineoutPixelI=LineoutPixelE;
end
if strcmp(lineoutloc.type,[char(hex2dec('03bc')) 'm'])
    [~,LineoutPixelE]=min(abs(axisxE-lineoutloc.val));
    [~,LineoutPixelI]=min(abs(axisxI-lineoutloc.val));
end
if strcmp(lineoutloc.type,'pixel')
    LineoutPixelE=lineoutloc.val;
    LineoutPixelI=LineoutPixelE;
end
if strcmp(bgloc.type,'ps')
    [~,BackgroundPixel]=min(abs(axisxE-bgloc.val));
end
if strcmp(bgloc.type,'pixel')
    BackgroundPixel=bgloc.val;
end
if strcmp(bgloc.type,'auto')
    BackgroundPixel=LineoutPixelE+100;
end

span = 2*dpixel +1; %(span must be odd)

LineoutTSE= mean( elecData_bsub(:,LineoutPixelE-dpixel:LineoutPixelE+dpixel) ,2 );
LineoutTSE_smooth(:) = smooth(LineoutTSE,span);

LineoutTSI= mean( ionData_bsub(:,LineoutPixelI-IAWtime-dpixel:LineoutPixelI-IAWtime+dpixel) ,2 );
LineoutTSI_smooth(:) = smooth(LineoutTSI,span)./10;

if strcmp(bgShot.type,'Fit')
    %exp2 bg seems to be the best but should be checked in other cases
    %[expbg,gof1]=fit([100:200 800:1024]',LineoutTSE_smooth([100:200 800:1024])','exp2');
    %[powerbg,gof2]=fit([100:200 800:1024]',LineoutTSE_smooth([100:200 800:1024])','power2');
    %[ratbg,gof3]=fit([100:200 800:1024]',LineoutTSE_smooth([100:200 800:1024])','rat21');
    [rat1bg,gof4]=fit([100:200 800:1024]',LineoutTSE_smooth([100:200 800:1024])','rat11');
    LineoutTSE_smooth=LineoutTSE_smooth-rat1bg(1:1024)';
end

noiseE= mean( elecData(:,BackgroundPixel-dpixel:BackgroundPixel+dpixel) ,2 );
noiseE = smooth(noiseE,span);
fitobj=fit([1:CCDsize(1)]',noiseE,'exp2','Exclude',[1:200,480:540,900:1024]);
noiseE = bgscalingE*fitobj(1:CCDsize(1));
noiseI= mean( ionData(:,BackgroundPixel-dpixel:BackgroundPixel+dpixel) ,2 );
noiseI = smooth(noiseI,span);
noiseI=mean(noiseI([200:400 700:850]));
noiseI(1:size(ionData,1))=bgscalingI*noiseI;

%temporary constant addition to the background
noiseE=noiseE+flatbg;
%% Plot Data

figure; imagesc(axisxE-shift_zero,axisyE,filter2(ones(5,3)/15,elecData));
set(gca,'FontWeight','bold', 'FontSize',12); set(gcf,'color','w');
axis xy;
title(['Shot : ',num2str(shotNum),' : ','TS : Thruput corrected'],'Fontsize',10,'Fontweight','Bold');
xlabel(xlab); ylabel('Wavelength (nm)')
colormap(gca,TS_mycmap)
line([axisxE(LineoutPixelE)'-shift_zero axisxE(LineoutPixelE)'-shift_zero],[axisy(1) axisy(end)],'color','r')
% line([axisx(BackgroundPixel)'-shift_zero axisx(BackgroundPixel)'-shift_zero],[axisy(1) axisy(end)],'color','k')
% 
figure; imagesc(axisxI-shift_zero,axisyI,filter2(ones(5,3)/15,ionData));
set(gca,'FontWeight','bold', 'FontSize',12); set(gcf,'color','w');
axis xy;
title(['Shot : ',num2str(shotNum),' : ','TS : Thruput corrected'],'Fontsize',10,'Fontweight','Bold');
xlabel(xlab); ylabel('Wavelength (nm)')
colormap(gca,TS_mycmap)
line([axisxI(LineoutPixelI-IAWtime)'-shift_zero axisxI(LineoutPixelI-IAWtime)'-shift_zero],[axisy(1) axisy(end)],'color','r')

%% Normalize Data before fitting

noiseE=noiseE/gain;
LineoutTSE_norm = LineoutTSE_smooth/gain;
LineoutTSE_norm = LineoutTSE_norm-noiseE';%new 6-29-20
ampE=max(LineoutTSE_norm(100:end)); %attempts to ignore 3w comtamination

noiseI=noiseI/gain;
LineoutTSI_norm = LineoutTSI_smooth/gain;
LineoutTSI_norm = LineoutTSI_norm-noiseI;%new 6-29-20
ampI=max(LineoutTSI_norm);
%PhysParams={[stddevE stddevI], [noiseE noiseI'], [ampE-noiseE, ampI-noiseI'], 0};%{width of IRF,background , amplitude ,Normalization of peaks}
PhysParams={[stddevE stddevI], [0 0], [ampE, ampI], 0};%{width of IRF,background , amplitude ,Normalization of peaks} new 6-29-20

%% Fitting
    
% D.Te=Te;
% D.Ti=Ti;
% D.ne=ne;
% if ~islogical(normalizeData)
%     D.amp1=normalizeData(1);
%     D.amp2=normalizeData(2);
%     D.amp3=normalizeData(3);
% end
% D.m=curDist{2};
% if exist('L','var')
%     D.LTe=L;
% end
% D.lam=lam;
% D.distName=curDist{1};
% D.Dsply='off';
% D.PlotFunc=[];
D.lamrangE=[axisyE(1) axisyE(end-1)];
% D.lamrangI=[axisyI(1) axisyI(end-1)];
D.iawoff=0;
D.iawfilter={1,4,24,528};
% D.Detector='OMEGA';
% D.npts=[];
% D.Z=Z;
% D.A=A;
% D.fract=fract;
% D.Va=Va;
% D.ud=ud;
% 
% D.curDist=curDist;
% D.distTable={D.distName};
% 
D.PhysParams=PhysParams;
D.extraoptions=extraoptions;
D.npts=(length(LineoutTSE_norm)-1)*20;

data=[(LineoutTSE_norm(1:end-1)); (LineoutTSI_norm(1:end-1))];
xie=linspace(-7,7,TSinputs.fe.Length);

if isempty(TSinputs.fe.Value)
    TSinputs=TSinputs.initFe(xie);
end
[TSinputs,x0,lb,ub]=genX(TSinputs);

if ~isempty(x0)
    %res = ThomsonFitter(data,x0,D);
    [x,~,~,~,~,grad,hess]=fmincon(@(x)chiSq2(x,TSinputs,xie,sa,D,data),...
        x0,[],[],[],[],lb,ub,[],options);
    
%     chisq=@(x)chiSq2(x,TSinputs,xie,sa,D,data);
%     hess1=calchess(x,1e-2,chisq);
%     hess2=calchess(x,1e-3,chisq);
%     hess3=calchess(x,1e-4,chisq);
else
    x=x0;
end
plotResult(x,TSinputs,xie,sa,D,data)

x=setTo(TSinputs,x);
result=x;
result.print
    %disp(res.x)

%% temporary plot of blue IAW vs time
%figure
%plot(axisxI,max(ionData(300:700,:)))
if strcmp(extraoptions.spectype,'Imaging')
    disp(['Lineout at ' num2str(LineoutPixelE) 'px ' num2str(axisxE(LineoutPixelE)) 'um'])
else
    disp(['Lineout at ' num2str(LineoutPixelE) 'px ' num2str(axisxE(LineoutPixelE)-shift_zero) 'ps'])
end
toc
end

function [eDat,iDat,xlab,zero]=loadData(sNum,sDay,specType,magE)
    if strcmp(specType,'Streaked')
        if sDay
            folder = '\\redwood\archive\tmp\thomson\';
            hdfnameE=strcat(strcat(folder,'epw\s',num2str(sNum),'.hdf'));
            hdfnameI=strcat(strcat(folder,'iaw\s',num2str(sNum),'.hdf'));
        else
            folder = 'Data\';
            hdfnameE=strcat(strcat('EPW-s',num2str(sNum),'.hdf'));
            hdfnameI=strcat(strcat(folder,'IAW-s',num2str(sNum),'.hdf'));
        end

        eDat=double(hdfread(hdfnameE,'Streak_array'));
        eDat = squeeze(eDat(1,:,:) - eDat(2,:,:));
        eDat = OMEGAWarpCorrection(eDat,'EPW',5,1);%correction file need to be updated
        eDat = eDat(16:1039,16:1039);

        iDat=double(hdfread(hdfnameI,'Streak_array'));
        iDat = squeeze(iDat(1,:,:) - iDat(2,:,:));

        xlab='Time (ps)';

        % Attempt to find t=0 from the fiducials
        %for Aug-26-2021 data this is roughly 130 pixels before the first upper
        %fiducial
        %shift_zero = 1590; %currently no T-0
        fidu=sum(eDat(50:100,:));
        [~,fiduLocs]=findpeaks(fidu,'MinPeakHeight',0.5*max(fidu));
        zero= magE*(fiduLocs(1)-130);
        %find the zero time location in the iaw then offset is
        %zeroi=zeroe+offset
        fidui=sum(iDat(100:200,:));
        [~,fiduiLocs]=findpeaks(fidui,'MinPeakHeight',0.5*max(fidui));
        zeroi= magE*(fiduiLocs(1)-154);
        ioff=zeroi-zero;
    else

        if sDay
            folder = '\\redwood\archive\tmp\thomson\';
            hdfnameE=strcat(strcat(folder,'epw_ccd\s',num2str(sNum),'.hdf')); %for use with archive/tmp
            hdfnameI=strcat(strcat(folder,'iaw_ccd\s',num2str(sNum),'.hdf'));
        else
            folder = 'Data\';
            hdfnameE=strcat(strcat('EPW_CCD-s',num2str(sNum),'.hdf'));
            hdfnameI=strcat(strcat('IAW_CCD-s',num2str(sNum),'.hdf'));
        end

        eDat=double(hdfread(hdfnameE,'Streak_array'));
        eDat = squeeze(eDat(1,:,:) - eDat(2,:,:));
        eDat = rot90(eDat,3);

        iDat=double(hdfread(hdfnameI,'Streak_array'));
        iDat = squeeze(iDat(1,:,:) - iDat(2,:,:));
        iDat = rot90(iDat,3);

        xlab='Radius (\mum)';
        zero=0;
    end
end
function elecData=correctThroughput(elecData,spectype,axisyE)
    if strcmp(spectype,'Streaked')
        sens = xlsread('Copy of MeasuredSensitivity_9.21.15.xls');
    else
        load('MeasuredSensitivity_11_30_21.mat');
    end
    sens(:,2)=1./sens(:,2);
    sens(1:18,2)=sens(19,2); %the sensitivity goes to zero in this location and is not usable

    vq1 = interp1(sens(:,1),sens(:,2),axisyE); % interpolate my wavelengths onto the throughput line

    %Note that C has NaN in it. This is were I'm getting them. It is okay but it has coused some problems when I'm coding things
    C = repmat(vq1, size(elecData, 2), 1)'; % expand my wavelength corrections vector into a matrix
    C(isnan(C))=0;

    elecData = elecData.*C; % Correct each wavelength/Row of the matrix
end
function chisq=chiSq2(x,TSinputs,xie,sas,D,data)
      
    modlE=fitModel2(TSinputs,xie,sas,x,D);
    [~,~,lam,~]=genTS(TSinputs,x,xie);
    [amp1,amp2,~]=genRest(TSinputs,x);
    [~,~,lamAxisE,~]=lamParse(D.lamrangE,lam,D.npts);
    %[omgL,omgsI,lamAxisI,~]=lamParse(D.lamrangI,lam,D.npts);
    
    
    %modlI=fitModel(Te,Ti,Z,D.A,D.fract,ne,Va,ud,omgsI,omgL,D.sa,curDist,...
    %    D.distTable,0,{0},D.lamrangI,lam,lamAxisI);
    
    originE=(max(lamAxisE)+min(lamAxisE))/2; %Conceptual_origin so the convolution donsn't shift the signal
    %originI=(max(lamAxisI)+min(lamAxisI))/2; %Conceptual_origin so the convolution donsn't shift the signal
        
    stddev = D.PhysParams{1};

    inst_funcE = (1/(stddev(1)*sqrt(2*pi)))*exp(-(lamAxisE-originE).^2/(2*(stddev(1))^2));%Gaussian
    %inst_funcI = (1/(stddev(2)*sqrt(2*pi)))*exp(-(lamAxisI-originI).^2/(2*(stddev(2))^2));%Gaussian

    ThryE = conv(modlE, inst_funcE,'same');
    ThryE=(max(modlE)./max(ThryE)).*ThryE;
    %ThryI = conv(modlI, inst_funcI,'same');
    %ThryI=(max(modlI)./max(ThryI)).*ThryI; 

    if D.PhysParams{4}
        ThryE(lamAxisE<lam) = amp1*(ThryE(lamAxisE<lam)/max(ThryE(lamAxisE<lam)));
        ThryE(lamAxisE>lam) = amp2*(ThryE(lamAxisE>lam)/max(ThryE(lamAxisE>lam)));
    end

    n=floor(length(ThryE)/length(data));
    ThryE=arrayfun(@(i) mean(ThryE(i:i+n-1)),1:n:length(ThryE)-n+1);
    %n=floor(length(ThryI)/length(data));
    %ThryI=arrayfun(@(i) mean(ThryI(i:i+n-1)),1:n:length(ThryI)-n+1);

    if ~D.PhysParams{4}
        lamAxisE=arrayfun(@(i) mean(lamAxisE(i:i+n-1)),1:n:length(lamAxisE)-n+1);
        ThryE = D.PhysParams{3}(1)*ThryE/max(ThryE);
        %lamAxisI=arrayfun(@(i) mean(lamAxisI(i:i+n-1)),1:n:length(lamAxisI)-n+1);
        %ThryI = amp3*D.PhysParams{3}(2)*ThryI/max(ThryI);
        ThryE(lamAxisE<lam) = amp1*(ThryE(lamAxisE<lam));
        ThryE(lamAxisE>lam) = amp2*(ThryE(lamAxisE>lam));
    end

    chisq=nan;
    redchi=nan;
    
    if isfield(D.extraoptions,'fitspecs')
        chisq=0;
%             if D.extraoptions.fitspecs(1)
%                 chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
%             end
        if D.extraoptions.fitspecs(2)
            %chisq=chisq+sum((data(1,lamAxisE<lam)-ThryE(lamAxisE<lam)).^2);
            chisq=chisq+sum((data(1,(lamAxisE>410) & (lamAxisE<510))-ThryE((lamAxisE>410) & (lamAxisE<510))).^2);
        end
        if D.extraoptions.fitspecs(3)
            %chisq=chisq+sum((data(1,lamAxisE>lam)-ThryE(lamAxisE>lam)).^2);
            chisq=chisq+sum((data(1,(lamAxisE>540) & (lamAxisE<680))-ThryE((lamAxisE>540) & (lamAxisE<680))).^2);
        end
    end
end
function plotResult(x,TSinputs,xie,sas,D,data)
    
    modlE=fitModel2(TSinputs,xie,sas,x,D);

    [~,~,lam,~]=genTS(TSinputs,x,xie);
    [amp1,amp2,~]=genRest(TSinputs,x);
    [~,~,lamAxisE,~]=lamParse(D.lamrangE,lam,D.npts);
    %[omgL,omgsI,lamAxisI,~]=lamParse(D.lamrangI,lam,D.npts);
    
    
    %modlI=fitModel(Te,Ti,Z,D.A,D.fract,ne,Va,ud,omgsI,omgL,D.sa,curDist,...
    %    D.distTable,0,{0},D.lamrangI,lam,lamAxisI);
    
    originE=(max(lamAxisE)+min(lamAxisE))/2; %Conceptual_origin so the convolution donsn't shift the signal
    %originI=(max(lamAxisI)+min(lamAxisI))/2; %Conceptual_origin so the convolution donsn't shift the signal
        
    stddev = D.PhysParams{1};

    inst_funcE = (1/(stddev(1)*sqrt(2*pi)))*exp(-(lamAxisE-originE).^2/(2*(stddev(1))^2));%Gaussian
    %inst_funcI = (1/(stddev(2)*sqrt(2*pi)))*exp(-(lamAxisI-originI).^2/(2*(stddev(2))^2));%Gaussian

    ThryE = conv(modlE, inst_funcE,'same');
    ThryE=(max(modlE)./max(ThryE)).*ThryE;
    %ThryI = conv(modlI, inst_funcI,'same');
    %ThryI=(max(modlI)./max(ThryI)).*ThryI; 

    if D.PhysParams{4}
        ThryE(lamAxisE<lam) = amp1*(ThryE(lamAxisE<lam)/max(ThryE(lamAxisE<lam)));
        ThryE(lamAxisE>lam) = amp2*(ThryE(lamAxisE>lam)/max(ThryE(lamAxisE>lam)));
    end

    n=floor(length(ThryE)/length(data));
    ThryE=arrayfun(@(i) mean(ThryE(i:i+n-1)),1:n:length(ThryE)-n+1);
    %n=floor(length(ThryI)/length(data));
    %ThryI=arrayfun(@(i) mean(ThryI(i:i+n-1)),1:n:length(ThryI)-n+1);

    if ~D.PhysParams{4}
        lamAxisE=arrayfun(@(i) mean(lamAxisE(i:i+n-1)),1:n:length(lamAxisE)-n+1);
        ThryE = D.PhysParams{3}(1)*ThryE/max(ThryE);
        %lamAxisI=arrayfun(@(i) mean(lamAxisI(i:i+n-1)),1:n:length(lamAxisI)-n+1);
        %ThryI = amp3*D.PhysParams{3}(2)*ThryI/max(ThryI);
        ThryE(lamAxisE<lam) = amp1*(ThryE(lamAxisE<lam));
        ThryE(lamAxisE>lam) = amp2*(ThryE(lamAxisE>lam));
    end
    
    LinePlots(lamAxisE,[data(1,:); ThryE],...
       'CurveNames',{'Data','Fit'},'XLabel','Wavelength (nm)');
    xlim([450 630])

%     LinePlots(lamAxisI,[data(2,:); ThryI],...
%        'CurveNames',{'Data','Fit'},'XLabel','Wavelength (nm)');
%     xlim([525 528])

    chisq=nan;
    redchi=nan;

    if isfield(D.extraoptions,'fitspecs')
        chisq=0;
%             if D.extraoptions.fitspecs(1)
%                 chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
%             end
        if D.extraoptions.fitspecs(2)
            %chisq=chisq+sum((data(1,lamAxisE<lam)-ThryE(lamAxisE<lam)).^2);
            chisq=chisq+sum((data(1,(lamAxisE>410) & (lamAxisE<510))-ThryE((lamAxisE>410) & (lamAxisE<510))).^2);
        end
        if D.extraoptions.fitspecs(3)
            %chisq=chisq+sum((data(1,lamAxisE>lam)-ThryE(lamAxisE>lam)).^2);
            chisq=chisq+sum((data(1,(lamAxisE>540) & (lamAxisE<680))-ThryE((lamAxisE>540) & (lamAxisE<680))).^2);
        end
    end

end
function modlE=fitModel2(TSins,xie,sa,x,D)

    [Te,ne,lam,fecur]=genTS(TSins,x,xie);
    [Te,ne]=genGradients(TSins,Te,ne,7);
    fecur=exp(fecur);
    
    %Te=[0.455928805318510,0.450441646945957,0.444968366928451,0.439510628172063,0.434070074402658,0.428648328233133,0.423246989285274];
    %ne=[0.207697651377665,0.205531183048812,0.203322141909742,0.201071148574029,0.198778918400705,0.196446262890277,0.194074090690791];
    if strcmp(TSins.fe.Type,'MYDLM')
        Thry=ApproxThomson4(Te,Te,1,1,1,ne*1E20,0,0,D.lamrangE,lam,sa.sa,...
            fecur,xie,TSins.fe.thetaphi);
    elseif strcmp(TSins.fe.Type,'Numeric')
        Thry=ApproxThomson4(Te,Te,1,1,1,ne*1E20,0,0,D.lamrangE,lam,sa.sa,...
            fecur,xie,[2*pi/3,0]);
    else
        Thry=ApproxThomson4(Te,Te,1,1,1,ne*1E20,0,0,D.lamrangE,lam,sa.sa,...
            fecur,xie);
    end
    
    Thry=real(Thry);
    %test=Thry(1:7,:,:).*[0.0635453189210914,0.171180337031858,0.199917471132199,0.199791935607436,0.199730361148521,0.140620073894124,0.0252145022647707]';
    %test=sum(test,1);
    %Thry=test;
    %modlE=sum(bsxfun(@times,squeeze(Thry)',sa.weights));
    Thry=mean(Thry,1);
    Thry=squeeze(Thry);
    Thry=permute(Thry,[2 1]);
    modlE=sum(Thry.*sa.weights);

    %[modl,lamAx]=S2Signal(Thry,lamAxis,D);
    [~,~,lamAxisE,~]=lamParse(D.lamrangE,lam,D.npts);
    if D.iawoff && (D.lamrangE(1)<lam && D.lamrangE(2)>lam)
        %set the ion feature to 0 %should be switched to a range about lam
        lamloc=find(abs(lamAxisE-lam)<(lamAxisE(2)-lamAxisE(1)));
        modlE(lamloc(1)-2000:lamloc(1)+2000)=0;
    end
    if D.iawfilter{1}
        filterb=(D.iawfilter{4}-D.iawfilter{3}/2);
        filterr=(D.iawfilter{4}+D.iawfilter{3}/2);
        if D.lamrangE(1)<filterr && D.lamrangE(2)>filterb
            if D.lamrangE(1)<filterb
                lamleft=find(abs(lamAxisE-filterb)<(lamAxisE(2)-lamAxisE(1)));
            else
                lamleft=1;
            end
            if D.lamrangE(2)>filterr
                lamright=find(abs(lamAxisE-filterr)<(lamAxisE(2)-lamAxisE(1)));
            else
                lamright=[0 length(lamAxisE)];
            end
            modlE(lamleft(1):lamright(2))=modlE(lamleft(1):lamright(2)).*10^(-D.iawfilter{2});
        end
    end

end
function myhess=calchess(x,epsilon,chisq)
    %% Temporary code to calculate the hessian at solution
    calcmyhess=1;
    if calcmyhess
    %epsilon=1e-4;
    f0=feval(chisq,x);
    myhess=zeros(length(x));
    for i=1:length(x)
        x1=x;
        x1(i)=x(i)+epsilon;
        f1=feval(chisq,x1);
        parfor j=1:i
            x2=x;
            x2(j)=x(j)+epsilon;
            x3=x1;
            x3(j)=x1(j)+epsilon;
            myhess(i,j)=(feval(chisq,x3)-feval(chisq,x2)-f1+f0)/epsilon^2;
        end
    end
    end
end