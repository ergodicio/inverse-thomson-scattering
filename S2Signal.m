function [signal,lamAxis,varargout]=S2Signal(data,lamAxis,D,varargin)
%% Input Parser
p=inputParser;
%currently built detectors are MTW,custom,ideal
%plan is to eventualy build OMEGA and NIF
%should be given in nm for real detectors and number of points in the range
%[0,5v/vth]
addOptional(p,'vaxis',[])
addOptional(p,'alpha',[])
addOptional(p,'DifNe',[])
addOptional(p,'BinWidth',[])
addOptional(p,'NumBinInRng',[])
addOptional(p,'TotalNumBin',[])
parse(p,varargin{:});
inputs=p.Results;
%% Main code inputs
%m=spectrometer dispersion[nm/um]
%sr=sweep rate[s/um]
%px=pixel size [um^2/pixel]
%M=Tube magnification
%QE=quantum efficiency [photo e-/photon]
%Trans=transmition
%En=probe energy [J]
%pulslen= pulse length [ps]
%L=length of focal volume [um]
%fnum= fnumber of colection

%check if any overrides have been submitted
if isempty(inputs.BinWidth)
    binWidth=D.BinWidth;
else
    binWidth=inputs.BinWidth;
end

if isempty(inputs.NumBinInRng)
    numBinInRng=D.NumBinInRng;
else
    numBinInRng=inputs.NumBinInRng;
end

if isempty(inputs.TotalNumBin)
    totalNumBin=D.TotalNumBin;
else
    totalNumBin=inputs.TotalNumBin;
end

%% Switch Detectors

switch D.Detector
    case 'MTW'
        PhysParams={.01,3.62*10^-14,170,1.6,.1,.0088,.5,55,60,3.1};

        if ~binWidth
            n=200;
        else
            n=floor(binWidth/0.00246);
        end
    case 'NIF'
        %most of these are estimates
        PhysParams={.004437,7*10^-13,170,1.35,.1,1,10,1000,50,8.3};
        
        if ~binWidth
            n=200;
        else
            n=floor(binWidth/0.00246);
        end
    case 'custom'
        PhysParams=D.PhysParams;

        if binWidth
            %since currently the code resolution is 0.00246nm
            n=floor(binWidth/0.00246);
        end
        
    case 'ideal'
        %Detector Params is
        %[SNR,count normalization]
        [sNR,cNorm]=D.PhysParams{:};
        if cNorm
            signal=data/max(data(:))*cNorm;
        else
            signal=data;
        end
        if sNR~=Inf
            signal=normrnd(signal,max(signal(:))/sNR);
            signal(signal<0)=0;
        end
        if numBinInRng
            %to keep resolution "constant the same number aof points (200)
            %are always set into the range 0-sqrt(a^2+3)
            %[~,minloc]=min(abs(inputs.vaxis-sqrt(inputs.alpha^2+3)));
            [~,minloc]=min(abs(inputs.vaxis-8));
            n=floor((length(data)-minloc)/numBinInRng);
            numbins=totalNumBin;
        elseif binWidth
            n=binWidth;
            numbins=totalNumBin;
        end
end

if ~strcmp(D.Detector,'ideal')
    %% Constants
    r0=2.82*10^-13;
    hc = 1.9864458*10^-25;%in joule meters

    %% Calculations
    [m,sr,px,M,QE,Trans,En,pulslen,L,fnum]=PhysParams{:};
    Pin=En/(pulslen*10^-12);
    L=L*10^-4; %convert L to cm

    domg=2*pi*(1-cos(atan(1/(2*fnum))));
    if isempty(inputs.DifNe)
        ne=D.ne;
    else
        ne=inputs.DifNe;
    end
    
    PsoPi=L*r0^2*ne*domg*data;
    signal=(m*sr*px/M)*((lamAxis*10^-9)/hc)*QE*Trans*Pin.*PsoPi;
end

%% Rebin the data to the approximate resolution of the streak camera
if exist('n','var')
    %summing every 200 give spectral reasolution of ~.5nm for the current
    %npts, if npts is changed or the desired resolution changes then it
    %will need to be updated
    presum=signal;
%     for ii=1:size(signal,1)
%         postsum(ii,:)=arrayfun(@(i) mean(signal(ii,i:i+n-1)),1:n:length(signal)-n+1);
%     end
    postsum=arrayfun(@(i) mean(signal(:,i:i+n-1),2),1:n:length(signal)-n+1,'UniformOutput',0);
    postsum=cell2mat(postsum);
    signal=postsum;

    %signal=arrayfun(@(i,row) mean(signal(row,i:i+n-1)),1:n:length(signal)-n+1,'UniformOutput',0);
    lamAxis=arrayfun(@(i) mean(lamAxis(i:i+n-1)),1:n:length(lamAxis)-n+1);
    if~isempty(inputs.vaxis)
        inputs.vaxis=arrayfun(@(i) mean(inputs.vaxis(i:i+n-1)),1:n:length(inputs.vaxis)-n+1);
    end
    if exist('numbins','var') && numbins<length(lamAxis)
        signal=signal(:,end-numbins:end);
        lamAxis=lamAxis(end-numbins:end);
        if~isempty(inputs.vaxis)
            inputs.vaxis=inputs.vaxis(end-numbins:end);
        end
    end
    if cNorm
        signal=signal/max(signal(:))*cNorm;
    end
    varargout{2}=n;
else
    varargout{2}=0;
end

varargout{1}=inputs.vaxis;
end