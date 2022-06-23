function x=AngTSDataFitter(Fshotnum,Bshotnum,lineLocs,TSinputs)
% This function takes the inputs from the ANGTSDATAFITTERGUI and preforms
% the data corrections then fits the data retunring the fit result
%
%The inputs from the GUI are Shot number, linout locations, background shot
%number, probe wavelength, electron temperature, electron density, m, amp1
%and amp2, ionization state, starting distribution function
%type and the number of distribution function points to use in numerical
%distribution function fitting. 

%% Hardcoded inputs
% These are the detector info and fitting options
D.Detector='ideal';
D.BinWidth=10;
D.NumBinInRng=0;
D.TotalNumBin=1023;

spectralFWHM=.9; %nominaly this is ~.8 or .9 %4 for h2
angularFWHM=1; %unknown

norm2B=0;   %0 no normalization
            %1 norm to blue
            %2 norm to red
            
options = optimoptions(@fmincon,'Display','iter','PlotFcn',[],...
    'UseParallel',true,'MaxIter',300,'MaxFunEval',10000,'TolX',1e-10);
% options = optimoptions(@fmincon,'Display','iter','PlotFcn',[],...
%     'UseParallel',false,'MaxIter',1,'MaxFunEval',10000,'TolX',1e-10);

scaterangs=19:.5:139;

feDecreaseStrict=1; %forces the result to have a decresing distribution function (no bumps)

%% Read Data
FG = hdfread(['ATS-s' num2str(Fshotnum) '.hdf'], 'Streak_array');
BG = hdfread(['ATS-s' num2str(Bshotnum) '.hdf'], 'Streak_array');

load ('MyColormaps_TS','TS_mycmap');
FG=squeeze(FG(1,:,:)-FG(2,:,:));
BG=squeeze(BG(1,:,:)-BG(2,:,:));
FG=double(FG);
BG=conv2(double(BG),1/25*ones(5),'same'); %1/27 for H2 %1/24 for kr
%BG=conv2(double(BG),1/27*ones(5),'same'); %1/27 for H2 %1/24 for kr

%Fit a line of the background times a polynomial to a line of the data
xx=1:1024;
fun=@(x) sum((FG(1000,:) -((x(1)*(xx-x(4)).^2 + x(2)*(xx-x(4)) + x(3)).*BG(1000,:))).^2);
corrfactor=fmincon(fun,[.1 .1 1.15 300],[],[]);
newBG=(corrfactor(1)*(xx-corrfactor(4)).^2 + corrfactor(2)*(xx-corrfactor(4)) + corrfactor(3)).*BG;
data=FG-newBG;
% data=FG-BG;

load('angsFRED.mat','angsFRED')
xax=angsFRED;

if Fshotnum<95000
    yax=(0:1023)*.214116+449.5272; %Update?
else
    yax=(0:1023)*.2129+439.8;
end

%% Correct Thruput
%W=importdata('\\sequoia\Project_Files\03_Diagnostics OMEGA\LLE\AngResTS\Archive\setup_images\Sepc_Sensitivity\BectelW.txt');

load('spectral_sensitivity.mat','speccal');
if Fshotnum<95000
    data=data./speccal;
else
    specax=(0:1023)*.214116+449.5272;
    speccalshift=interp1(specax,speccal,yax,'linear',speccal(1));
    data=data./speccalshift;
end

if norm2B==1
    %normalize to blue peak
    data=data./max(data);
    D.PhysParams={Inf,max(max(double(FG-BG)))};
elseif norm2B==2
    %normalize to red peak
    data=data./max(data(470:900,:));
    D.PhysParams={Inf,max(max(double(FG-BG)))};
else
    D.PhysParams={Inf,0};
end



ttl=['Artemis data: Shot ' num2str(Fshotnum)];
ColorPlots(yax,xax,rot90(data),'Title',ttl,'Kaxis',[TSinputs.ne.Value*1E20,TSinputs.Te.Value,526.5],'Name','Data')
curlims=caxis;
caxis([0 curlims(2)])
line([yax(1) yax(end)],[xax(end-lineLocs)' xax(end-lineLocs)'])

%% f/5 angular smearing

load('angleWghtsFredfine.mat','weightMatrix')


%%

xie=linspace(0,7,TSinputs.fe.Length);
[~,~,lamAxis,~]=lamParse([min(yax) max(yax)],TSinputs.lam.Value,10250);

%% Setup x0
if isempty(TSinputs.fe.Value)
    TSinputs=TSinputs.initFe(xie);
end
[TSinputs,x0,lb,ub]=genX(TSinputs);

%% Temporary fe overwrite
% allows us to get amplitudes and plots with the max tail model
% fepars=[0.0304 2.0309 4.0295 -9.1440 2.0339];
% thry3=@(y,x) y(1)*exp(-(abs(x)/y(2)).^y(3)) + exp(y(4))*exp(-(abs(x)/y(5)).^2);
% function thry1d=to1d(y,xax,fun)
%     for ii=1:length(xax)
%         thry1d=integral(@(vr)2*pi*vr.*fun(y,sqrt(vr.^2+xax.^2)),0,Inf,'ArrayValued',1);
%     end
% end
% TSinputs.fe.Value= log(to1d(fepars,xie,thry3));

%% Plot starting point

Thryinit=ArtemisModel(TSinputs,xie,scaterangs,x0,weightMatrix,...
    spectralFWHM,angularFWHM,lamAxis,xax,D,norm2B);
if ~norm2B
    Thryinit=Thryinit./max(Thryinit(470:900,:));
    Thryinit=Thryinit.*max(data(470:900,:));
    Thryinit=TSinputs.amp1.Value*Thryinit;
end
chisq = sum(sum((data([40:330 470:900],90:1015)-Thryinit([40:330 470:900],90:1015)).^2));
Thryinit(330:470,:)=0;

ColorPlots(yax,xax,rot90(Thryinit),'Kaxis',[TSinputs.ne.Value*1E20,TSinputs.Te.Value,526.5],...
    'Title','Starting point','Name','Initial Spectrum');
ColorPlots(yax,xax,rot90(data-Thryinit),'Title',...
    ['Initial difference: \chi^2 =' num2str(chisq)],'Name','Initial Difference');
load('diffcmap.mat','diffcmap');
colormap(diffcmap);

if norm2B
    caxis([-1 1]);
else
    caxis([-8000 8000]);
end
%% Perform fit
tic
if ~isempty(x0)
    if feDecreaseStrict && TSinputs.fe.Active
        A=zeros(length(x0));
        inds=sub2ind(size(A),TSinputs.fe.Location-TSinputs.fe.Length+1:TSinputs.fe.Location-1,TSinputs.fe.Location-TSinputs.fe.Length+1:TSinputs.fe.Location-1);
        A(inds)=-1;
        A(inds+length(x0))=1;
        b=zeros(length(x0),1);
    else
        A=[];
        b=[];
    end
    n=10;
    dataResunit=arrayfun(@(i) sum(data(:,i:i+n-1),2),1:n:1024-n+1,'UniformOutput',0);
    dataResunit=cell2mat(dataResunit);
    n=5;
    dataResunit=arrayfun(@(i) sum(dataResunit(i:i+n-1,:),1),1:n:1024-n+1,'UniformOutput',0);
    dataResunit=cell2mat(dataResunit');
    
    %% Temporary code to calculate the hessian at x0
    calcmyhess=0;
    if calcmyhess
    chisq=@(x)chiSqArtemis(x,TSinputs,xie,...
        scaterangs,weightMatrix,spectralFWHM,angularFWHM,lamAxis,xax,D,...
        dataResunit,norm2B);
    tic
    epsilon=1e-3;
    f0=feval(chisq,x0);
    myhess=zeros(length(x0));
    for i=1:length(x0)-22
        x1=x0;
        x1(i)=x0(i)+epsilon;
        f1=feval(chisq,x1);
        parfor j=1:i
            x2=x0;
            x2(j)=x0(j)+epsilon;
            x3=x1;
            x3(j)=x1(j)+epsilon;
            myhess(i,j)=(feval(chisq,x3)-feval(chisq,x2)-f1+f0)/epsilon^2;
        end
        toc
    end
    toc
    end
    %% Resume Preform fit
    [x,~,~,~,~,grad,hess]=fmincon(@(x)chiSqArtemis(x,TSinputs,xie,...
        scaterangs,weightMatrix,spectralFWHM,angularFWHM,lamAxis,xax,D,...
        dataResunit,norm2B),x0,A,b,[],[],lb,ub,[],options);
else
    x=chisq
    for i=1:length(lineLocs)
        figure('Name',['Lineout ' num2str(i)])
        plot(yax,sum(data(:,lineLocs(i):lineLocs(i)+25),2))
        hold on
        plot(yax,sum(Thryinit(:,lineLocs(i):lineLocs(i)+25),2))
        title(['Lineout at pixel ' num2str(lineLocs(i)) ': ' ...
        num2str(xax(end-lineLocs(i))) '\circ'])
        
    end
    
    figure('Units','normalized','position',[.1 .1 .25 .6],'Name',['Lineouts'])
    hold on
    for i=1:length(lineLocs)
        plot(yax,50000*i+sum(data(:,lineLocs(i):lineLocs(i)+25),2))
        plot(yax,50000*i+sum(Thryinit(:,lineLocs(i):lineLocs(i)+25),2))
    end
    xlabel('Wavelength (nm)')
    xlim([450 650])
    set(gca,'YColor','none','fontsize',16,'fontweight','bold')
    title(['Lineouts ' num2str(xax(end-lineLocs)) '\circ'])
    
    return
end
toc

    function csq=chiSqArtemis(x,TSinputs,xie,sas,...
            wghts,spectralFWHM,angularFWHM,lamAxis,xax,D,data,norm2B)
        Thry=ArtemisModel(TSinputs,xie,sas,...
            x,wghts,spectralFWHM,angularFWHM,lamAxis,xax,D,norm2B);
        
%         if ~norm2B
%             [amp1,amp2]=genRest(TSinputs,x);
%             Thry=Thry./max(Thry(470:900,:));
%             Thry=Thry.*max(data(470:900,:));
%             Thry(1:450,:)=amp1*Thry(1:450,:);
%             Thry(450:end,:)=amp2*Thry(450:end,:);
%         end
%         
%         %csq = sum(sum((data([40:330 470:900],90:1015)-Thry([40:330 470:900],90:1015)).^2));
%         %csq = sum(sum((data([40:330 470:900],50:end-300)-Thry([40:330 470:900],50:end-300)).^2));%used for shots on July 23
%         %csq = sum(sum((data([470:900],90:1015)-Thry([470:900],90:1015)).^2));
%         csq = sum(sum((data([40:330 470:1000],300:end-120)-Thry([40:330 ...
%         470:1000],300:end-120)).^2)); %current best version
%         %csq = sum(sum((data([470:1000],300:end-120)-Thry([470:1000],300:end-120)).^2)); %red only
        n2=10;
        Thry=arrayfun(@(i) sum(Thry(:,i:i+n2-1),2),1:n2:1024-n2+1,'UniformOutput',0);
        Thry=cell2mat(Thry);
        n2=5;
        Thry=arrayfun(@(i) sum(Thry(i:i+n2-1,:),1),1:n2:1024-n2+1,'UniformOutput',0);
        Thry=cell2mat(Thry');
        if ~norm2B
            [amp1,amp2]=genRest(TSinputs,x);
            Thry=Thry./max(Thry(94:180,:));
            Thry=Thry.*max(data(94:180,:));
            Thry(1:90,:)=amp1*Thry(1:90,:);
            Thry(90:end,:)=amp2*Thry(90:end,:);
        end
        
        uncert=(0.01*(data([8:66 94:200],30:end-12)./500)).^2;
        uncert(uncert<200)=200;
        %csq = sum(sum((data([8:66 94:200],30:end-12)-Thry([8:66 94:200],30:end-12)).^2)); %current best version
        %csq = sum(sum(((data([8:66 94:200],30:end-12)./500-Thry([8:66 94:200],30:end-12)./500).^2)./abs(data([8:66 94:200],30:end-12)./500))); %attempt to get real errorbars 5-13-20
        csq = sum(sum(((data([8:66 94:200],30:end-12)./500-Thry([8:66 94:200],30:end-12)./500).^2)./uncert));
    end

%% Plot Result
Thryfin=ArtemisModel(TSinputs,xie,scaterangs,...
    x,weightMatrix,spectralFWHM,angularFWHM,lamAxis,xax,D,norm2B);
n2=10;
ThryfinRes=arrayfun(@(i) sum(Thryfin(:,i:i+n2-1),2),1:n2:1024-n2+1,'UniformOutput',0);
ThryfinRes=cell2mat(ThryfinRes);
xaxRes=arrayfun(@(i) mean(xax(i:i+n2-1)),1:n2:1024-n2+1,'UniformOutput',0);
xaxRes=cell2mat(xaxRes);
n2=5;
ThryfinRes=arrayfun(@(i) sum(ThryfinRes(i:i+n2-1,:),1),1:n2:1024-n2+1,'UniformOutput',0);
ThryfinRes=cell2mat(ThryfinRes');
yaxRes=arrayfun(@(i) mean(yax(i:i+n2-1)),1:n2:1024-n2+1,'UniformOutput',0);
yaxRes=cell2mat(yaxRes);

if ~norm2B
    [amp1,amp2]=genRest(TSinputs,x);
    ThryfinRes=ThryfinRes./max(ThryfinRes(94:180,:));
    ThryfinRes=ThryfinRes.*max(dataResunit(94:180,:));
    ThryfinRes(1:90,:)=amp1*ThryfinRes(1:90,:);
    ThryfinRes(90:end,:)=amp2*ThryfinRes(90:end,:);
end
ThryfinRes(66:94,:)=0;
chisqfin = sum(sum((dataResunit([8:66 94:200],30:end-12)-ThryfinRes([8:66 94:200],30:end-12)).^2));

if ~norm2B
    [amp1,amp2]=genRest(TSinputs,x);
    Thryfin=Thryfin./max(Thryfin(470:900,:));
    Thryfin=Thryfin.*max(data(470:900,:));
    Thryfin(1:450,:)=amp1*Thryfin(1:450,:);
    Thryfin(450:end,:)=amp2*Thryfin(450:end,:);
end
Thryfin(330:470,:)=0;
chisqfin = sum(sum((data([40:330 470:900],90:1015)-Thryfin([40:330 470:900],90:1015)).^2));

[Tefin,nefin,~,~]=genTS(TSinputs,x);
ColorPlots(yaxRes,xaxRes,rot90(ThryfinRes),'Kaxis',[nefin*1E20,Tefin,526.5],...
    'Title','Final Spectrum','Name','Final Spectrum');
ColorPlots(yax,xax,rot90(data-Thryfin),'Title',...
    ['Final difference: \chi^2 =' num2str(chisqfin)],'Name','Final Difference');
load('diffcmap.mat','diffcmap');
colormap(diffcmap);
if norm2B
    caxis([-1 1]);
else
    caxis([-8000 8000]);
end

for i=1:length(lineLocs)
    figure('Name',['Lineout ' num2str(i)])
    plot(yaxRes,sum(dataResunit(:,lineLocs(i)/10:lineLocs(i)/10+1),2))
    hold on
    plot(yaxRes,sum(ThryfinRes(:,lineLocs(i)/10:lineLocs(i)/10+1),2))
    title(['Lineout at pixel ' num2str(lineLocs(i)) ': ' ...
        num2str(xax(end-lineLocs(i))) '\circ'])
end

    figure('Units','normalized','position',[.1 .1 .25 .6],'Name',['Lineouts'])
    hold on
    %500,000 for 94481
    for i=1:length(lineLocs)
        plot(yaxRes,100000*i+sum(dataResunit(:,lineLocs(i)/10:lineLocs(i)/10+1),2))
        plot(yaxRes,100000*i+sum(ThryfinRes(:,lineLocs(i)/10:lineLocs(i)/10+1),2))
    end
    xlabel('Wavelength (nm)')
    xlim([450 650])
    set(gca,'YColor','none','fontsize',16,'fontweight','bold')
    title(['Lineouts ' num2str(xax(end-lineLocs)) '\circ'])
    
x=setTo(TSinputs,x);
x.fitprops.grad=grad;
x.fitprops.hess=hess;
end

function modl=ArtemisModel(TSins,xie,sas,x,wghts,spectralFWHM,angularFWHM,lamAxis,xax,D,norm2B)
    
    extend=0; %binary determining the use of the approximate extension for high velocity

    if TSins.m.Active && ~TSins.fe.Active
        [Te,ne,lam,fecur]=genTS(TSins,x,xie);
    else
        [Te,ne,lam,fecur]=genTS(TSins,x);
    end
    [Te,ne]=genGradients(TSins,Te,ne,7);
    fecur=exp(fecur);
        
    xiecur=[-flip(xie) xie(2:end)];
    fecur=[flip(fecur) fecur(2:end)];
    Thry=ApproxThomson3(Te,ne*1E20,lamAxis([1 end]),lam,sas,fecur,xiecur,extend);
    
    Thry=mean(Thry,1);
    Thry=squeeze(Thry);
    Thry=permute(Thry,[2 1]);
    
    [amp1,amp2,blur]=genRest(TSins,x);
%     if extend
%         %The small angles have area under the curve -> inf (when it gets to inf
%         %the value goes to nan, effectivly divide by 0)
%         %this is a cheap way to correct it but not correct
%         maxcount=10000;
%         nanlocs=isnan(Thry);
%         nanlocs(:,end)=zeros(size(Thry,1),1);
%         Thry(nanlocs)=maxcount;
%         Thryb=Thry(:,1:floor(end/2));
%         Thryr=Thry(:,ceil(end/2):end);
% 
%         Thryb=settonext(Thryb,1);
%         Thryr=settonext(Thryr,1);
%         Thry=[Thryb Thryr];
%     end
%     
%     function vec=settonext(vec,ind)
%         [maxvec,l]=max(vec(ind,:));
% 
%         nextvec=vec(ind,l+1);
%         if maxvec>30*nextvec
%             vec=settonext(vec,ind+1);
%             vec(ind,:)=(vec(ind,:)./vec(ind,l))*max(vec(ind+1,:));
%         else
%             vec(ind,:)=(vec(ind,:)./vec(ind,l))*max(vec(ind+1,:));
%         end
%     end
    modl=wghts*Thry;

    [modl,lamAx]=S2Signal(modl,lamAxis,D);
    modl=addIRF2D(spectralFWHM+blur,angularFWHM,lamAx,xax,modl);
    modl=rot90(modl,3);
    %modl(1:450,:)=amp1*modl(1:450,:);
    %modl(450:end,:)=amp2*modl(450:end,:);
    
    %normalize to blue peak
    if norm2B
        modl=modl./max(modl(470:900,:));
        modl=amp1*modl;
    end
    
    if ~isempty(TSins.specCurvature)&&TSins.specCurvature.Active
        invfunc= @(xy) [xy(:,1),xy(:,2)-(TSins.specCurvature.Value/(512^2))*(xy(:,1)-512).^2];
        modl=imwarp(modl,geometricTransform2d(invfunc),'OutputView',imref2d(size(modl)));
    end
    
end