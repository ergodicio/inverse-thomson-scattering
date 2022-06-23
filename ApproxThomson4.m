function [formfactor, formfactorE] = ApproxThomson4(Te,Ti,Z,A,fract,ne,Va,ud,lamrang,lam,sa,varargin)
    % NONMAXWTHOMSON calculates the Thomson spectrum using ratint and is
    % capable of taking a vector for sa,Te,ne. All are expected as row
    % vectors and are reshaped as needed. Calculates ion contribtuion and
    % returns 2 versions of the formfactor, one with and one without the
    % ion component. The high alpha extension from approx3 has been removed
    % (a better version needs to be developed).
    
    %interpAlg='spline';
    interpAlg='linear';

    % basic quantities
    C=2.99792458e10;
    Me=510.9896/C^2;               % electron mass KeV/C^2 
    Mp=Me*1836.1;                  % proton Mass KeV/C^2
    Mi=A*Mp;                       % ion mass
    re=2.8179e-13;                 % classical electron radius cm
    e=1.6e-19;                     % electron charge        
    Esq = Me*C^2*re;               % sq of the electron charge KeV-cm
    constants = sqrt(4*pi*Esq/Me); % sqrt(4*pi*e^2/Me)
    sarad=sa*2*pi/360;             % scattering angle in radians
    sarad=reshape(sarad,1,1,[]);% scattering angle in radians
    Va=Va*1e6;                     % flow velocity in 1e6 cm/s
    ud=ud*1e6;                     % drift velocity in 1e6 cm/s
    npts=20460;%10250;
    %%
    [omgL,omgs,~,~]=lamParse(lamrang,lam,npts,0);%the final boolean true add fine ion gridding
    % calculating k and omega vectors
    omgpe=constants*sqrt(ne)';      % plasma frequency Rad/s
    omg = omgs - omgL;
    ks=sqrt (omgs.^2-omgpe.^2)/C; 
    kL=sqrt (omgL^2-omgpe.^2)/C;     % laser wavenumber in Rad/cm
    k=sqrt(ks.^2+kL.^2-2*ks.*kL.*cos(sarad));
    %k=sqrt(bsxfun(@minus,ks.^2+kL^2,bsxfun(@times,2*ks*kL,cos(sarad'))));
    kdotv = k*Va;
    omgdop=omg - kdotv;
    %omgdop=bsxfun(@minus,omg,kdotv);
    
    % plasma parameters

    % electrons
    vTe=sqrt(Te/Me)';              % electron thermal velocity;                     
    klde=(vTe./omgpe).*k ;             
    
    % ions
    Z=reshape(Z,1,1,1,[]);
    A=reshape(A,1,1,1,[]);
    Mi=reshape(Mi,1,1,1,[]);
    fract=reshape(fract,1,1,1,[]);
    Zbar = sum(Z.*fract);
    ni = fract.*ne./Zbar;
    omgpi = constants.*Z.*sqrt(ni.*Me./Mi);
    vTi=sqrt(Ti./Mi);               % ion thermal velocity
    kldi=permute(vTi./omgpi,[2 1 3 4]).*k;
    
    % ion susceptibilities
    % finding derivative of plasma dispersion function along xii array
    %proper handeling of multiple ion temperatures is not implemented
    xii=1./permute((sqrt(2.)*vTi),[2 1 3 4]).*(omgdop./k);
    num_species=length(fract);
    num_ion_pts = size(xii);
    chiI=zeros(num_ion_pts);


    h=0.01;
    minmax=8.2;
    h1=1000;
    xi1=linspace(-minmax-sqrt(2.)/h1,minmax+sqrt(2.)/h1,h1);
    xi2=(-minmax:h:minmax);
    
    Zpi = zprimeSteven(xi2);
    ZpiR=interp1(xi2,Zpi(1,:),xii,'spline',0);
    ZpiI=interp1(xi2,Zpi(2,:),xii,'spline',0);
    chiI =sum(-0.5./(kldi.^2).* (ZpiR+sqrt(-1)*ZpiI),4);

    
    % electron susceptibility
    % calculating normilized phase velcoity (xi's) for electrons
    xie=omgdop./(k.*vTe)-ud./vTe ;
    

    %% Old system for interpolating the distribution function
    %works for an isotropic distribution function
    %the catch and display statements are due to common error that occured
    %during ARTS fitting, i belive they have been mainly worked out
    %(8/2/21)
%     if sum(abs(imag(DF)))
%         disp('complex')
%     end
%     if sum(abs(imag(x)))
%         disp('complex')
%     end
%     if sum(abs(imag(xie)))
%         disp('complex')
%     end
%     
%     try
%         %fe_vphi=interp1(x,DF,xie,interpAlg,0);
%         fe_vphi=exp(interp1(x,log(DF),xie,interpAlg,-Inf));
%         fe_vphi(isnan(fe_vphi))=0;
%     catch
%         disp('??')
%         x
%         DF
%     end

    %% New system for interpolating distribution function
    %capable of handleing isotropic or anisotropic distribtuion functions
    
    %varargin is separated into components, distribtuion function, v/vth
    %axis, angles between f1 and kL
    switch length(varargin)
        case 2
            [DF,x]=varargin{:};
            fe_vphi=exp(interp1(x,log(DF),xie,interpAlg,-Inf));
            fe_vphi(isnan(fe_vphi))=0;
        case 3
            [DF,x,thetaphi]=varargin{:};
            %the angle each k makes with the anisotropy is calculated
            thetak=pi-asin((ks./k).*sin(sarad));
            thetak(:,omg<0,:)=-asin((ks(omg<0)./k(:,omg<0,:)).*sin(sarad));
            %asin can only return values from -pi/2 to pi/2
            %this attempts to find the real value
            theta90=asin((ks./k).*sin(sarad));
            ambcase=logical((ks>kL/cos(sarad)).* (sarad<pi/2));
            thetak(ambcase)=theta90(ambcase);
            

            beta=acos(sin(thetaphi(2))*sin(thetaphi(1))*sin(thetak)+ ...
                cos(thetaphi(1))*cos(thetak));
            
            %[vmesh,thetamesh]=meshgrid(x,0:10^-1.2018:2*pi);
            %[vmesh,thetamesh]=meshgrid(x,linspace(0,2*pi,1000));
            %[vmeshq,thetameshq]=meshgrid(xie,beta);
            %fe_vphi=exp(interp2(vmesh,thetamesh,log(DF),vmeshq,thetameshq,interpAlg,-Inf));
            %here the abs(xie) handles the double counting of the direction
            %of k changing and delta omega being negative
            fe_vphi=exp(interpn(0:10^-1.2018:2*pi,x,log(DF),beta,abs(xie),interpAlg,-Inf));
            
            fe_vphi(isnan(fe_vphi))=0;
    
    end
    
    df=diff(fe_vphi,1,2)./diff(xie,1,2);
    df(:,end+1,:)=zeros(length(ne),1,length(sa));
    
    chiEI = pi./(klde.^2).*sqrt(-1).*df;
    
    ratdf=gradient(exp(interp1(x,log(DF'),xi1,interpAlg,-Inf))',xi1(2)-xi1(1));
    ratdf(isnan(ratdf))=0;
    if size(ratdf,2)==1
        ratdf=ratdf';
    end
    
    chiERratprim=zeros(size(ratdf,1),length(xi2));
    for iw=1:length(xi2)
        chiERratprim(:,iw)=real(ratintn(ratdf,xi1-xi2(iw),xi1));
    end
    
    if length(varargin)==2
        chiERrat=interp1(xi2,chiERratprim,xie,'spline');
    else
        chiERrat=interpn(0:10^-1.2018:2*pi,xi2,chiERratprim,beta,xie,'spline');
    end
    chiERrat=- 1./(klde.^2).*chiERrat;
    
    %chiI=zeros(size(chiI));
    chiE=chiERrat+chiEI;
    epsilon=1+(chiE)+(chiI);
    
    %This line needs to be changed if ion distribution is changed!!!
    %ion_comp=Z.*sqrt(Te/Ti.*A*1836)*(abs(chiE)).^2.*exp(-(xii.^2))/sqrt(2*pi);
    ion_comp=permute(fract.*Z.^2/Zbar./vTi,[2 1 3 4]).*(abs(chiE)).^2.*exp(-(xii.^2))/sqrt(2*pi); 
    ele_comp=(abs(1+chiI)).^2.*double(fe_vphi)./vTe;
    ele_compE=double(fe_vphi)./vTe;

    SKW_ion_omg = 2*pi*1./klde.*(ion_comp)./((abs(epsilon)).^2) *1./omgpe;
    SKW_ion_omg = sum(SKW_ion_omg,4);
    SKW_ele_omg = 2*pi*1./klde.*(ele_comp)./((abs(epsilon)).^2) .*vTe./omgpe;
    SKW_ele_omgE = 2*pi*1./klde.*(ele_compE)./((abs(1+(chiE))).^2) .*vTe./omgpe;

    PsOmg = (SKW_ion_omg + SKW_ele_omg).*(1+2*omgdop/omgL)*re^2.*ne';
    PsOmgE = (SKW_ele_omgE).*(1+2*omgdop/omgL)*re^2.*ne';
    lams=2*pi*C./omgs;
    PsLam = PsOmg *2*pi*C./lams.^2 ;
    PsLamE = PsOmgE *2*pi*C./lams.^2 ;

    formfactor = PsLam;
    formfactorE = PsLamE;
    
end

