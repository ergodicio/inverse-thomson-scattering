function [omgL,omgs,lamAxis,npts]=lamParse(lamrang,lam,varargin)
%This function replaces the section at the begining of all the codes which
%parses lamrang and calculates the lamAxis and the corresponding omegas
%Update 7-19-21: a new case has been added specificaly designed to work
%with the ApproxThomson4 giving a higher resolution in the ion region
c=2.99792458e10;
if ischar(lamrang)
    switch lamrang
        case 'EPWb'
    %         rng=50;
    %         off=76.25; %looks at blue shifted epw feature
    %         rng=46.75;
    %         off=54.75; %looks at blue shifted epw feature
    %         npts=37910;
            rng=59.2;
            off=67.2; %looks at blue shifted epw feature
            npts=48000;
        case 'EPWr'
    %         rng=50;
    %         off=-123.75;
    %         rng=46.75; %These are the correct
    %         off=-54.75; %These are the correct
            off=-173.5;
            rng=150;
            npts=37910;
        case 'Full'
    %         rng=150;
    %         off=-23.75; %looks at entire spectrum 400-700
            %rng=126.4;%correct MTW range
            rng=450;
            off=0; %looks at the correct spectrum
            npts=102500;
        case 'EPWbNIF'
            rng=35;
            off=45.6; %looks at blue shifted epw feature
            npts=20273;
    end

    min_lam=lam-off-rng; % Minimum wavelength to run code over
    max_lam=lam-off+rng; % Max wavelength
else
    min_lam=lamrang(1);
    max_lam=lamrang(2);
    npts=102500;
end
fineion=0;
if ~isempty(varargin) && ~isempty(varargin{1})
    npts=varargin{1};
    if length(varargin)>1
        if varargin{2}
            fineion=1;
        end
    end
end
if fineion && (min_lam<lam && max_lam>lam)
    lamAxis=linspace(min_lam, max_lam, npts);
    L = find(lamAxis>=lam-2,1,'first');
    R = find(lamAxis<=lam+2,1,'last');
    V = linspace(lamAxis(L),lamAxis(R),npts);
    lamAxis = [lamAxis(1:L-1),V,lamAxis(R+1:end)];
    %spac=(max_lam-min_lam)/npts;
    %lamAxis = [min_lam:spac:lam-2 lam-2:spac/75:lam+2 lam+2:spac:max_lam];
    %lamAxis = unique(lamAxis);
    
    %lamAxis = [linspace(min_lam,lam-2,floor(npts./2)) ...
    %    linspace(lam-2,lam+2,npts) linspace(lam+2,max_lam,floor(npts./2))];
else
    lamAxis = linspace(min_lam, max_lam, npts);
end


omgs = 2e7*pi*c./lamAxis;      % Scattered frequency axis (1/sec)
omgL=2*pi*1e7*c/lam;           % laser frequency Rad/s

end
