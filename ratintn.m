function out=ratintn(f,g,z)

%    Integrate f/g dz taking each to be piecwise linear. This is
%    more accurate when f/g has a near-pole in an interval
%    f,g and z are 1D complex arrays.
%
%    Based on newlip routine by Ed Williams.

    zdif = z(2:end)-z(1:end-1);

    out  = sum(ratcen(f,g).*zdif,2);

end

function out=ratcen(f,g)

%    Return "rationally centered" f/g
%    such that int_s(1)^s(0) ds f(s)/g(s) = sum(ratcen(f,g)*s(dif)) when
%    f and g are linear functions of s.
%    This allows accurate integration through near poles of f/g
%
%    Based on newlip routine by Ed Williams.

    fdif = f(:,2:end)-f(:,1:end-1);
    gdif = g(2:end)-g(1:end-1);
    fav  = 0.5*(f(:,2:end)+f(:,1:end-1));
    gav  = 0.5*(g(2:end)+g(1:end-1));
    
    out  = 0.*fdif;
        
    iflat  = abs(gdif) < 1.e-4*abs(gav);

    tmp    = (fav.*gdif - gav.*fdif);
    rf     = fav./gav + tmp.*gdif./(12.*gav.^3);
        
    rfn    = fdif./gdif + tmp.*log((gav+0.5*gdif)./ ...
                                                  (gav-0.5*gdif))./gdif.^2;
    
    out(:,iflat)  = rf(:,iflat);
    out(:,~iflat) = rfn(:,~iflat);
        
end