def zprimeMaxw(xi):

    # This function calculates the derivitive of the Z - function given an array of normilzed phase velocities(xi) as
    # defined in Chapter 5. For values of xi between - 10 and 10 a table is used. Outside of this range the assumtotic
    # approximation(see Eqn. 5.2.10) is used.

    p.rdWT
if isempty(rdWT)
    load('rdWT');
end
persistent
idWT;
if isempty(idWT)
    load('idWT');
end

% xi = -70:0.1: 70;
% ai = find(xi < -10, 1, 'last');
% bi = find(xi > 10, 1, 'first');
bi = find(xi < -10, 1, 'first');
ai = find(xi > 10, 1, 'last');

aj = find(xi < -10);
bj = find(xi > 10);
cj = setdiff(1:length(xi), [aj bj]);

[tmp npts] = size(xi);
if ((isempty(ai)) & & (isempty(bi)))
    rZp = interp1(rdWT(1,:), rdWT(2,:), xi);
    iZp = interp1(idWT(1,:), idWT(2,:), xi);

    elseif((isempty(ai)) & & (isempty(bi) == 0))
    rZp(1: bi - 1)=interp1(rdWT(1,:), rdWT(2,:), xi(1: bi - 1));
    iZp(1: bi - 1)=interp1(idWT(1,:), idWT(2,:), xi(1: bi - 1));
    rZp(bi: npts)= xi(bi: npts).^ (-2);
    iZp(bi: npts)=0.0;

elseif((isempty(ai) == 0) & & (isempty(bi)))
rZp(ai + 1: npts)=interp1(rdWT(1,:), rdWT(2,:), xi(ai + 1: npts));
iZp(ai + 1: npts)=interp1(idWT(1,:), idWT(2,:), xi(ai + 1: npts));
rZp(1: ai) = xi(1: ai).^ (-2);
iZp(1: ai)=0.0;

else
rZp(setdiff(1: length(xi), [aj bj])) = interp1(rdWT(1,:), rdWT(2,:), xi(cj));
iZp(setdiff(1: length(xi), [aj bj])) = interp1(idWT(1,:), idWT(2,:), xi(cj));
rZp(aj) = xi(aj). ^ (-2);
iZp(aj) = 0.0;
rZp(bj) = xi(bj). ^ (-2);
iZp(bj) = 0.0;
end

Zp(1,:) = rZp;
Zp(2,:) = iZp;
% plot(Zp)
% StevensTime = toc