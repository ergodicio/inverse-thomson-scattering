function ZpNonMaxw = ZprimeNonMaxw(curDist,xs,distTable)
% ZPRIMENONMAXW loads the specified table or calls MAKEINTEGRALTABLE to
% create theappropriate table if one cannot be found. Interpolation is then
% preformed on the table to return values for the queried x vaules(xs) and
% the queried distribution function(curDist). queried x values outside a
% normalized phase velocity of -8 to 8 are calculated using an asymptotic
% approximation.
%
%
%Last update: 3/12/19 A.L.Milder
%Confirmed working with DLM and numeric distributions
%% Check if open table is the wrong table or empty, if this is the case load correct table

persistent IT;
persistent DFNOld;

%check if distribution is numeric and treat differently
if isempty(distTable) && isnumeric(curDist)
    if isempty(IT) || ~isnumeric(DFNOld) || sum(DFNOld-curDist(:,2))
        warning('Susceptibility is being calculated for a numeric distribution function')
        Zp=calcZprimeNumeric(curDist);
    else
        Zp=IT;
    end
    if xs(1)>xs(end)
        xs=flip(xs);
    end
    Zpr=interpn(Zp(:,1),Zp(:,2),xs,'linear',0);
    Zpi=interpn(Zp(:,1),Zp(:,3),xs,'linear',0);
    ZpNonMaxw=[flip(Zpr); flip(Zpi)];
    
    IT=Zp;
    DFNOld=curDist(:,2);
    return
end

[DFName,params]=GenTableName(distTable{:});

if isempty(IT) || ~strcmp(DFNOld,DFName)
    tabl=['susceptibility\' DFName '.mat'];

    if exist(tabl,'file')
        load(tabl,'IT')
        DFNOld=DFName;

    else
        answer= questdlg(['No susceptibility table for this '...
            'distribution function was found. Do you want to create a '...
            'table? This may take hours.'],'Warning','Yes','No','No');
        if strcmp(answer,'Yes')
            tic
            MakeIntegralTable(distTable,DFName);
            toc
            load(tabl,'IT');
            DFNOld=DFName;
        else
            error('No susceptibility table was found or created')
        end

    end
end

%% Interpolate to find requested susceptibility
%check is xs is ascending or decending
ascend=1;
if xs(1)>xs(end)
    ascend=0;
    xs=flip(xs);
end
xs1=xs(xs<-8);
xs3=xs(xs>8);
xs2=xs(xs<=8);
xs2=xs2(xs2>=-8);
switch length(curDist)
        case 2
            [X1,X2]=ndgrid(xs2,curDist{2});
            interpedSectionr=interpn(IT.x,params.m,IT.Chir,X1,X2);
            interpedSectioni=interpn(IT.x,params.m,IT.Chii,X1,X2);
        case 3
            [X1,X2,X3]=ndgrid(xs2,curDist{2},curDist{3});
            interpedSectionr=interpn(IT.x,params.m,params.Z,IT.Chir,X1,X2,X3);
            interpedSectioni=interpn(IT.x,params.m,params.Z,IT.Chii,X1,X2,X3);
end
rZp=[xs1.^(-2), interpedSectionr' ,xs3.^(-2)];
iZp=[zeros(1,length(xs1)), interpedSectioni' ,zeros(1,length(xs3))];

if ~ascend
    %this might fail for higher dimentional rZp,iZp
    rZp=flip(rZp);
    iZp=flip(iZp);
end

ZpNonMaxw(1,:) = rZp;
ZpNonMaxw(2,:) = iZp;


end
