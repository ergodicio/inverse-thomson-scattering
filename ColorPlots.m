function ColorPlots(x,y,C,varargin)
%options={'Line','CLims','LogPlot','XLabel','CurveNames','Residuals'}
p=inputParser;
addRequired(p,'x',@isnumeric);
addRequired(p,'y',@isnumeric);
addRequired(p,'C',@(C) isequal(size(C),[length(y) length(x)]));
addOptional(p,'Line',[],@isnumeric);
addOptional(p,'CLims',[],@isnumeric);
addOptional(p,'LogPlot',0,@isboolean);
addOptional(p,'Title','EPW feature v \theta', @isstr);
addOptional(p,'Name','', @isstr);
addOptional(p,'XLabel','\lambda_s(nm)');
%kaxis requires density and temperature as inputs in order to
%calculate the alpha values [ne,Te,lam]
addOptional(p,'KAxis',[],@isnumeric);
addOptional(p,'CurveNames',{},@iscellstr);
addOptional(p,'Residuals',[])
parse(p,x,y,C,varargin{:});
options=p.Results;

figure('Units','normalized','position',[.1 .1 .35 .6],'Name', options.Name)
set(gcf,'color','w')

load('cmap_white0.mat')
colormap(TS_mycmap);
%colormap('bone');
%colormap(flipud(colormap));

if ~isempty(options.Residuals)
    subplot(1,3,[1,2])
end

if options.LogPlot
    imagesc(x,y,log(C))%Log Plot
else
    %imagesc(x,y,C) %Linear plot
    im=pcolor(x,y,C); %Linear plot
    im.EdgeColor='none';
    %if y(1)>y(end)
    set(gca,'YDir','Reverse');
    set(gca,'layer','top');
    %end
end

if ~isempty(options.CLims)
    caxis(options.CLims)
end

if ~isempty(options.Line)
    hold on
    plot(x,options.Line,'g');
    hold off
end

title(options.Title,'FontSize',16,'FontWeight','bold')
xlabel(options.XLabel,'FontSize',16,'FontWeight','bold')
ylabel('\theta (\circ)','FontSize',16,'FontWeight','bold')

if ~isempty(options.KAxis) && isempty(options.Residuals)
    ax1=gca;
    colorbar('Location','manual','Position',[.87,.12,.05,.8])
    set(ax1,'OuterPosition',[0,0,.8,1],'FontSize',16,'FontWeight','bold')
    ax2 = axes('Position', get(ax1, 'Position'),'yaxislocation', 'right','xaxislocation', 'top', 'box', 'off', 'color', 'none');
    set(ax2, 'XTick',[],'XLabel',[],'FontSize',16,'FontWeight','bold')

    tempyax=get(ax1,'YTick');
    c=2.99792458e10;
    omgL=2*pi*1e7*c/options.KAxis(3);
    omgpe = 5.64*10^4 *sqrt(options.KAxis(1));
    ko = sqrt((omgL^2 - omgpe^2)/c^2);
    tempyax=sqrt(options.KAxis(1)/(1000*options.KAxis(2)))*1./(1486*ko*sin(tempyax/360 *pi));
    set(ax2,'YLim',get(ax1,'YLim'),'YTick',get(ax1,'YTick'),'YDir','reverse','FontSize',16,'FontWeight','bold')
    set(ax2,'YTickLabel',cellstr(num2str(round(tempyax',1)))','FontSize',16,'FontWeight','bold')
    ylabel('~v_p/v_{th}','FontSize',16,'FontWeight','bold')
    axes(ax1)
end

if ~isempty(options.CurveNames)
    legend(options.CurveNames{:})
end


if ~isempty(options.Residuals)
    subplot(1,3,3)
    stem(options.Residuals);
    view(90,90);
end

end


