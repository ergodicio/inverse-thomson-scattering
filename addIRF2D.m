function [Thry]=addIRF2D(width1,width2,ax1,ax2,data)
% ADDIRF adds a gaussian instrument response function to the data with a
% FWHM given the the input parameter width
%ax1 is the xaxis or the axis corresponding to a row lineout
%ax2 is the yaxis or the axis corresponding to a column lineout

stddev1=width1/2.3548;
origin1=(max(ax1)+min(ax1))/2; %Conceptual_origin so the convolution donsn't shift the signal

inst_func1 = (1/(stddev1*sqrt(2*pi)))*exp(-(ax1-origin1).^2/(2*(stddev1)^2));

stddev2=width2/2.3548;
origin2=(max(ax2)+min(ax2))/2; %Conceptual_origin so the convolution donsn't shift the signal

inst_func2 = (1/(stddev2*sqrt(2*pi)))*exp(-(ax2-origin2).^2/(2*(stddev2)^2));
%gamma=2*sqrt(2*log(2))*stddev;
%inst_func=(1/pi)*.5*gamma/((lamAxis-origin).^2 + (.5*gamma)^2);

Thry=conv2(inst_func2,inst_func1,data,'same');
Thry=repmat(sum(data,2)./sum(Thry,2),1,size(data,2)).*Thry;
% for i=1:size(data,1)
%     Thry(i,:) = conv(data(i,:), inst_func,'same');
%     Thry(i,:)=(sum(data(i,:))./sum(Thry(i,:))).*Thry(i,:);
% end
%     Thry = conv(data, inst_func,'same');
%     %Thry=(max(data)./max(Thry)).*Thry;
%     Thry=(sum(data)./sum(Thry)).*Thry; 
end