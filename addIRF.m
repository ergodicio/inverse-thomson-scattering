function [Thry]=addIRF(width,lamAxis,data)
% ADDIRF adds a gaussian instrument response function to the data with a
% FWHM given the the input parameter width

    stddev=width/2.3548;
    origin=(max(lamAxis)+min(lamAxis))/2; %Conceptual_origin so the convolution donsn't shift the signal
    
    
    inst_func = (1/(stddev*sqrt(2*pi)))*exp(-(lamAxis-origin).^2/(2*(stddev)^2));
    %gamma=2*sqrt(2*log(2))*stddev;
    %inst_func=(1/pi)*.5*gamma/((lamAxis-origin).^2 + (.5*gamma)^2);
    for i=1:size(data,1)
        Thry(i,:) = conv(data(i,:), inst_func,'same');
        Thry(i,:)=(sum(data(i,:))./sum(Thry(i,:))).*Thry(i,:);
    end
%     Thry = conv(data, inst_func,'same');
%     %Thry=(max(data)./max(Thry)).*Thry;
%     Thry=(sum(data)./sum(Thry)).*Thry; 
end