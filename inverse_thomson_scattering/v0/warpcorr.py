from os.path import join
import scipy.io as sio
from scipy.ndimage import geometric_transform
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sciint

def warpCorrection(warpedData, instrument = 'EPW', sweepSpeed = 5, flatField = True):
    """
    Returns a dewarped streak camera image.
    
    Args:
        warpedData: The streak camera image to be dewarped
        instrument: 'EPW' or 'IAW' corresponding to the diangostic instrument 
        sweepSpeed: sweep time in ns based on camera settings
        flatField: Flag to use flatfiled data for a flat field correction
        
    Returns:
        dewarped: The dewarped data
    
    function [deWarped,time,xValidity,yValidity] = OMEGAWarpCorrection(warped,instrument,sweepSpeed,flatField)
    """


    all_correctionData = sio.loadmat(join('files','OMEGAWarpData.mat'))

    if instrument == 'EPW':
        if sweepSpeed == 5:
            correctionData = all_correctionData['EPW5ns']
            A=[0.0217315398715013, 1.04771565298347, 0.00114076491134120, -0.00365092193432767, -0.0331078993435843, 0.00621625693661584, -0.000782921156147619, -0.00395775516184532, -0.0197461507855143, 9.85950939022705e-05, 7.80298773058964e-05, -0.000336292572749386, -1.72585636425512e-05, 0.00386060540157644, -0.000320180057158966]
            B=[-0.000394641149009378, -0.0202145445848165, 0.989391957822401, -0.00115020532228386, -0.000475857852773813, -0.000123467717788895, -0.00209963649799810, -0.000535680132109373, 0.000553276977418358, 0.00237074532417962, 0.000315740097099913, -0.000195236640456297, 8.29262922032786e-05, 0.000177042193100683, 0.000520654963240587]
            flatfieldData = all_correctionData['EPWFlatFieldCorrection5ns']
        elif sweepSpeed == 15:
            correctionData = all_correctionData['EPW15ns']
            flatfieldData = all_correctionData['EPWFlatFieldCorrection15ns']
        else:
            correctionData = all_correctionData['EPW5ns']
            flatfieldData = all_correctionData['EPWFlatFieldCorrection5ns']
            print('no specific data avaiable for this sweep speed - using 5ns dewarp')

    if instrument == 'IAW':
        if sweepSpeed == 5:
            correctionData = all_correctionData['IAW5ns']
            flatfieldData = all_correctionData['IAWFlatFieldCorrection5ns']
        elif sweepSpeed == 15:
            correctionData = all_correctionData['IAW15ns']
            flatfieldData = all_correctionData['IAWFlatFieldCorrection15ns']
        else:
            correctionData = all_correctionData['IAW5ns']
            flatfieldData = all_correctionData['IAWFlatFieldCorrection5ns']
            print('no specific data avaiable for this sweep speed - using 5ns dewarp')
            
    #Set up transform
    #print(correctionData)
    #print(correctionData.keys())
    #A = correctionData['transform']['A']
    #B = correctionData['transform']['B']
    #xValidity = correctionData['xValidity']
    #yValidity = correctionData['yValidity']
    def polynomialtransform2D(coords):
        [X,Y]=coords
        X=X/1024
        Y=Y/1024
        U = (A[0] + A[1]*X + A[2]*Y + A[3]*X*Y + A[4]*X**2 + A[5]*Y**2
            + A[6]*X**2.*Y + A[7]*X*Y**2 + A[8]*X**3 + A[9]*Y**3 
            + A[10]*X**3.*Y + A[11]*X**2.*Y**2 + A[12]*X*Y**3 + A[13]*X**4 + A[14]*Y**4)
        
        V = (B[0] + B[1]*X + B[2]*Y + B[3]*X*Y + B[4]*X**2 + B[5]*Y**2
            + B[6]*X**2.*Y + B[7]*X*Y**2 + B[8]*X**3 + B[9]*Y**3 
            + B[10]*X**3.*Y + B[11]*X**2.*Y**2 + B[12]*X*Y**3 + B[13]*X**4 + B[14]*Y**4)
        return (U*1024,V*1024)


    print(polynomialtransform2D((0,0)))
    print(polynomialtransform2D((1,1)))
    print(polynomialtransform2D((1024,1024)))
    #Apply the flat field correction
    #if flatField:
    #    warpedData = warpedData*flatfieldData

    #warpedData=np.arange(0,1024**2).reshape(1024,1024)
    #Dewarp the image
    dewarped = geometric_transform(warpedData, polynomialtransform2D, mode = 'constant', cval = 0.0)
    #deWarped = imwarp(warped,tForm,'FillValues',nan);
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    imI = ax[0].imshow(warpedData)
    imI = ax[1].imshow(dewarped)
    
    #dewarp v2
    [XT,YT] = np.shape(warpedData)
    dewarped2=np.zeros((XT,YT))
    XT = np.arange(0, XT, 1)
    YT = np.arange(0, YT, 1)
    XT, YT = np.meshgrid(XT,YT)
    UT = geometric_transform(XT, polynomialtransform2D, mode = 'constant', cval = 0.0)
    #print(XT)
    #print(UT)
    VT = geometric_transform(YT, polynomialtransform2D, mode = 'constant', cval = 0.0)
    #print(YT)
    #print(VT)
    points=np.transpose(np.vstack([UT.ravel(),VT.ravel()]));
    vals=warpedData.ravel()
    dewarped2=sciint.griddata(points,warpedData.ravel(),(XT,YT))
    print(np.shape(dewarped2))
    
    #for i in XT:
    #    for j in YT:
    #        U,V=polynomialtransform2D((i,j))
    #        val=sciint.interpn((XT,YT), warpedData,(U,V),bounds_error=False)
    #        dewarped2[i,j]=val
    
    imI = ax[2].imshow(dewarped2)
    return dewarped2
    ##### Omiting the rest of the code to test transfrom
    #Calculate the intensity correction by calcaulating area mapping
    #Calculate the area of each pixel in the transformed space
    #find the corners of each square pixel:
    [YT,XT] = np.shape(dewarped2)
    XT = np.arange(0.5, XT+0.5, 1)
    YT = np.arange(0.5, YT+0.5, 1)
    XT, YT = np.meshgrid(XT,YT)
    #Calculate the corresponding area for this pixel from the sampled space
    #[XD, YD] = transformPointsInverse(tForm,XT,YT);
#areaSource =quadrilateralGridArea(XD,YD);
#% Pixel value is interpolated from the warped  space. If pixel "area" was
#% smaller in the warped space then signal is temporally and spacially
#% over-concentrated and signal level should therefore be reduced in the
#% transformed space
#deWarped = double(deWarped).*double(areaSource);


#%% Get the time base
#time = warpCorrectionData.time;
#    function A = quadrilateralGridArea(X,Y)
#       % Function returns the areas of the irregular quadrilateral grids
#        % from which the dewarped image is sampled.
#        % Used to correct the
#        % X Y are arrays of XY corrds of the verticles of a grid of
#        % irregular quadrilaterals.
#        % Returns A, the area of of the quadrilaterals
#        
#        % Calculate the length of 'Top' edges
#        A = sqrt(diff(X,[],2).^2+diff(Y,[],2).^2);
#        % Calculate the length of 'Left' edges
#        B = sqrt(diff(X,[],1).^2+diff(Y,[],1).^2);
#       % Calculate the length of diagnoals
#       D = sqrt((X(1:end-1,1:end-1)-X(2:end,2:end)).^2+(Y(1:end-1,1:end-1)-Y(2:end,2:end)).^2);
#        % For wach cell:
#        % Top
#        A1 = A(1:end-1,:);
#        % Bottom
#       A2 = A(2:end,:);
#       % Left
#        B1 = B(:,1:end-1);
#       % Right
#       B2 = B(:,2:end);
#       % Perimeter of upper triangle
#       P1 = (A1+B1+D)./2;
#      % Area of upper triangle
#        T1 = sqrt(P1.*(P1-A1).*(P1-B1).*(P1-D));
#        %Perimeter of lower triangle
#        P2 = (A2+B2+D)./2;
#        % Area of lower triangle
#        T2 = sqrt(P2.*(P2-A2).*(P2-B2).*(P2-D));
#        % Total area of irregular quadrilatal
#        A = T1+T2;
#    end
#
#end

