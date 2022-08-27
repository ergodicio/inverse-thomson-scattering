import matplotlib.pyplot as plt
import numpy as np

def LinePlots(x, y, f=[], XLabel='v/v_{th}', YLabel='Amplitude (arb.)', CurveNames=[], Residuals=[], title='Simulated Thomson Spectrum'):

    if np.shape(x)[0]==0:
        x=np.arange(max(np.shape(y)))
        
    if np.shape(x)[0]!=np.shape(y)[0]:
        #This occurs if multiple curves are submitted as part of y
        y=y.transpose()
        
    if Residuals:
        fig, ax1 = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [2,1]})
        ax1[0].plot(x,y)
        #possibly change to stem
        ax1[1].plot(x,Residuals)

    if f:
        fig, ax1 = plt.subplots()
        ax1.plot(x,y,'b')
        ax2=ax1.twinx()
        ax2.plot(x,f,'h')
    
        ax2.set_yscale('log')
    
        ax1.tick_params(axis='y', color='k', labelcolor='k')
        ax2.tick_params(axis='y', color='g', labelcolor='g')
    
        ax2.set_ylabel('Amplitude (arb.)', color='g')

    else:
        fig, ax1 = plt.subplots()
        ax1.plot(x,y,)

    ax1.set_title(title)
    ax1.set_ylabel(YLabel)
    ax1.set_xlabel(XLabel)

    if CurveNames:
        ax1.legend(CurveNames)
