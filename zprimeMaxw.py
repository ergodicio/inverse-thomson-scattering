def zprimeMaxw(xi):

    import numpy as np
    import scipy.interpolate as sp
    import os
    # This function calculates the derivitive of the Z - function given an array of normilzed phase velocities(xi) as
    # defined in Chapter 5. For values of xi between - 10 and 10 a table is used. Outside of this range the assumtotic
    # approximation(see Eqn. 5.2.10) is used.
    # xi is expected to be ascending

    if not 'rdWT' in globals():
        #global rdWT
        rdWT= np.vstack(np.loadtxt('rdWT.txt'))
    if not 'idWT' in globals():
        #global idWT
        idWT = np.vstack(np.loadtxt('idWT.txt'))
    ai = xi <-10
    bi = xi > 10
    ci = ~(ai + bi)

    rinterp=sp.interp1d(rdWT[:,0], rdWT[:,1],'linear')
    rZp=np.concatenate((xi[ai]**-2, rinterp(xi),xi[bi]**-2))
    iinterp = sp.interp1d(idWT[:, 0], idWT[:, 1], 'linear')
    iZp=np.concatenate((0*xi[ai],iinterp(xi),0*xi[bi]))

    Zp=np.vstack((rZp, iZp))
    #print(np.shape(Zp))
    return Zp