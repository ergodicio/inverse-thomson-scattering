def ratintn(f, g, z):

    # Integrate f / g dz taking each to be piecwise linear.This is more accurate when f / g has a near - pole in an
    # interval f, g and z are 1D complex arrays.
    #
    # Based on newlip routine by Ed Williams.

    zdif = z[1:-1]-z[0:-2]

    out = sum(ratcen(f, g) * zdif, 1)
    return out

def ratcen(f, g):

    # Return "rationally centered" f / g such that int_s(1) ^ s(0) ds f(s) / g(s) = sum(ratcen(f, g) * s(dif)) when
    # f and g are linear functions of s.
    # This allows accurate integration through near poles of f / g
    #
    # Based on newlip routine by Ed Williams.

    from numpy import log
    fdif = f[:, 1:-1]-f[:, 0:-2]
    gdif = g[1:-1]-g[0:-2]
    fav = 0.5 * (f[:,1:-1] + f[:, 0:-2])
    gav = 0.5 * (g[1:-1] + g[0:-2])

    out = 0. * fdif

    iflat = abs(gdif) < 1.e-4 * abs(gav)

    tmp = (fav * gdif - gav * fdif)
    rf = fav / gav + tmp * gdif / (12. * gav**3)

    rfn = fdif / gdif + tmp * log((gav + 0.5 * gdif) / (gav - 0.5 * gdif)) / gdif**2

    out[:,iflat]  = rf[:, iflat]
    out[:, ~iflat] = rfn[:, ~iflat]

    return out