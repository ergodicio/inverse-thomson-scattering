# This file contains 3 functions for calculating and handleing numerical distribtuion functions
import jax.scipy.interpolate
from jax import numpy as jnp
import numpy as np
import scipy.interpolate as sp
import scipy.io as sio
from os.path import join, exists


def get_num_dist_func(fe_type, xie):
    """NUMDISTFUNC loads the specified table or calls MAKEFTABLE to create the
     appropriate table if one cannot be found. Interpolation is then preformed
     on the table to return values for the queried distribution(curDist) at the
     query points(xs). The input arguments are the ditribution function
     descriptor(curDist), a list of query points(xs), and the table
     discriptor(varargin). Note: queried points outside the table are returned
     as zeros.

       NUMDISTFUNC(curDist) returns the default table for the current
       distribution function at the default points.

       NUMDISTFUNC(curDist,xs) returns the default table for the current
       distribution function at the specified points.

       NUMDISTFUNC(curDist,xs,tableSpecs) returns the specified table for the
       current distribution function at the specified points. each element of
       tableSpecs must be its own variable.

       NUMDISTFUNC(numDist,...) when the first input is a numeric array it is
       interpreted as a numerical distribtuion function and either returned as
       is or inperpolated.

    Last update: 3/12/19 A.L.Milder
    As written it is compatable with DLM and Fourkal, may need slight changes
    for SpitzerDLM, and will need major changes for Gauss
    works in all modes as of 3/12/19 this includes numerical distributions
    Notes:
    curDist is the current distribution function or the requested function
    curDist={'name',param2,param3}
    xs are the query points (referenced to parm1)"""

    # persistent IT;
    # persistent DFNOld;
    # get xs from args
    # if len(args) > 0 and isinstance(args[0], np.ndarray):
    #     xs = args[0]
    xs = xie
    #     if xs[0] < xs[-1]:
    #         xs = xs[::-1]
    #         invertedxs = 1
    #     else:
    #         invertedxs = 0
    #     if len(args) == 1:
    #         args = []
    #     else:
    #         args = args[1:-1]

    # Check if input is numeric fe, assumes numeric fe is ndarray and curDist is tuple
    # if isinstance(curDist, np.ndarray):
    #     if 'xs' in locals():
    #         finterp = sp.interp1d(curDist[:, 0], curDist[:, 1], "cubic", bounds_error=False, fill_value=0)
    #         f1D = finterp(xs)
    #     else:
    #         f1D = curDist[:, 1]
    #     return f1D

    # If input is tuple parse as normal
    #nameCur = fe_type
    fe_type = list(fe_type.keys())[0]
    nameCur = fe_type

    # # Generate Table Name
    # if len(args) == 0:
    #     args = tuple([nameCur])
    # elif args[0] != nameCur:
    #     raise NameError('Specified distribtuion is not part of specified table')

    [DFName, params] = GenTableName(fe_type, nout=2)

    # Load Table
    # if isempty(IT) || ~strcmp(DFNOld,DFName)
    tabl = join("numDistFuncs", str(DFName) + ".mat")
    if exists(tabl):
        tablevar = sio.loadmat(tabl, variable_names="IT")
        IT = tablevar["IT"]
        # DFNOld=DFName;
    else:
        raise NameError("No table for this distribution function was found. Table creation is currently disabled")
        # answer= questdlg(['No table for this distribution function was' ...
        #    'found. Do you want to create a table? This may take hours.'],...
        #    'Warning','Yes','No','No');
        # if strcmp(answer,'Yes')
        #    MakeFTable(varargin{:});
        # else
        #    error('No distribution function was found or created')
        # load(tabl,'IT')

    # Return correct section of table
    if "xs" in locals():
        ai = xs < min(params["x"])
        bi = xs > max(params["x"])
        ci = ~(ai + bi)

    x_float_inds = np.interp(xs, params["x"], np.linspace(0, params["x"].size, params["x"].size))

    def NumDistFunc(m):
        # if len(curDist) == 1:
        #     if len(np.shape(IT)) == 2:
        #         [X1, X2] = np.meshgrid(xs[ci], params["m"])
        #         interpedSection = sp.interpn((params["x"], params["m"]), IT, (X1, X2))
        #     elif len(np.shape(IT)) == 3:
        #         [X1, X2, X3] = np.meshgrid(xs[ci], params["m"], params["Z"])
        #         interpedSection = sp.interpn((params["x"], params["m"], params["Z"]), IT, (X1, X2, X3))
        # elif len(curDist) == 2:
        # X1, X2 = np.meshgrid(xs[ci], m)

        m_float_inds = jnp.array([jnp.interp(m, params["m"], np.linspace(0, params["m"].size, params["m"].size))])

        ind_x_mesh, ind_m_mesh = jnp.meshgrid(x_float_inds, m_float_inds)
        indices = jnp.concatenate([ind_x_mesh.flatten()[:, None], ind_m_mesh.flatten()[:, None]], axis=1)

        f1D = jax.scipy.ndimage.map_coordinates(IT, indices.T, order=1, mode="constant", cval=0.0)
        #
        # interpedSection = jnp.reshape(interpedSection, X1.shape, order="F")

        # interpedSection = sp.interpn((params["x"], params["m"]), IT, (X1, X2))
        # elif len(curDist) == 3:
        #     [X1, X2, X3] = np.meshgrid(xs[ci], curDist[1], curDist[2])
        #     interpedSection = sp.interpn((params["x"], params["m"], params["Z"]), IT, (X1, X2, X3))
        # elif len(curDist) == 4:
        #     if (
        #         len(curDist[2]) == 1
        #     ):  # this currently allows the angle for Spitzer to be specificed as a single angle or a range but there might be a better solution
        #         [X1, X2, X3, X4] = np.meshgrid(xs[ci], curDist[1], curDist[2], curDist[3])
        #         interpedSection = sp.interpn(
        #             (params["x"], params["m"], params["theta"], params["delT"]), IT, (X1, X2, X3, X4)
        #         )
        #     elif len(curDist[2]) == 2:
        #         ts = ~((params["theta"] < curDist[2][0]) + (params["theta"] > curDist[2][1]))
        #         [X1, X2, X3, X4] = np.meshgrid(xs[ci], curDist[1], params["theta"][ts], curDist[3])
        #         interpedSection = sp.interpn(
        #             (params["x"], params["m"], params["theta"], params["delT"]), IT, (X1, X2, X3, X4)
        #         )

        # f1D=np.append([np.zeros([len(xs[bi]),*np.shape(interpedSection)[1:]]), interpedSection, np.zeros([len(xs[ai]),*np.shape(interpedSection)[1:]])],axis=0)
        # This will need to be shecked in cases where ai and bi have finite size
        # f1D = np.append(np.zeros([len(xs[bi]), *np.shape(interpedSection)[1:]]), interpedSection, axis=0)
        # f1D = np.append(f1D, np.zeros([len(xs[ai]), *np.shape(interpedSection)[1:]]), axis=0)

        #     if invertedxs:
        #         f1D=np.flip(f1D);
        # else:
        #     if len(curDist) == 1:
        #         f1D=IT
        #     elif len(curDist) == 2:
        #         [X1, X2] = np.meshgrid(params['x'],curDist[1])
        #         f1D = sp.interpn(params['x'],params['m'],IT,X1,X2)
        #     elif len(curDist) == 3:
        #         [X1, X2, X3] = np.meshgrid(params['x'],curDist[1],curDist[2])
        #         f1D = sp.interpn(params['x'],params['m'],params['Z'],IT,X1,X2,X3)

        return jnp.squeeze(f1D)

    return NumDistFunc


def GenTableName(name, *args, nout=1):
    """GENTABLENAME generates the name of the table based on the the following naming syntax
     DFName_param1_spacing1_min1_max1_param2_spacing2_min2_max2...
     GENTABLENAME also creates the parameters structure, checking to ensure
     correct syntax based off the distribution function.

        name=GENTABLENAME(name) selects a name from the availible hard
         coded functions and uses the default spaceings for the relevent
         parameters. The current hardcoded functions are DLM, Fourkal,
         SpitzerDLM, Gauss. A table is created with the default options and the
         corresponding name.

        name=GENTABLENAME(name, fun, {param1,spacing1,domain1}) creates a name
         from the user supplied function. This table is saved using the name
         supplied. This name cannot conflict with one of the reserved names.

       name=GENTABLENAME(name,{param1,spacing1,domain1},....) selects a function
       from the availible hard coded functions and uses the user defined
       spaceings for the relevent parameters. These spacing should be given in
       log base 10. The domain should be a two element array with the starting
       and ending values. The current hardcoded functions are DLM, Fourkal,
       SpitzerDLM, Gauss.

       name=GENTABLENAME(...) returns just the name of the table

       [name,params]=GENTABLENAME(...) returns the name of the table and the
       parameter structure.

       [name,params,inputs]=GENTABLENAME(...) returns the name of the table,
       the parameter structure and the input structure.


    Dramatically stripped down for python rewrite: 8/19/22 A.L.Milder"""

    # Handle inputs
    numargs = len(args)

    # Prevents reserved funtion names from being passed as custom functions
    if numargs > 0 and callable(args[0]):
        if name in ["DLM", "SpitzerDLM", "Fourkal", "Gauss", "MYDLM"]:
            raise NameError("Distribution function name " + name + " is reserved")

    if numargs > 0:
        raise NameError("Table names with non-default values is currently not supported")

    inputs = dict([])
    params = dict([])
    if name in ["SpitzerDLM", "MYDLM"]:
        inputs["p1"] = ("x", -2, [-10, 10])
        inputs["p2"] = ("m", 0, [2, 5])
        inputs["p3"] = ("theta", -1.2018, [0, 2 * np.pi])
        inputs["p4"] = ("delT", -3, [0, 0.007])
    else:
        inputs["p1"] = ("x", -3, [-10, 10])
        inputs["p2"] = ("m", -1, [2, 5])
        inputs["p3"] = ("Z", 0, [1, 25])
        inputs["p4"] = ("y", -4, [-10, 10])

    for key in inputs:
        params[inputs[key][0]] = np.arange(
            inputs[key][2][0], inputs[key][2][1] + 10 ** inputs[key][1], 10 ** inputs[key][1]
        )

    if name == "DLM":
        paramsUsed = ["p1", "p2"]
    elif name == "Fourkal":
        paramsUsed = ["p1", "p2", "p3"]
    elif name in ["SpitzerDLM", "MYDLM"]:
        paramsUsed = ["p1", "p2", "p3", "p4"]

    # Construct name
    DFName = name
    for i in paramsUsed:
        DFName += "_" + inputs[i][0] + "_" + str(inputs[i][1]) + "_" + str(inputs[i][2][0]) + "_" + str(inputs[i][2][1])

    if nout == 1:
        return DFName
    elif nout == 2:
        return DFName, params
    elif nout == 3:
        DFName, params, inputs
