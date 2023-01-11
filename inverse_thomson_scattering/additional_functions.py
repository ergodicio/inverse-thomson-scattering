#collection of small functions used by the new datafitter
import mlflow, tempfile

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from os.path import join

from inverse_thomson_scattering.numDistFunc import get_num_dist_func
from inverse_thomson_scattering.plotstate import plotState

def get_scattering_angles(spectype):
    if spectype > 1:
        # Scattering angle in degrees for OMEGA TIM6 TS
        sa = dict(
            sa=np.linspace(53.637560, 66.1191, 10),
            weights=np.array(
                [
                    0.00702671050853565,
                    0.0391423809738300,
                    0.0917976667717670,
                    0.150308544660150,
                    0.189541011666141,
                    0.195351560740507,
                    0.164271879645061,
                    0.106526733030044,
                    0.0474753389486960,
                    0.00855817305526778,
                ]
            ),
        )
    else:
        # Scattering angle in degrees for Artemis
        imp=sio.loadmat(join('files','angleWghtsFredfine.mat'), variable_names='weightMatrix')
        weights=imp['weightMatrix']
        sa = dict(sa=np.arange(19, 139.5, 0.5), weights=weights)
    return sa
        
def plotinput(config, sa, fit_model):
    parameters = config["parameters"]

    # Setup x0
    xie = np.linspace(-7, 7, parameters["fe"]["length"])

    NumDistFunc = get_num_dist_func(parameters["fe"]["type"], xie)
    parameters["fe"]["val"] = np.log(NumDistFunc(parameters["m"]["val"]))
    parameters["fe"]["lb"] = np.multiply(parameters["fe"]["lb"], np.ones(parameters["fe"]["length"]))
    parameters["fe"]["ub"] = np.multiply(parameters["fe"]["ub"], np.ones(parameters["fe"]["length"]))

    x0 = []
    lb = []
    ub = []
    xiter = []
    for i, _ in enumerate(config["lineoutloc"]["val"]):
        for key in parameters.keys():
            if parameters[key]["active"]:
                if np.size(parameters[key]["val"])>1:
                    x0.append(parameters[key]["val"][i])
                elif isinstance(parameters[key]["val"], list):
                    x0.append(parameters[key]["val"][0])
                else:
                    x0.append(parameters[key]["val"])
                lb.append(parameters[key]["lb"])
                ub.append(parameters[key]["ub"])

    x0=np.array(x0)
    fit_model = get_fit_model(config, xie, sa)

    print("plotting")
    mlflow.set_tag("status", "plotting")

    fig = plt.figure(figsize=(14, 6))
    with tempfile.TemporaryDirectory() as td:
        fig.clf()
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        fig, ax = plotState(
            x0,
            config,
            [1, 1],
            xie,
            sa,
            [],
            fitModel2=fit_model,
            fig=fig,
            ax=[ax, ax2],
            )
        fig.savefig(os.path.join(td, "simulated_spectrum.png"), bbox_inches="tight")
        mlflow.log_artifacts(td, artifact_path="plots")
    return