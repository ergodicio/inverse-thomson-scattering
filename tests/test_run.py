import tempfile
import time
import multiprocessing as mp
import yaml
import numpy as np
import mlflow
import os
import flatdict
from flatten_dict import flatten, unflatten
from jax import config
from jax import jit
from jax import numpy as jnp
from copy import deepcopy

config.update("jax_enable_x64", True)

from scipy.signal import find_peaks
from inverse_thomson_scattering.v0 import datafitter
from inverse_thomson_scattering.jax.form_factor import get_form_factor_fn
from inverse_thomson_scattering.v0.numDistFunc import get_num_dist_func


def log_params(cfg):
    flattened_dict = dict(flatdict.FlatDict(cfg, delimiter="."))
    num_entries = len(flattened_dict.keys())

    if num_entries > 100:
        num_batches = num_entries % 100
        fl_list = list(flattened_dict.items())
        for i in range(num_batches):
            end_ind = min((i + 1) * 100, num_entries)
            trunc_dict = {k: v for k, v in fl_list[i * 100 : end_ind]}
            mlflow.log_params(trunc_dict)
    else:
        mlflow.log_params(flattened_dict)


def update(base_dict, new_dict):
    combined_dict = {}
    for k, v in new_dict.items():
        combined_dict[k] = base_dict[k]
        if isinstance(v, dict):
            combined_dict[k] = update(base_dict[k], v)
        else:
            combined_dict[k] = new_dict[k]

    return combined_dict


def test_run():
    #Test #1: Bohm-Gross test, calculate a spectrum and compare the resonance to the Bohm gross dispersion relation
    #Test #2: IAW test, calculate a spectrum and compare the resonance to the IAW dispersion relation
    #Test #3: Data test, compare fit to a preknown fit result
    
    with open("./defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("./inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)
            
    #Test #1: 
    nonMaxwThomsonE_jax, _ = get_form_factor_fn([400, 700])
    xie = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
    sa = np.array([60])
    num_dist_func = get_num_dist_func({"DLM":[]}, xie)
    fecur = num_dist_func(2.0)
    lam = 526.5

    ThryE, lamAxisE = jit(nonMaxwThomsonE_jax)(0.5,0.2,1,1,1,np.array([.2*1e20]),0,0, sa, (fecur, xie),lam)

    ThryE = np.squeeze(ThryE)
    #print(type(ThryE))
    test = deepcopy(np.asarray(ThryE))
    #test.setflags(write=1)
    #print(type(test))
    peaks, peak_props = find_peaks(test, height = (.1, 1.1), prominence = .2)
    #print(peaks)
    highest_peak_index = peaks[np.argmax(peak_props['peak_heights'])]
    second_highest_peak_index = peaks[np.argpartition(peak_props['peak_heights'],-2)[-2]]
    
    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    Mp = Me * 1836.1  # proton mass keV/C^2
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm
    constants = jnp.sqrt(4 * jnp.pi * Esq / Me)
    
    #print(highest_peak_index)
    #print(second_highest_peak_index)
    lams = lamAxisE[0,[highest_peak_index, second_highest_peak_index],0]
    #print(lams)
    omgs = 2 * jnp.pi * C / lams #peak frequencies
    #print(omgs)
    omgpe = constants * jnp.sqrt(.2*1e20)
    omgL = 2 * np.pi * 1e7 * C / lam  # laser frequency Rad / s
    ks = jnp.sqrt(omgs**2 - omgpe**2) / C
    kL = jnp.sqrt(omgL**2 - omgpe**2) / C
    k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sa * jnp.pi / 180))
    vTe = jnp.sqrt(0.5/Me)
    omg = jnp.sqrt(omgpe**2 + 3*k**2*vTe**2)
    #print(omg)
    omgs2 = [omgL+omg[0], omgL-omg[1]]
    #print(omgs2)
    #print(omgs)
    Deltas = np.asarray(omgs2)-np.asarray(omgs)
    #print(Deltas)
    if abs(Deltas[0]/omgs[0])<0.005 and abs(Deltas[1]/omgs[1])<0.005:
        test1 = True
        print("EPW peaks are within 0.5% of Bohm-Gross values")
    else:
        test1 = False
        print("EPW peaks are NOT within 0.5% of Bohm-Gross values")
        
    #Test #2:
    nonMaxwThomsonI_jax, _ = get_form_factor_fn([525, 528])
    ThryI, lamAxisI = jit(nonMaxwThomsonI_jax)(0.5,0.2,1,1,1,np.array([.2*1e20]),0,0, sa, (fecur, xie),lam)

    ThryI = jnp.real(ThryI)
    ThryI = jnp.mean(ThryI, axis=0)
    
    ThryI = np.squeeze(ThryI)
    test = deepcopy(np.asarray(ThryI))
    peaks, peak_props = find_peaks(test, height = .1, prominence = .2)
    #print(peaks)
    highest_peak_index = peaks[np.argmax(peak_props['peak_heights'])]
    second_highest_peak_index = peaks[np.argpartition(peak_props['peak_heights'],-2)[-2]]
    
    lams = lamAxisI[0,[highest_peak_index, second_highest_peak_index],0]
    omgs = 2 * jnp.pi * C / lams #peak frequencies
    #print(omgs)
    omg = 2* kL * jnp.sqrt((0.5+3*0.2)/Mp)
    omgs2 = [omgL+omg, omgL-omg]
    #print(omg)
    #print(omgs2)
    #print(omgs)
    Deltas = np.asarray(omgs2)-np.asarray(omgs)
    #print(Deltas)
    if abs(Deltas[0]/omgs[0])<0.005 and abs(Deltas[1]/omgs[1])<0.005:
        test2 = True
        print("IAW peaks are within 0.5% of dispersion relation values")
    else:
        test2 = False
        print("IAW peaks are NOT within 0.5% of dispersion relation values")

    
    #Test #3: Data test, compare fit to a preknown fit result
    #currently just runs one line of shot 101675 for the electron, should be expanded in the future
    
    with open("./defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("./inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    bgshot = {"type": [], "val": []}
    # bgshot = {"type": "Fit", "val": 102584}
    lnout = {"type": "pixel", "val": [500]}
    # lnout = {"type": "um", "val": slices}
    bglnout = {"type": "pixel", "val": 900}
    extraoptions = {"spectype": 2}
    
    config["parameters"]["Te"]["val"] = 0.5
    config["parameters"]["ne"]["val"] = 0.2 #0.25
    config["parameters"]["m"]["val"] = 3.0 #2.2

    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        log_params(config)

        config["bgshot"] = bgshot
        config["lineoutloc"] = lnout
        config["bgloc"] = bglnout
        config["extraoptions"] = extraoptions
        config["num_cores"] = int(mp.cpu_count())

        config = {**config, **dict(shotnum=101675, bgscale=1, dpixel=2)}

        mlflow.log_params({"num_slices": 1})
        t0 = time.time()
        # mlflow.log_params(flatten(config))
        fit_results = datafitter.fit(config=config)
        metrics_dict = {"datafitter_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")
        
        #print((fit_results["amp1"]["val"]-0.9257)/0.9257)
        #print((fit_results["amp2"]["val"]-0.6727)/0.6727)
        #print((fit_results["lam"]["val"]-524.2455)/524.2455)
        #print((fit_results["Te"]["val"]-0.67585)/0.67585)
        #print((fit_results["ne"]["val"]-0.21792)/0.21792)
        #print((fit_results["m"]["val"]-3.3673)/3.3673)
        if ((fit_results["amp1"]["val"]-0.9257)/0.9257 < 0.05 and
            (fit_results["amp2"]["val"]-0.6727)/0.6727 < 0.05 and
            (fit_results["lam"]["val"]-524.2455)/524.2455 < 0.05 and
            (fit_results["Te"]["val"]-0.67585)/0.67585 < 0.05 and
            (fit_results["ne"]["val"]-0.21792)/0.21792 < 0.05 and
            (fit_results["m"]["val"]-3.3673)/3.3673 < 0.1):
            test3 = True
            print("Fit values are within 5-10% of known values")
        else:
            test3 = False
            print("Fit values do NOT agree with known values")