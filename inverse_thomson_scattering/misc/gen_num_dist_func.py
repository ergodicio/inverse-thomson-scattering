# import jax.scipy.interpolate
# from jax.scipy.special import gamma
from jax import numpy as jnp

# import scipy.io as sio
# from os.path import join, exists
# from typing import Dict
import time
from inverse_thomson_scattering.misc import dist_functional_forms


# needs the ability to enforce symetry
class DistFunc:
    def __init__(self, config):
        self.velocity_res = config["parameters"]["fe"]["v_res"]
        self.fe_name = list(config["parameters"]["fe"]["type"].keys())[0]

        if "dim" in config["parameters"]["fe"].keys():
            self.dim = config["parameters"]["fe"]["dim"]
        else:
            self.dim = 1

        if "dt" in config["parameters"]["fe"].keys():
            self.dt = config["parameters"]["fe"]["dt"]

        # normalized here so it only is done once
        if "f1_direction" in config["parameters"]["fe"].keys():
            self.f1_direction = config["parameters"]["fe"]["f1_direction"] / jnp.sqrt(
                jnp.sum([ele**2 for ele in config["parameters"]["fe"]["f1_direction"]])
            )
        # temperature asymetry for biDLM with Tex = Te and Tey = Te*temp_asym
        if "temp_asym" in config["parameters"]["fe"].keys():
            self.temp_asym = config["parameters"]["fe"]["temp_asym"]
        else:
            self.temp_asym = 1.0

        # m asymetry for biDLM with mx = m and my = m*m_asym (with a min of 2)
        if "m_asym" in config["parameters"]["fe"].keys():
            self.m_asym = config["parameters"]["fe"]["m_asym"]
        else:
            self.m_asym = 1.0

        # rotion angle for the biDLM defined counter clockwise from the x-axis in degrees
        if "m_theta" in config["parameters"]["fe"].keys():
            self.m_theta = config["parameters"]["fe"]["m_theta"] / 180.0 * jnp.pi
        else:
            self.m_theta = 0.0

    def __call__(self, mdict):
        if self.fe_name == "DLM":
            if self.dim == 1:
                v, fe = dist_functional_forms.DLM_1D(mdict["val"], self.velocity_res)
            elif self.dim == 2:
                # v, fe = dist_functional_forms.DLM_2D(mdict["val"], self.velocity_res)
                v, fe = dist_functional_forms.BiDLM(
                    mdict["val"], max(mdict["val"] * self.m_asym, 2.0), self.temp_asym, self.m_theta, self.velocity_res
                )

        elif self.fe_name == "Spitzer":
            if self.dim == 2:
                if len(self.f1_direction) == 2:
                    v, fe = dist_functional_forms.Spitzer_2V(self.dt, self.f1_direction, self.velocity_res)
                elif len(self.f1_direction) == 3:
                    v, fe = dist_functional_forms.Spitzer_3V(self.dt, self.f1_direction, self.velocity_res)
            else:
                raise ValueError("Spitzer distribution can only be computed in 2D")

        elif self.fe_name == "MYDLM":
            if self.dim == 2:
                if len(self.f1_direction) == 2:
                    v, fe = dist_functional_forms.MoraYahi_2V(self.dt, self.f1_direction, self.velocity_res)
                elif len(self.f1_direction) == 3:
                    v, fe = dist_functional_forms.MoriYahi_3V(self.dt, self.f1_direction, self.velocity_res)
            else:
                raise ValueError("Mora and Yahi distribution can only be computed in 2D")

        return v, fe

    # need a function that just changes the numerical values of the distribtuion function
    # need a function that changes the values based off changes to the parameters (this may just be a call to the constructor)
    def rotate(self, fe, v, theta):
        # create new grid
        x, y = jnp.meshgrid(jnp.arange(fe.shape(0)), jnp.arange(fe.shape(0)))
        return


# def get_num_dist_func
