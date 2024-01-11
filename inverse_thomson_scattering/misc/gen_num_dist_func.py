#import jax.scipy.interpolate
#from jax.scipy.special import gamma
from jax import numpy as jnp
#import scipy.io as sio
#from os.path import join, exists
#from typing import Dict
import time
from inverse_thomson_scattering.misc import dist_functional_forms

class DistFunc:
    def __init__(self, config):
        self.dict = config["parameters"]["fe"]
        self.dict["m"] = config["parameters"]["m"]["val"]
        self.velcity_res = 0.01
        
        #done here so it only is done once
        if "f1_direction" in self.dict.keys():
            self.dict["f1_direction"] = self.dict["f1_direction"]/jnp.sqrt(jnp.sum([ele**2 for ele in self.dict["f1_direction"]])) 
        self.velocity, self.value = self.gen_num_dist()
        
        
    def gen_num_dist(self):
        
        fe_name = list(self.dict["type"].keys())[0]
    
        if fe_name == "DLM":
            if self.dict["dim"] == 1:
                v,fe = dist_functional_forms.DLM_1D(self.dict["m"], self.velcity_res)
            elif self.dict["dim"] == 2:
                v,fe = dist_functional_forms.DLM_2D(self.dict["m"], self.velcity_res)

        elif fe_name == "Spitzer":
            if self.dict["dim"] == 2:
                if len(self.dict["f1_direction"]) == 2:
                    v,fe = dist_functional_forms.Spitzer_2V(self.dict["dt"], self.dict["f1_direction"], self.velcity_res)
                elif len(self.dict["f1_direction"]) == 3:
                    v,fe = dist_functional_forms.Spitzer_3V(self.dict["dt"], self.dict["f1_direction"], self.velcity_res)
            else:
                raise ValueError('Spitzer distribution can only be computed in 2D')

        elif fe_name == "MYDLM":
            if self.dict["dim"] == 2:
                if len(self.dict["f1_direction"]) == 2:
                    v,fe = dist_functional_forms.MoraYahi_2V(self.dict["dt"], self.dict["f1_direction"], self.velcity_res)
                elif len(self.dict["f1_direction"]) == 3:
                    v,fe = dist_functional_forms.MoriYahi_3V(self.dict["dt"], self.dict["f1_direction"], self.velcity_res)
            else:
                raise ValueError('Mora and Yahi distribution can only be computed in 2D')
    

        return v, fe
        
    
    #need a function that just changes the numerical values of the distribtuion function
    #need a function that changes the values based off changes to the parameters (this may just be a call to the constructor)
    def project_onto_k(self, k):
        return
        
#def get_num_dist_func