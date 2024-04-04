
'''The folloring script allows to run Jansen-Rit simulation with dynamic Feedback Inhibiton Control algorithm 
for chosen GC and y0 target values. It consititutes the necesaary basis to replicate the findings from:
    
'Homeodynamic feedback inhibition control in whole-brain simulations'

and for adjusting the code to insividual needs.
code written by J.Stasinski
Brain Simulation Section, BIH, Charite Universitatzmedizin
    '''

# load libraries
import sys
import time
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import zscore
from tvb.simulator.lab import *
from tvb.simulator.monitors import Bold
import os
import itertools
from tvb.basic.neotraits.api import NArray, List, Range, Final
import random
import sys

import math
import numpy
#from .base import ModelNumbaDfun, Model
from numba import guvectorize, float64

## SEtting some params
your_y0_target = float(0.01)
print(f'your_y0_target :  {your_y0_target }')

run_fic_tunning = True
run_post_fic = True
save_the_data = True

gc_val = numpy.array([10])
your_y0_target = numpy.array([0.01])
your_eta = numpy.array([0.001])
your_tau_d =  numpy.array([1000])
detORstoch = 'det'

class BoldJRMonitor(Bold):
    def sigmoidal(self, state):
        e0 = numpy.array([0.0025]) # 2.5 Half of the maximum firing rate
        v0 = numpy.array([6.])
        r0 = numpy.array([0.56])
        return (2  * e0) / (1 + numpy.exp(r0 * (v0 - state)))
    def sample(self, step, state):
        state[self.voi] = self.sigmoidal(state[self.voi])
        return super(BoldJRMonitor, self).sample(step, state)

def SigmoidJR(ts, sim):
    e0 = sim.model.nu_max # 2.5 Half of the maximum firing rate
    v0 = sim.model.v0 #V1/2
    r0 = sim.model.r
    return (2 * e0) / (1 + numpy.exp(r0 * (v0 - ts)))

"""
Jansen-Rit and current Feedback Inhibitory Control implementation by J.Stasinski et al

"""
from jansen_rit_FIC import wFICJansenRit
from jansen_rit_postFIC import JansenRitPostFIC

# set up a sample connectome:
sample_sc_weights = numpy.array([[1,3],[3,1]])
sample_sc_lengths = numpy.array([[0.7,1],[1,0.7]])
sample_rls = numpy.array(['1','2'])
sample_centres = numpy.array([[0,0],[0,0]])

s_conn = connectivity.Connectivity(weights = sample_sc_weights,
                                tract_lengths= sample_sc_lengths)
s_conn.weights = s_conn.weights - s_conn.weights*numpy.eye(s_conn.weights.shape[0])
s_conn.weights = s_conn.scaled_weights(mode='tract')
#s_conn.weights= numpy.log10(s_conn.weights+1)
s_conn.region_labels=sample_rls
s_conn.centres = sample_centres
s_conn.speed = numpy.array([5])
s_conn.configure()


### setting the mu and y0_target corresponding array based on uncoupled bif diagram:
mu_y0_dict = numpy.load('mu_y0_dict.npy' , allow_pickle=True).item()

def find_closest_value(input_value, muy_dict, shift=1):
    # Find the index of the closest value in FP_y0 to the input_value
    if input_value <= 0.0189 or (input_value > 0.079 and input_value < 0.097) or input_value >= 0.125:
        reg = 'FP' 
    elif input_value > 0.0189 and input_value < 0.065:
        reg = 'SLC'
    elif input_value >= 0.097 and input_value < 0.125:
        reg = 'FLC'
    mu_arr, y0_arr = muy_dict[reg]
    mu_arr = numpy.array(mu_arr)
    y0_arr = numpy.array(y0_arr)
    if input_value > numpy.max(y0_arr) or input_value < numpy.min(y0_arr) :
        #print('y0 input value > max or < min of the y0 array: setting mu to closest available value')
        closest_index = numpy.argmin(numpy.abs(y0_arr - input_value))
    else:
        closest_index = numpy.argmin(numpy.abs(y0_arr - input_value))
    # Get the corresponding value from FP_mu using the index
    try: 
        corresponding_value = mu_arr[closest_index+shift]
    except:
        corresponding_value = mu_arr[closest_index]
    #print(f'y0:{input_value} on {reg} -------> mu: {numpy.round(corresponding_value, 4)} idx ={closest_index}  !!!')
    return numpy.round(corresponding_value, 4)

### setting the y0 specific mu value:
mu_val = find_closest_value(your_y0_target, mu_y0_dict, 3)


def get_init_conds(input_value, FP_y0, FP_y1, FP_y2, FP_y3, FP_y4, FP_y5, shift=5):
    # Find the index of the closest value in FP_y0 to the input_value
    closest_index = numpy.argmin(numpy.abs(FP_y0 - input_value))
    # Get the corresponding value from state variables using the index + the shift
    #print(f'y0:{input_value} -------> idx: {closest_index+shift} !!!')
    try: 
        corresponding_values = [FP_y0[closest_index+shift], FP_y1[closest_index+shift], FP_y2[closest_index+shift],
         FP_y3[closest_index+shift],FP_y4[closest_index+shift],FP_y5[closest_index+shift]]
    except:
        corresponding_values = [FP_y0[closest_index], FP_y1[closest_index], FP_y2[closest_index],
         FP_y3[closest_index],FP_y4[closest_index],FP_y5[closest_index]]
    return numpy.round(corresponding_values, 4)


# loading state variables to mu correspondence from file
mu_values, y0s, y1s, y2s, y3s, y4s, y5s, _ =numpy.load('all_ys_vs_mu_array.npy' , allow_pickle=True)


def run_fic_sim(fic_settings):
    '''defines the separate runction to run the simulation with the model defined above
    input: parameter'''
    s_conn = connectivity.Connectivity(weights = sample_sc_weights,
                                    tract_lengths= sample_sc_lengths)
    s_conn.weights = s_conn.weights - s_conn.weights*numpy.eye(s_conn.weights.shape[0])
    s_conn.weights = s_conn.scaled_weights(mode='tract')
    #s_conn.weights= numpy.log10(s_conn.weights+1)
    s_conn.region_labels=sample_rls
    s_conn.centres = sample_centres
    s_conn.speed = numpy.array([5])
    s_conn.configure()

    jrm_fic = wFICJansenRit(v0=numpy.array([6.]), 
                            variables_of_interest=['PSP', 'y0', 'y0_d', 'y2_d', 'wFIC' ],
                            eta = your_eta,
                            y0_target=your_y0_target,
                            tau_d = your_tau_d)
    
    if fic_settings[4] == 'det':
        your_integrator = integrators.HeunDeterministic(dt=0.1)

        
    if fic_settings[4] == 'stoch':
        sigma         = numpy.zeros(9) 
        sigma[3]      = 0.00000025
        your_integrator = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=sigma))

        
    sim = simulator.Simulator(connectivity=s_conn,
                        model=jrm_fic,
                        coupling=coupling.SigmoidalJansenRit(a=gc_val,r=numpy.array([0.56])),
                        integrator=your_integrator,
                        monitors=[monitors.SubSample(period=1)],
                        initial_conditions = fic_init_UNI,
                        simulation_length=15000)
    
    sim.model.mu = numpy.array([fic_settings[5]]);
    sim.configure()

    
    # run the 'transient' simulation
    t_output = sim.run()
    t_ssamp = t_output[0][1]
    t_PSP = t_ssamp[:,0,:,0]
    t_wFIC_ada = t_ssamp[:,4,:,0]


    sim.simulation_length=180000

    output = sim.run()
    ssamp = output[0][1]
    PSP = ssamp[:,0,:,0]

    y0_psp = ssamp[:,1,:,0]
    wFIC_ada = ssamp[:,4,:,0]
    return PSP,  wFIC_ada, y0_psp

### initial conditions for CASE1:
init_y0 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 0)[0], (1,1,s_conn.weights.shape[0],1))
init_y1 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 0)[1], (1,1,s_conn.weights.shape[0],1))
init_y2 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 0)[2], (1,1,s_conn.weights.shape[0],1))
init_y3 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 0)[3], (1,1,s_conn.weights.shape[0],1))
init_y4 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 0)[4], (1,1,s_conn.weights.shape[0],1))
init_y5 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 0)[5], (1,1,s_conn.weights.shape[0],1))
init_wFIC = numpy.random.uniform(1,1, size=(1,1,s_conn.weights.shape[0],1));
init_PSP = init_y1 - (init_wFIC * init_y2)
init_y0d = init_y0.copy();
init_y2d = init_y2.copy();
fic_init_UNI =  numpy.stack([init_y0,init_y1,init_y2,
                                init_y3,init_y4,init_y5,
                                init_y0d, init_y2d, init_wFIC, init_PSP],axis = 1).squeeze(axis=2)

print(fic_init_UNI.shape)
#print(fic_init_UNI)

if run_fic_tunning:
    print('Running fic tunning...')
    if your_y0_target <= 0.02 or your_y0_target > 0.095:
        fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, str(your_y0_target)]
        #fic_init = fic_init_UNI
    else:
        print('y0_target in the incorrect range')

    print(f'Running tunning for y0_target {fic_settings[2]} and gc: {gc_val}')
    fic_results = run_fic_sim(fic_settings)
    tunning_wFICs = fic_results[1]
    if save_the_data:
        numpy.save(f'sim_results_FICtuning_gc_{float(gc_val)}_{float(fic_settings[2])}.npy', fic_results, allow_pickle=True)
        print(f'done for gc: {gc_val} y0t: {fic_settings[2]}')
else:
    print('Skipping tuning step or it already has been done')
    print('Running post FIC only')
    fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, str(your_y0_target)]
    fic_results = numpy.load(f'sim_results_FICtuning_gc_{float(gc_val)}_{float(fic_settings[2])}.npy', allow_pickle=True)
    tunning_wFICs = fic_results[1]

# if run_fic_tunning:
#     print('loading fic data..')
#     res_dict = {}
#     for res_file in os.listdir(OUTPUT_LOC):
#         if res_file.startswith('sim_results') and res_file.endswith('HCP_Xx.npy'):
#             if your_y0_target == float(res_file.split('_')[6]):
#                 print(res_file)
#                 gc_val = int(float(res_file.split('_')[4]))
#                 print(f'gcval: {gc_val}')
#                 #print(res_file.split('_'))
#                 key_string = 'C_' + res_file.split('_')[6] + '_' + str(gc_val)
#                 print('key_string', key_string)
#                 res_dict[key_string] = numpy.load(OUTPUT_LOC + res_file, allow_pickle=True)
#     res_dict.keys()

# tunning_wFICs = dict(sorted(tunning_wFICs.items()))
# #print(tunning_wFICs)
# tunning_wFICs_z =  [[key, value] for key, value in tunning_wFICs.items()]
# print(tunning_wFICs.keys())
# #print(tunning_wFICs.values())
# print('len(tunning_wFICs_z)', len(tunning_wFICs_z))
# print('!')




def run_tuned_jr(post_input , seed):
    postfic_sv_inds = list(range(6)) + [9]

    post_settings = fic_settings
    gc_val = float(post_settings[0])
    wFIC_ada = post_input
    wFIC_means = numpy.mean(wFIC_ada[-8000:,:], axis=0) 
    s_conn = connectivity.Connectivity(weights = sample_sc_weights,
                                    tract_lengths= sample_sc_lengths)
    s_conn.weights = s_conn.weights - s_conn.weights*numpy.eye(s_conn.weights.shape[0])
    s_conn.weights = s_conn.scaled_weights(mode='tract')

    s_conn.region_labels=sample_rls
    s_conn.centres = sample_centres
    s_conn.speed = numpy.array([5])
    s_conn.configure()
    
    if post_settings[4]== 'det':
        your_integrator = integrators.HeunDeterministic(dt=0.1)
        your_mu = numpy.array([post_settings[5]]);
        #your_mu = numpy.array([0.]);
    if post_settings[4] == 'stoch':
        sigma         = numpy.zeros(7) 
        sigma[3]      = 0.0000001
        your_integrator = integrators.HeunStochastic(dt=1, noise=noise.Additive(nsig=sigma,
                                                                                  noise_seed=seed))
        your_mu = numpy.array(post_settings[5]);

    premium_SubSamp = monitors.SubSample


    jrm_tuned = JansenRitPostFIC(mu=your_mu,
                                    v0=numpy.array([6.]),
                                    variables_of_interest=['PSP', 'y0'], #,'y1','y2','y3','y4', 'y5'],
                                    pFIC = wFIC_means)

    tuned_sim = simulator.Simulator(connectivity=s_conn,
                        model=jrm_tuned,
                        coupling=coupling.SigmoidalJansenRit(a=post_settings[0],r=numpy.array([0.56])),
                        integrator=your_integrator,
                        monitors=[premium_SubSamp(period=1), BoldJRMonitor(period=720)],
                        initial_conditions = post_init,
                        simulation_length=60000)
    
    tuned_sim.configure()
    print('configured and ready to start post simulation')

    output = tuned_sim.run()
    ssamp = output[0][1]
    bold = output[1][1]
    psp = ssamp[:,0,:,0]
    y0_psp = ssamp[:,1,:,0]

    print('outputs processed and ready to be returned')

    return psp, y0_psp, bold

if run_post_fic:
    print(f'Running postFIC simulations for {your_y0_target} ...')
    postfic_sv_inds = list(range(6)) + [9]
    your_eta = 0.001
    your_tau_d = 1000
    detORstoch = 'stoch'
    case = str(your_y0_target)
    fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
    post_init = fic_init_UNI[:,postfic_sv_inds,:,:]
    
    postFIC_results = run_tuned_jr(tunning_wFICs, random.randint(0, 1000))

    t2 = time.time()
    
    print(f'post sims for {your_y0_target} finished in {time.time()-t2}')
    if save_the_data:
        
        numpy.save(f'post_fic_results_HCP_PSPs_{float(your_y0_target)}.npy' , [postFIC_results[0], postFIC_results[1]], allow_pickle=True)
        numpy.save(f'post_fic_results_HCP_BOLDSs_{float(your_y0_target)}.npy' , postFIC_results[2], allow_pickle=True)
        print('..and saved')
else:
    print('Skipping post-FIC step')
