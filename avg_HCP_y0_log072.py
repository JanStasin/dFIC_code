# load libraries
import sys
import time
import scipy.io as sio
import scipy.signal as sig
from scipy.stats import zscore
from tvb.simulator.lab import *
from tvb.simulator.monitors import Bold
import matplotlib.pyplot as plt
import os
import multiprocessing as mp 
import itertools
from tvb.basic.neotraits.api import NArray, List, Range, Final
import random
import sys

import math
import numpy
#from .base import ModelNumbaDfun, Model
from numba import guvectorize, float64


sys_args = sys.argv
your_y0_target = float(sys_args[1])
print(f'your_y0_target :  {your_y0_target }')

run_fic_tunning = False
run_post_fic = True
run_noFIC = True
save_the_data = True


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
Jansen-Rit and current Feedback Inhibitory Control implementation by...

"""
from jansen_rit_FIC import wFICJansenRit
from jansen_rit_postFIC import JansenRitPostFIC

SC_LOC = '/data/gpfs-1/users/stasinsj_c/scratch/JR_sims/FIC_sims/HCP_mats/'
OUTPUT_LOC = '/data/gpfs-1/users/stasinsj_c/scratch/JR_sims/FIC_sims/HCP/y0_log/'

s_conn = connectivity.Connectivity(weights = sio.loadmat(SC_LOC + 'avgSC_DK.mat')['SC_avg_weights'],
                                tract_lengths= sio.loadmat(SC_LOC +'avgSC_DK.mat')['SC_avg_dists'])
s_conn.weights = s_conn.weights - s_conn.weights*numpy.eye(s_conn.weights.shape[0])
#s_conn.weights = s_conn.scaled_weights(mode='tract')
s_conn.weights= numpy.log10(s_conn.weights+1)
s_conn.region_labels=numpy.loadtxt('/fast/users/stasinsj_c/work/SCZ_replic/sample_SC/VP_4013/region_labels.txt', dtype=str) 
s_conn.centres = numpy.loadtxt('/fast/users/stasinsj_c/work/SCZ_replic/sample_SC/VP_4013/dummy_centres2.txt',dtype=float)
s_conn.configure()


### setting the mu and y0_target corresponding array based on uncoupled bif diagram:
mu_y0_dict = numpy.load('/data/gpfs-1/users/stasinsj_c/scratch/JR_sims/FIC_sims/HCP/mu_y0_dict.npy' , allow_pickle=True).item()

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
        print('y0 input value > max or < min of the y0 array: setting mu to closest available value')
        closest_index = numpy.argmin(numpy.abs(y0_arr - input_value))
    else:
        closest_index = numpy.argmin(numpy.abs(y0_arr - input_value))
    # Get the corresponding value from FP_mu using the index
    try: 
        corresponding_value = mu_arr[closest_index+shift]
    except:
        corresponding_value = mu_arr[closest_index]
    print(f'y0:{input_value} on {reg} -------> mu: {numpy.round(corresponding_value, 4)} idx ={closest_index}  !!!')
    return numpy.round(corresponding_value, 4)

### setting the y0 specific mu value:
mu_val = find_closest_value(your_y0_target, mu_y0_dict, 3)


def get_init_conds(input_value, FP_y0, FP_y1, FP_y2, FP_y3, FP_y4, FP_y5, shift=10):
    # Find the index of the closest value in FP_y0 to the input_value
    closest_index = numpy.argmin(numpy.abs(FP_y0 - input_value))
    # Get the corresponding value from state variables using the index + the shift
    print(f'y0:{input_value} -------> idx: {closest_index+shift} !!!')
    #print(FP_y0.shape)
    try: 
        corresponding_values = [FP_y0[closest_index+shift], FP_y1[closest_index+shift], FP_y2[closest_index+shift],
         FP_y3[closest_index+shift],FP_y4[closest_index+shift],FP_y5[closest_index+shift]]
    except:
        corresponding_values = [FP_y0[closest_index], FP_y1[closest_index], FP_y2[closest_index],
         FP_y3[closest_index],FP_y4[closest_index],FP_y5[closest_index]]
    #print(numpy.round(corresponding_values, 4))
    return numpy.round(corresponding_values, 4)

mu_values, y0s, y1s, y2s, y3s, y4s, y5s, _ =numpy.load('/data/gpfs-1/users/stasinsj_c/scratch/JR_sims/FIC_sims/HCP/all_y2_vs_mu_arrays.npy' , allow_pickle=True)


def run_fic_sim(fic_settings):
    '''defines the separate runction to run the simulation with the model defined above
    input: parameter'''
    s_conn = connectivity.Connectivity(weights = sio.loadmat(SC_LOC + 'avgSC_DK.mat')['SC_avg_weights'],
                                    tract_lengths= sio.loadmat(SC_LOC +'avgSC_DK.mat')['SC_avg_dists'])
    s_conn.weights = s_conn.weights - s_conn.weights*numpy.eye(s_conn.weights.shape[0])
    #s_conn.weights = s_conn.scaled_weights(mode='tract')
    s_conn.weights= numpy.log10(s_conn.weights+1)
    s_conn.region_labels=numpy.loadtxt('/fast/users/stasinsj_c/work/SCZ_replic/sample_SC/VP_4013/region_labels.txt', dtype=str)
    s_conn.centres = numpy.loadtxt('/fast/users/stasinsj_c/work/SCZ_replic/sample_SC/VP_4013/dummy_centres2.txt',dtype=float)
    s_conn.speed = numpy.array([5])
    s_conn.configure()

    jrm_fic = wFICJansenRit(v0=numpy.array([6.]), 
                                        variables_of_interest=['PSP', 'y0', 'y0_d', 'y2_d', 'wFIC' ],
                                        eta = numpy.array([0.]),
                                        y0_target=numpy.array([fic_settings[2]]),
                                        tau_d=numpy.array([fic_settings[3]]))
    
    if fic_settings[4] == 'det':
        your_integrator = integrators.HeunDeterministic(dt=0.1)
    if fic_settings[4] == 'stoch':
        sigma         = numpy.zeros(9) 
        sigma[3]      = 0.00000025
        your_integrator = integrators.HeunStochastic(dt=0.1, noise=noise.Additive(nsig=sigma))
        
    sim = simulator.Simulator(connectivity=s_conn,
                        model=jrm_fic,
                        coupling=coupling.SigmoidalJansenRit(a=numpy.array([fic_settings[0]]),r=numpy.array([0.56])),
                        integrator=your_integrator,
                        monitors=[monitors.SubSample(period=1)],
                        initial_conditions = fic_init,
                        simulation_length=15000)
    
    sim.model.mu = numpy.array([fic_settings[5]]);
    sim.configure()
    
    # run the 'transient' simulation
    t_output = sim.run()
    t_ssamp = t_output[0][1]
    t_PSP = t_ssamp[:,0,:,0]
    t_wFIC_ada = t_ssamp[:,4,:,0]
    
    sim.model.eta=numpy.array([fic_settings[1]])
    sim.simulation_length=180000


    output = sim.run()
    ssamp = output[0][1]
    PSP = ssamp[:,0,:,0]
    

    y0_psp = ssamp[:,1,:,0]
    # y0d_psp = ssamp[:,2,:,0]
    # y2d_psp = ssamp[:,3,:,0]
    wFIC_ada = ssamp[:,4,:,0]
    dist_y0_perc = 100-(numpy.mean(numpy.abs(fic_settings[2] - y0_psp[-3000:])/fic_settings[2])*100)
    etaTau_pair = (fic_settings[1],fic_settings[3])
    return (PSP,  wFIC_ada, y0_psp, dist_y0_perc, etaTau_pair)


# def scale_wFICs(weights_array):
#     indegrees = numpy.sum(weights_array, axis=0)
#     norm_indegrees = indegrees / numpy.max(indegrees)
#     wfs = 16 * norm_indegrees
#     #print('scaled wFICs: ', wfs)
#     init_wFIC = numpy.random.uniform(1,1, size=(1,1,weights_array.shape[0],1));
#     for idx, scaled_wFIC in enumerate(wfs):
#         init_wFIC[0,0,idx,0] = scaled_wFIC
        
#     return init_wFIC



### initial conditions for CASE1:
init_y0 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, -10)[0], (1,1,s_conn.weights.shape[0],1))
init_y1 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, -10)[1], (1,1,s_conn.weights.shape[0],1))
init_y2 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, -10)[2], (1,1,s_conn.weights.shape[0],1))
init_y3 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, -10)[3], (1,1,s_conn.weights.shape[0],1))
init_y4 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, -10)[4], (1,1,s_conn.weights.shape[0],1))
init_y5 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, -10)[5], (1,1,s_conn.weights.shape[0],1))
init_wFIC = numpy.random.uniform(1,1, size=(1,1,s_conn.weights.shape[0],1));
#init_wFIC = scale_wFICs(s_conn.weights)
init_PSP = init_y1 - (init_wFIC * init_y2)
init_y0d = init_y0.copy();
init_y2d = init_y2.copy();
fic_init_CASE1 =  numpy.stack([init_y0,init_y1,init_y2,
                                init_y3,init_y4,init_y5,
                                init_y0d, init_y2d, init_wFIC, init_PSP],axis = 1).squeeze(axis=2)
#print(init_wFIC)
### initial conditions for CASE2
init_y0 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 10)[0], (1,1,s_conn.weights.shape[0],1))
init_y1 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 10)[1], (1,1,s_conn.weights.shape[0],1))
init_y2 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 10)[2], (1,1,s_conn.weights.shape[0],1))
init_y3 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 10)[3], (1,1,s_conn.weights.shape[0],1))
init_y4 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 10)[4], (1,1,s_conn.weights.shape[0],1))
init_y5 = numpy.reshape(numpy.ones(s_conn.weights.shape[0]) * get_init_conds(your_y0_target, y0s, y1s, y2s, y3s, y4s, y5s, 10)[5], (1,1,s_conn.weights.shape[0],1))
init_wFIC = numpy.random.uniform(1,1, size=(1,1,s_conn.weights.shape[0],1));
#init_wFIC = scale_wFICs(s_conn.weights)
init_PSP = init_y1 - (init_wFIC * init_y2)
init_y0d = init_y0.copy();
init_y2d = init_y2.copy();

fic_init_CASE2 =  numpy.stack([init_y0,init_y1,init_y2,
                                init_y3,init_y4,init_y5,
                                init_y0d, init_y2d, init_wFIC, init_PSP],axis = 1).squeeze(axis=2)


if run_fic_tunning:
    print('Running fic tunning...')
    #gc_vals = numpy.array([0,10])
    gc_vals = numpy.arange(0,30,1)
    for gc_val in gc_vals:
        your_tau_d = 100 ## in ms
        your_eta = 0.0025 # so far 0.003 is reasonable
        detORstoch = 'det'
        
        case = f'C_{str(your_y0_target)}'
        print(f'setting up tunning target {case} gc: {gc_val} ..')

        if your_y0_target <= 0.02:
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            fic_init = fic_init_CASE1
        elif your_y0_target > 0.095:
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            fic_init = fic_init_CASE2
        elif your_y0_target <= 0.095 and your_y0_target >= 0.08:
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            fic_init = fic_init_CASE2
        else:
            print('y0_target in the incorrect range')

        print(fic_settings)
        print(f'Running tunning for {case} and gc: {gc_val} ..')
        fic_results = run_fic_sim(fic_settings)#gc_range_arr #or: b_range_arr #or: noise_var_arr
        if save_the_data:
            numpy.save(OUTPUT_LOC + f'sim_results_FICtuning_gc_{gc_val}_{fic_settings[-1]}_HCP_Xx.npy', fic_results, allow_pickle=True)
        print('done for... ', gc_val, case)
else:
    print('Skipping tuning step or it already has been done')

if run_fic_tunning:
    print('loading fic data..')
    res_dict = {}
    for res_file in os.listdir(OUTPUT_LOC):
        if res_file.startswith('sim_results') and res_file.endswith('HCP_Xx.npy'):
            if your_y0_target == float(res_file.split('_')[6]):
                print(res_file)
                gc_val = int(float(res_file.split('_')[4]))
                print(f'gcval: {gc_val}')
                #print(res_file.split('_'))
                key_string = 'C_' + res_file.split('_')[6] + '_' + str(gc_val)
                print('key_string', key_string)
                res_dict[key_string] = numpy.load(OUTPUT_LOC + res_file, allow_pickle=True)
    res_dict.keys()

    tunning_wFICs = {}

    for key, res in res_dict.items():
        tunning_wFICs[key] = res[1]
else:
    print('loading wFICs from file')
    tunning_wFICs = {}
    wFIC_dict = numpy.load('/data/gpfs-1/users/stasinsj_c/scratch/JR_sims/FIC_sims/HCP/wFICs_preload.npy', allow_pickle=True).item()
    print(wFIC_dict.keys())
    for key in wFIC_dict.keys():
        if your_y0_target == float(key.split('_')[1]):
            print(float(key.split('_')[1]))
            tunning_wFICs[key] = wFIC_dict[key]


tunning_wFICs = dict(sorted(tunning_wFICs.items()))
#print(tunning_wFICs)
tunning_wFICs_z =  [[key, value] for key, value in tunning_wFICs.items()]
print(tunning_wFICs.keys())
#print(tunning_wFICs.values())
print('len(tunning_wFICs_z)', len(tunning_wFICs_z))
print('!')


def GetTransientMonitorExtension(monitorClass: monitors.Monitor):
    class ExtentedMonitor(monitorClass):
        def __init__(self, transient_time=0, **kwargs):
            super(ExtentedMonitor, self).__init__(**kwargs)
            self.transient_time = transient_time

        def sample(self, step, state):
            time = step * self.dt
            if (time >= self.transient_time):
                return super(ExtentedMonitor, self).sample(step, state)

    return ExtentedMonitor

def run_tuned_jr(post_input , seed):
    postfic_sv_inds = list(range(6)) + [9]
    #case = f'C_{str(str(your_y0_target))}'
    gc_value = post_input[0].split('_')[2]
    your_y0 = post_input[0].split('_')[1]

    print(f'gc = {gc_value }, y0: {your_y0}')

    post_settings = fic_settings
    post_settings[0] = float(gc_value)
    wFIC_ada = post_input[1]
    wFIC_means = numpy.mean(wFIC_ada[-8000:,:], axis=0) 
    #print(gc_value, wFIC_means)
    s_conn = connectivity.Connectivity(weights = sio.loadmat(SC_LOC + 'avgSC_DK.mat')['SC_avg_weights'],
                                    tract_lengths= sio.loadmat(SC_LOC +'avgSC_DK.mat')['SC_avg_dists'])
    s_conn.weights = s_conn.weights - s_conn.weights*numpy.eye(s_conn.weights.shape[0])
    #s_conn.weights = s_conn.scaled_weights(mode='tract')
    s_conn.weights= numpy.log10(s_conn.weights+1)
    s_conn.region_labels=numpy.loadtxt('/fast/users/stasinsj_c/work/SCZ_replic/sample_SC/VP_4013/region_labels.txt', dtype=str) 
    s_conn.centres = numpy.loadtxt('/fast/users/stasinsj_c/work/SCZ_replic/sample_SC/VP_4013/dummy_centres2.txt',dtype=float)
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


    premium_SubSamp = GetTransientMonitorExtension(monitors.SubSample)
    premium_AffCoup = GetTransientMonitorExtension(monitors.AfferentCoupling)

    jrm_tuned = JansenRitPostFIC(mu=your_mu,
                                    v0=numpy.array([6.]),
                                    variables_of_interest=['PSP', 'y0'], #,'y1','y2','y3','y4', 'y5'],
                                    pFIC = wFIC_means)

    tuned_sim = simulator.Simulator(connectivity=s_conn,
                        model=jrm_tuned,
                        coupling=coupling.SigmoidalJansenRit(a=numpy.array([post_settings[0]]),r=numpy.array([0.56])),
                        integrator=your_integrator,
                        monitors=[premium_SubSamp(transient_time=1.8e+6-150000, period=1), BoldJRMonitor(period=720),  premium_AffCoup(transient_time=1.8e+6-150000)],
                        initial_conditions = post_init,
                        simulation_length=1.8e+6)
    
    tuned_sim.configure()
    print('configured and ready to start post simulation')

    output = tuned_sim.run()
    ssamp = output[0][1]
    bold = output[1][1]
    psp = ssamp[:,0,:,0]
    y0_psp = ssamp[:,1,:,0]
    aff_coup= output[2][1][:,0,:,0]
    print('outputs processed and ready to be returned')

    return (psp, y0_psp, bold, gc_value, aff_coup)

if not run_fic_tunning:
    for gc_val in range(30):
        if run_post_fic:
            print(f'Running postFIC simulations for {your_y0_target} ...')
            postfic_sv_inds = list(range(6)) + [9]
            your_eta = 0.0025
            your_tau_d = 100
            detORstoch = 'stoch'
            case = f'C_{str(your_y0_target)}'
            if your_y0_target <= 0.02:
                print(f'1 y0t: {your_y0_target} ')
                fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
                post_init = fic_init_CASE1[:,postfic_sv_inds,:,:]
            elif your_y0_target > 0.095:
                print(f'2 y0t: {your_y0_target} ')
                print(gc_val)
                fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
                post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]
            elif your_y0_target <= 0.095 and your_y0_target >= 0.08:
                print(f'3 y0t: {your_y0_target} ')
                fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
                post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]
            
            t2 = time.time()
            p = mp.Pool(processes=30)
            #C1_sim_results_100z = p.map(run_tuned_jr, C1_gc_wFICs_100z)
            postFIC_results = p.starmap(run_tuned_jr, [(arg, random.randint(0, 1000)) for arg in tunning_wFICs_z])
            p.close()

            print(f'post sims for {your_y0_target} finished in {time.time()-t2}')
            if save_the_data:
                numpy.save(OUTPUT_LOC + f'post_fic_results_stability_HCP_{your_y0_target}_log072.npy' , postFIC_results, allow_pickle=True)
                print('..and saved')
        else:
            print('Skipping post-FIC step')

        C0_gc_noFIC = {}
        for key in tunning_wFICs.keys():
            C0_gc_noFIC[key] = numpy.ones([10,84])
        C0_gc_noFIC = [[key, value] for key, value in C0_gc_noFIC.items()]

        if run_noFIC:
            print('Running no fic simulations...')
            postfic_sv_inds = list(range(6)) + [9]
            your_eta = 0.0025
            your_tau_d = 100
            if your_y0_target <= 0.02:
                fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
                post_init = fic_init_CASE1[:,postfic_sv_inds,:,:]
            elif your_y0_target > 0.09:
                fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
                post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]
            elif your_y0_target <= 0.09 and your_y0_target >= 0.08:
                fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
                post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]

            t6 = time.time()
            p = mp.Pool(processes=30)
            sim_results_noFIC = p.starmap(run_tuned_jr, [(arg, random.randint(0, 1000)) for arg in C0_gc_noFIC])
            if save_the_data:
                numpy.save(OUTPUT_LOC + f'no_fic_sim_results_HCP_{your_y0_target}_log072.npy' , sim_results_noFIC, allow_pickle=True)
                print(case, ' saved')
            p.close()
            print(time.time()-t6)
            print('no_FIC finished')
        else:
            print('Skipping no-FIC step')

        print(f'GC search script for {your_y0_target} has run till the end')

else:
    if run_post_fic:
        print(f'Running postFIC simulations for {your_y0_target} ...')
        postfic_sv_inds = list(range(6)) + [9]
        your_eta = 0.0025
        your_tau_d = 100
        detORstoch = 'stoch'
        case = f'C_{str(your_y0_target)}'
        if your_y0_target <= 0.02:
            print(f'1 y0t: {your_y0_target} ')
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            post_init = fic_init_CASE1[:,postfic_sv_inds,:,:]
        elif your_y0_target > 0.095:
            print(f'2 y0t: {your_y0_target} ')
            print(gc_val)
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]
        elif your_y0_target <= 0.095 and your_y0_target >= 0.08:
            print(f'3 y0t: {your_y0_target} ')
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]
        
        t2 = time.time()
        p = mp.Pool(processes=30)
        #C1_sim_results_100z = p.map(run_tuned_jr, C1_gc_wFICs_100z)
        postFIC_results = p.starmap(run_tuned_jr, [(arg, random.randint(0, 1000)) for arg in tunning_wFICs_z])
        p.close()

        print(f'post sims for {your_y0_target} finished in {time.time()-t2}')
        if save_the_data:
            numpy.save(OUTPUT_LOC + f'post_fic_results_stability_HCP_{your_y0_target}_log072.npy' , postFIC_results, allow_pickle=True)
            print('..and saved')
    else:
        print('Skipping post-FIC step')

    C0_gc_noFIC = {}
    for key in tunning_wFICs.keys():
        C0_gc_noFIC[key] = numpy.ones([10,84])
    C0_gc_noFIC = [[key, value] for key, value in C0_gc_noFIC.items()]

    if run_noFIC:
        print('Running no fic simulations...')
        postfic_sv_inds = list(range(6)) + [9]
        your_eta = 0.0025
        your_tau_d = 100
        if your_y0_target <= 0.02:
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            post_init = fic_init_CASE1[:,postfic_sv_inds,:,:]
        elif your_y0_target > 0.09:
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]
        elif your_y0_target <= 0.09 and your_y0_target >= 0.08:
            fic_settings = [gc_val, your_eta, your_y0_target, your_tau_d, detORstoch, mu_val, case]
            post_init = fic_init_CASE2[:,postfic_sv_inds,:,:]

        t6 = time.time()
        p = mp.Pool(processes=30)
        sim_results_noFIC = p.starmap(run_tuned_jr, [(arg, random.randint(0, 1000)) for arg in C0_gc_noFIC])
        if save_the_data:
            numpy.save(OUTPUT_LOC + f'no_fic_sim_results_HCP_{your_y0_target}_log072.npy' , sim_results_noFIC, allow_pickle=True)
            print(case, ' saved')
        p.close()
        print(time.time()-t6)
        print('no_FIC finished')
    else:
        print('Skipping no-FIC step')

    print(f'GC search script for {your_y0_target} has run till the end')