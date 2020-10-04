# - Disable warning display
import warnings
warnings.filterwarnings('ignore')

# Import required modules and configure
from rockpool import TSContinuous
from rockpool.networks import NetworkDeneve

import numpy as np

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt

from Utils import plot_matrices, discretize

import ujson as json


"""
Params: duration (ms), min_fr, max_fr, Num_input
Returns: Triple: rateVectors (Nx-dimensional rate signal)
"""
def get_input_rate(duration, dt, Amp, Nx, sigma):
    # Kernel
    w = (1/(sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,int(1/dt))-500)**2)/(2*sigma**2))
    w = w / np.sum(w)
    # Get the rate vectors
    rateVectors = (np.random.multivariate_normal(np.zeros(Nx), np.eye(Nx), int(duration / dt))).T
    for d in range(Nx):
        rateVectors[d,:100] = 0
        rateVectors[d,:] = Amp*np.convolve(rateVectors[d,:], w, 'same')

    return rateVectors

np.random.seed(42)

tDuration = 1.0
tDt = 0.0001
fCommandAmp = 5000
nNumVariables = 2
sigma = 30

# Generate random decoding weights
nNetSize = 20
mfGamma = np.random.randn(nNumVariables, nNetSize)
# Re-define system parameters 
fLambda_d = 100
fLambda_V = 20
fTauN = 1/fLambda_V
fTauS = 1/fLambda_d
fMu = 0.0001
fNu = 0.001

angles = np.linspace(0,2*np.pi,num=nNetSize+1)[:-1]
D = np.random.randn(nNumVariables, nNetSize)

# Normalize
D = np.divide(D, np.sqrt(np.matmul(np.ones((nNumVariables,1)), np.sum(D**2, axis=0).reshape((1,nNetSize)))))
mfGamma = D

weight_granularity = 2

mfOmega_f = np.random.randn(nNetSize, nNetSize) * 30
# Discretize initial weights
max_weight = max(abs(np.min(mfOmega_f)),np.max(mfOmega_f))
max_weight_array = int(max_weight / weight_granularity)
bin_edges = weight_granularity*np.asarray(np.linspace(-max_weight_array,max_weight_array,2*max_weight_array +1))
mfOmega_f_discrete = discretize(mfOmega_f, bin_edges)
np.fill_diagonal(mfOmega_f_discrete, 0)

# Set slow recurrent weights to zero
mfOmega_s = np.zeros((nNetSize,nNetSize))

# Determine the firing thresholds
vfT = (fNu * fLambda_d + fMu * fLambda_d**2 + np.sum(abs(mfGamma.T), -1, keepdims = True)**2) / 2

optimal_reset = vfT - np.reshape(np.diag(mfGamma.T @ mfGamma + fMu * fLambda_d**2 * np.identity(nNetSize)), (-1, 1))
# Discretize optimal reset
optimal_reset_discrete = np.asarray(np.round(10*optimal_reset) / 10.0, dtype=np.float)

vfV_reset = 0.0

np.fill_diagonal(mfOmega_f, 0)

W_i = -mfOmega_f_discrete
tau_syn_r_fast = 1e-3

noise_std = 0.01

# Construct a spiking reservoir, setting the weights explicitly
dr_initial = NetworkDeneve.SpecifyNetwork(weights_fast = -mfOmega_f_discrete ,#* fTauN / 1e-3,
                                  weights_slow = mfOmega_s * fTauN,
                                  weights_in = mfGamma * fTauN,
                                  weights_out = mfGamma.T,
                                  tau_mem = fTauN,
                                  tau_syn_r_fast = tau_syn_r_fast,
                                  tau_syn_r_slow = fTauS,
                                  tau_syn_out = fTauS,
                                  v_thresh = vfT,
                                  v_rest = vfV_reset,
                                  v_reset = vfV_reset,
                                  noise_std = noise_std,
                                  dt = tDt,
                                  granularity = weight_granularity,
                                  margin = 0.0,
                                 )

# Construct a spiking reservoir, setting the weights explicitly
dr = NetworkDeneve.SpecifyNetwork(weights_fast = -mfOmega_f_discrete,
                                  weights_slow = mfOmega_s * fTauN,
                                  weights_in = mfGamma * fTauN,
                                  weights_out = mfGamma.T,
                                  tau_mem = fTauN,
                                  tau_syn_r_fast = tau_syn_r_fast,
                                  tau_syn_r_slow = fTauS,
                                  tau_syn_out = fTauS,
                                  v_thresh = vfT,
                                  v_rest = vfV_reset,
                                  v_reset = vfV_reset,
                                  noise_std = noise_std,
                                  dt = tDt,
                                  granularity = weight_granularity,
                                  margin = 0.0,
                                 )

omega_optimal = mfGamma.T @ mfGamma + fMu * fLambda_d**2 * np.identity(nNetSize)

np.fill_diagonal(omega_optimal, 0)
omega_optimal = -omega_optimal * fTauN / tau_syn_r_fast

dr_optimal = NetworkDeneve.SpecifyNetwork(weights_fast = omega_optimal,
                                            weights_slow = mfOmega_s * fTauN,
                                            weights_in = mfGamma * fTauN,
                                            weights_out = mfGamma.T,
                                            tau_mem = fTauN,
                                            tau_syn_r_fast = tau_syn_r_fast,
                                            tau_syn_r_slow = fTauS,
                                            tau_syn_out = fTauS,
                                            v_thresh = vfT,
                                            v_rest = optimal_reset,
                                            v_reset = optimal_reset,
                                            noise_std = noise_std,
                                            dt = tDt,
                                            granularity = weight_granularity,
                                            margin = 0.0,)

# Discretize optimal omega
max_weight = max(abs(np.min(omega_optimal)),np.max(omega_optimal))
max_weight_array = int(max_weight / weight_granularity)
bin_edges = weight_granularity*np.asarray(np.linspace(-max_weight_array,max_weight_array,2*max_weight_array +1))

omega_optimal_discrete = discretize(omega_optimal, bin_edges)

plot_matrices(omega_optimal, omega_optimal_discrete, title_A="Continuous", title_B="Discrete")


(iter_num,distance_to_optimal_weights,decoding_error) = dr.train(func=get_input_rate,
                                                                    omega_optimal=omega_optimal_discrete,
                                                                    num_iterations=51,
                                                                    fCommandAmp=fCommandAmp,
                                                                    nNumVariables=nNumVariables,
                                                                    sigma=sigma,
                                                                    tDt=tDt,
                                                                    tDuration=tDuration,
                                                                    verbose=False,
                                                                    validation_step=10)

print(decoding_error)


# First, create TSContinuous layer for the input and later rate conversion
rv_train = get_input_rate(duration=tDuration, dt=tDt, Amp=fCommandAmp,Nx=nNumVariables,sigma=sigma)
t = np.linspace(0, tDuration, int(tDuration / tDt))
ts_input_train = TSContinuous(t,rv_train.T) # Needs shape (#Samples,#Num_input_neurons)

dResp = dr.evolve(ts_input_train, tDuration)
dResp_optimal = dr_optimal.evolve(ts_input_train, tDuration)
dResp_initial = dr_initial.evolve(ts_input_train, tDuration)

# - Do the rescaling of the output signals
reconstructed_intial = dResp_initial['Output'].samples[:int(tDuration/tDt),:]
reconstructed_optimal = dResp_optimal['Output'].samples[:int(tDuration/tDt),:]
reconstructed_learned = dResp['Output'].samples[:int(tDuration/tDt),:]
target = ts_input_train.samples / fCommandAmp

scales_initial = np.linalg.pinv(reconstructed_intial.T @ reconstructed_intial) @ reconstructed_intial.T @ target
scales_optimal = np.linalg.pinv(reconstructed_optimal.T @ reconstructed_optimal) @ reconstructed_optimal.T @ target
scales_learned = np.linalg.pinv(reconstructed_learned.T @ reconstructed_learned) @ reconstructed_learned.T @ target

recon_initial = scales_initial @ reconstructed_intial.T
recon_optimal = scales_optimal @ reconstructed_optimal.T
recon_learned = scales_learned @ reconstructed_learned.T

dr.reset_all()


# - Write data in a dictionary and save to file
save_dict = {}
save_dict["t"] = t.tolist()
save_dict["recon_initial"] = recon_initial.tolist()
save_dict["recon_optimal"] = recon_optimal.tolist()
save_dict["recon_learned"] = recon_learned.tolist()
save_dict["target"] = target.tolist()
save_dict["initial_channels"] = dResp_initial['DeneveReservoir'].channels.tolist()
save_dict["initial_times"] = dResp_initial['DeneveReservoir'].times.tolist()
save_dict["optimal_channels"] = dResp_optimal['DeneveReservoir'].channels.tolist()
save_dict["optimal_times"] = dResp_optimal['DeneveReservoir'].times.tolist()
save_dict["learned_channels"] = dResp['DeneveReservoir'].channels.tolist()
save_dict["learned_times"] = dResp['DeneveReservoir'].times.tolist()
save_dict["W_initial"] = W_i.tolist()
save_dict["W_optimal"] = omega_optimal.tolist()
save_dict["W_learned"] = dr.lyrRes.weights.tolist()

with open("data.json", "w") as f:
    json.dump(save_dict, f)





