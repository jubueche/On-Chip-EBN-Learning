import numpy as np 
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import json
import os
import csv
from scipy.interpolate import interp1d
from copy import deepcopy
import matplotlib.patches as patches

def compute_on_off_set(times, values, threshold = 0.01, plot = True):
    t_onset = []
    t_offset = []
    is_plateau = False
    for (t,val) in zip(times,values):
        if(not is_plateau and val > threshold):
            t_onset.append(t)
            is_plateau = True
        elif(is_plateau and val < threshold):
            t_offset.append(t)
            is_plateau = False
    if(plot):
        plt.plot(times, values)
        for (t_on,t_off) in zip(t_onset,t_offset):
            plt.axvline(x=t_on, color='g')
            plt.axvline(x=t_off, color='r')
        plt.show()

    return (t_onset,t_offset)

def filter_bit_data(bit_time, bit_data):
    f_bit_data = interp1d(bit_time, bit_data, kind="nearest")
    t = np.linspace(min(bit_time),max(bit_time), 10000)
    y = f_bit_data(t)
    return (t,y)

######################

base_path = "/home/julian/Documents/On-Chip-EBN-Learning/Simulations/"
sim_prefix = "Sim2_"
data_path = os.path.join(base_path, "Simulation 2/")
file_name = "sim2_plot.png"

# - Get the data
# - BitValue
bit_value_path = os.path.join(data_path, sim_prefix+"BitValue.csv")
bit_values = []
with open(bit_value_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx,row in enumerate(csv_reader):
        if(idx > 0): # - Skip header
            bit_values.append(np.asarray(row, dtype=float))
            
bit_values = np.asarray(bit_values)
t_bit_values = bit_values[:,0]
bit_values = bit_values[:,1]
t_bit_values, bit_values = filter_bit_data(t_bit_values, bit_values)

# - I_mem
I_mem_path = os.path.join(data_path, sim_prefix+"I_mem.csv")
I_mem = []
with open(I_mem_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx,row in enumerate(csv_reader):
        if(idx > 0): # - Skip header
            I_mem.append(np.asarray(row, dtype=float))

I_mem = np.asarray(I_mem)
t_I_mem = I_mem[:,0]
I_mem = I_mem[:,1] * 1e9 # - Scale it to nano Amp

# - Input
input_path = os.path.join(data_path, sim_prefix+"INPUT.csv")
INPUT = []
with open(input_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx,row in enumerate(csv_reader):
        if(idx > 0): # - Skip header
            INPUT.append(np.asarray(row, dtype=float))

INPUT = np.asarray(INPUT)
t_INPUT = INPUT[:,0]
INPUT = INPUT[:,1]

# - INC
INC_path = os.path.join(data_path, sim_prefix+"INC.csv")
INC = []
with open(INC_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx,row in enumerate(csv_reader):
        if(idx > 0): # - Skip header
            INC.append(np.asarray(row, dtype=float))

INC = np.asarray(INC)
t_INC = INC[:,0]
INC = INC[:,1]

# - SL
SL_path = os.path.join(data_path, sim_prefix+"SL.csv")
SL = []
with open(SL_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for idx,row in enumerate(csv_reader):
        if(idx > 0): # - Skip header
            SL.append(np.asarray(row, dtype=float))

SL = np.asarray(SL)
t_SL = SL[:,0]
SL = SL[:,1]

# SL_on_off = compute_on_off_set(t_SL, SL, threshold=0.2, plot=False)
SL_onset, SL_offset = compute_on_off_set(t_SL, SL, threshold=0.2, plot=False)
# INC_on_off = compute_on_off_set(t_INC, INC, threshold=0.2, plot=False)
INC_onset, INC_offset = compute_on_off_set(t_INC, INC, threshold=0.2, plot=False)
# - Compute the on and offset of DEC
# - Now interpolate SL and check for each time step in t_DEC_prim if SL is above threshold
f_SL = interp1d(t_SL, SL, kind='nearest')
DEC = np.zeros((len(t_INC),))
for idx,t in enumerate(t_INC):
    if(f_SL(t) < 0.2 and INC[idx] < 0.2):
        DEC[idx] = 1.0

# DEC_on_off = compute_on_off_set(t_INC, DEC, threshold=0.2, plot=False)
DEC_onset, DEC_offset = compute_on_off_set(t_INC, DEC, threshold=0.2, plot=False) 
spike_onset, spike_offset = compute_on_off_set(t_INPUT, INPUT, threshold=0.2, plot=False)
spike_onset, spike_offset = zip(*(zip(spike_onset,spike_offset)))

# - Interpolate I_mem
f_I_mem = interp1d(t_I_mem, I_mem, kind="nearest")

# - Find out I_Omega_n,k by comparing the current before and after the onset
I_mem_at_onset = f_I_mem(spike_onset)
I_mem_at_offset = f_I_mem(spike_offset)

I_Omega_n_k = I_mem_at_offset - I_mem_at_onset
t_Omega_n_k = spike_onset
f_I_Omega_n_k = interp1d(t_Omega_n_k, I_Omega_n_k, kind='nearest')

# - Plot I_mem and I_Omega_n,k
# plt.plot(t_I_mem, I_mem)
# plt.step(t_Omega_n_k, I_Omega_n_k, where = 'post', label = 'flat_first')
# plt.show()

# - Calculate the decision boundaries
I_rest = 9 # 9 nA
I_SL = 0.3 # nA
lower_bound = I_rest - I_Omega_n_k / 2 - I_SL
upper_bound = I_rest - I_Omega_n_k / 2 + I_SL

fig = plt.figure(figsize=(1.6*3.49,1.6*1.97),constrained_layout=True)
gs = fig.add_gridspec(2, 1)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])

ax1_twin = ax1.twinx()
ax1_twin.plot(t_bit_values, bit_values, color='r')
ax1_twin.set_ylabel("Bit value", color='r')
ax1.step(t_Omega_n_k,lower_bound, where = 'post', color="C2", linestyle='--', label=r"$I_{\textnormal{lower}}$")
ax1.step(t_Omega_n_k,upper_bound, where = 'post', color="C4", linestyle='--', label=r"$I_{\textnormal{upper}}$")
ax1.plot(t_I_mem, I_mem, label=r"$I_{\textnormal{mem}}$")
ax1.axhline(y=I_rest, color="C5", linestyle='--', label=r"$I_{\textnormal{rest}}$")

for (on,off) in zip(SL_onset,SL_offset):
    ax1.axvspan(on, off, facecolor='y', alpha=0.1)
for (on,off) in zip(INC_onset,INC_offset):
    ax1.axvspan(on, off, facecolor='g', alpha=0.1)
for (on,off) in zip(DEC_onset,DEC_offset):
    ax1.axvspan(on, off, facecolor='r', alpha=0.1)
if(len(SL_onset) > len(SL_offset)):
    ax1.axvspan(SL_onset[-1],max(t_I_mem), facecolor='y', alpha=0.1)
if(len(INC_onset) > len(INC_offset)):
    ax1.axvspan(INC_onset[-1],max(t_I_mem), facecolor='g', alpha=0.1)
if(len(DEC_onset) > len(DEC_offset)):
    ax1.axvspan(DEC_onset[-1],max(t_I_mem), facecolor='r', alpha=0.1)
ax1.set_ylabel(r"$I_{\textnormal{mem}}$ \textbf{[nA]}")
ax1.legend(loc=1, prop={'size': 5})
ax1.yaxis.set_ticks_position('none')
ax1_twin.yaxis.set_ticks_position('none') 
ax1.set_yticks([3,8,13])
ax1_twin.set_yticks([5, -10, -25])

axes = [ax1,ax1_twin]
for ax_tmp in axes:
    ax_tmp.spines["top"].set_visible(False)
    ax_tmp.spines["right"].set_visible(False)
    ax_tmp.spines["left"].set_visible(False)
    ax_tmp.spines["bottom"].set_visible(False)
ax1.set_xticks([])

ax1_twin.plot([0.01,0.06], [-27,-27], color="k", linewidth=0.5)
ax1_twin.text(x=0.01, y=-31, s="50 ms")


t_start = 0.2
t_stop = 0.3
ax2.plot(t_I_mem, I_mem)
ax2_twin = ax2.twinx()
ax2_twin.plot(t_bit_values, bit_values, color='r')
ax2_twin.set_ylim([-35,0])
for (on,off) in zip(SL_onset,SL_offset):
    ax2.axvspan(on, off, facecolor='y', alpha=0.1)
for (on,off) in zip(INC_onset,INC_offset):
    ax2.axvspan(on, off, facecolor='g', alpha=0.1)
for (on,off) in zip(DEC_onset,DEC_offset):
    ax2.axvspan(on, off, facecolor='r', alpha=0.1)
ax2.step(t_Omega_n_k,lower_bound, where = 'post', color="C2", linestyle='--')
ax2.step(t_Omega_n_k,upper_bound, where = 'post', color="C4", linestyle='--')
ax2.set_xlim([t_start,t_stop])
ax2.set_ylim([7,10])
ax2.axhline(y=I_rest, color="C5", linestyle='--')

axes = [ax2,ax2_twin]
for ax_tmp in axes:
    ax_tmp.spines["top"].set_visible(False)
    ax_tmp.spines["right"].set_visible(False)
    ax_tmp.spines["left"].set_visible(False)
    ax_tmp.spines["bottom"].set_visible(False)

ax2_twin.set_yticks([])
ax2_twin.yaxis.set_ticks_position('none') 
ax2_twin.set_xticks([])

ax2_twin.plot([0.202,0.212], [-31,-31], color="k", linewidth=0.5)
ax2_twin.text(x=0.202, y=-34, s="10 ms")
ax2.set_yticks([])

# - Draw rectangle in first plot
# Create a Rectangle patch
rect = patches.Rectangle((t_start,7),0.1,3,linewidth=0.5,edgecolor='grey',facecolor='none')
# Add the patch to the Axes
ax1.add_patch(rect)

plt.savefig(file_name, dpi=1200)
plt.show()