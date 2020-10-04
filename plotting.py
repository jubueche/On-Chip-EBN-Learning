import numpy as np

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt

from Utils import plot_matrices

import ujson as json

with open("data.json", 'r') as f:
    load_dict = json.load(f)


t = np.asarray(load_dict["t"])
recon_initial = np.asarray(load_dict["recon_initial"])
recon_optimal = np.asarray(load_dict["recon_optimal"])
recon_learned = np.asarray(load_dict["recon_learned"])
target = np.asarray(load_dict["target"])
initial_channels = np.asarray(load_dict["initial_channels"])
initial_times = np.asarray(load_dict["initial_times"])
optimal_channels = np.asarray(load_dict["optimal_channels"])
optimal_times = np.asarray(load_dict["optimal_times"])
learned_cahnnels = np.asarray(load_dict["learned_channels"])
learned_times = np.asarray(load_dict["learned_times"])
W_initial = np.asarray(load_dict["W_initial"])
W_optimal = np.asarray(load_dict["W_optimal"])
W_learned = np.asarray(load_dict["W_learned"])

fig = plt.figure(figsize=(5,4.73))

t_start = 0.5
t_stop = 0.8
t_sub = t[(t > t_start) & (t < t_stop)]
target_sub = target[(t > t_start) & (t < t_stop)]
recon_initial_sub = recon_initial[:,(t > t_start) & (t < t_stop)]
recon_optimal_sub = recon_optimal[:,(t > t_start) & (t < t_stop)]
recon_learned_sub = recon_learned[:,(t > t_start) & (t < t_stop)]

ax0 = fig.add_subplot(331)
ax0.set_title(r"\textbf{A) Pre-learning}")
ax0.plot(t_sub, recon_initial_sub.T, color="C2")
ax0.plot(t_sub, target_sub, color="C4")
ax0.set_xticks([t_start,t_stop])
ax0.set_ylabel(r"$x,\hat{x}$")
ax0.set_yticks([],[])

ax1 = fig.add_subplot(332)
ax1.set_title(r"\textbf{B) Optimal}")
ax1.plot(t_sub,recon_optimal_sub.T, color="C2")
ax1.plot(t_sub,target_sub, color="C4")
ax1.set_xticks([t_start,t_stop])
ax1.set_yticks([],[])

ax2 = fig.add_subplot(333)
ax2.set_title(r"\textbf{C) Post-learning}")
ax2.plot(t_sub,recon_learned_sub.T, color="C2")
ax2.plot(t_sub,target_sub, color="C4")
ax2.set_xticks([t_start,t_stop])
ax2.set_yticks([],[])

ax3 = fig.add_subplot(334)
nNetSize = W_initial.shape[0]
tDuration = t[-1]
channels = initial_channels
times = initial_times
channels_id_initial = channels[channels >= 0]
times_initial = times[channels >= 0]
ax3.scatter(times_initial, channels_id_initial, color="k")
ax3.set_yticks([0,nNetSize])
ax3.set_xticks([0,tDuration])
ax3.set_ylabel(r"Neuron ID")

ax4 = fig.add_subplot(335)
channels = optimal_channels
times = optimal_times
channels_id_optimal = channels[channels >= 0]
times_optimal = times[channels >= 0]
ax4.scatter(times_optimal, channels_id_optimal, color="k")
ax4.set_yticks([0,nNetSize])
ax4.set_xticks([0,tDuration])

ax5 = fig.add_subplot(336)
channels = learned_cahnnels
times = learned_times
channels_id_learned = channels[channels >= 0]
times_learned = times[channels >= 0]
ax5.scatter(times_learned, channels_id_learned, color="k")
ax5.set_yticks([0,nNetSize])
ax5.set_xticks([0,tDuration])

ax6 = fig.add_subplot(337)
ax6.set_ylabel(r"Initial $\Omega$")
im = plt.matshow(W_initial, fignum=False, cmap='RdBu')
ax6.set_xticks([], [])

ax7 = fig.add_subplot(338)
ax7.set_ylabel(r"Optimal $\Omega$")
im = plt.matshow(W_optimal, fignum=False, cmap='RdBu')
ax7.set_xticks([], [])

ax8 = fig.add_subplot(339)
ax8.set_ylabel(r"Learned $\Omega$")
im = plt.matshow(W_learned, fignum=False, cmap='RdBu')
ax8.set_xticks([], [])
plt.tight_layout()

plt.savefig("figure1.png", dpi=1200)

plt.show()


plot_matrices(W_i, omega_optimal_discrete, dr.lyrRes.weights, title_A=r"Initial $\Omega$", title_B=r"Optimal $\Omega$",title_C=r"Learned $\Omega$")