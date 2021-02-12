import numpy as np

import matplotlib
matplotlib.rc('font', family='Sans-Serif')
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

fig = plt.figure(figsize=(5.49,4.36))

t_start = 0.5
t_stop = 0.8
t_sub = t[(t > t_start) & (t < t_stop)]
target_sub = target[(t > t_start) & (t < t_stop)]
recon_initial_sub = recon_initial[:,(t > t_start) & (t < t_stop)]
recon_optimal_sub = recon_optimal[:,(t > t_start) & (t < t_stop)]
recon_learned_sub = recon_learned[:,(t > t_start) & (t < t_stop)]

ax0 = fig.add_subplot(331)
top_lim = [-0.1,0.1]
ax0.set_title(r"\textbf{A}")
l1 = ax0.plot(t_sub, recon_initial_sub.T, color="C2")
l2 = ax0.plot(t_sub, target_sub, color="C4")
ax0.set_xticks([t_start,t_stop])
ax0.plot([0.5,0.55], [-0.08,-0.08], color="k", linewidth=0.5)
ax0.text(x=0.5, y=-0.11, s="50 ms")
ax0.set_ylim(top_lim)
lines = [l1[0],l2[0]]
ax0.legend(lines,[r"$\hat{x}$", r"$x$"], frameon=False, loc=2, prop={'size': 7})

ax1 = fig.add_subplot(332)
ax1.set_title(r"\textbf{B}")
ax1.plot(t_sub,recon_optimal_sub.T, color="C2")
ax1.plot(t_sub,target_sub, color="C4")
ax1.set_xticks([t_start,t_stop])
ax1.set_ylim(top_lim)

ax2 = fig.add_subplot(333)
ax2.set_title(r"\textbf{C}")
ax2.plot(t_sub,recon_learned_sub.T, color="C2")
ax2.plot(t_sub,target_sub, color="C4")
ax2.set_xticks([t_start,t_stop])
ax2.set_ylim(top_lim)

axes_top = [ax1,ax2]
for ax_tmp in axes_top:
    ax_tmp.spines["top"].set_visible(False)
    ax_tmp.spines["right"].set_visible(False)
    ax_tmp.spines["left"].set_visible(False)
    ax_tmp.spines["bottom"].set_visible(False)
    ax_tmp.set_xticks([])
    ax_tmp.set_yticks([])

ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)
ax0.spines["bottom"].set_visible(False)
ax0.set_xticks([])

ax3 = fig.add_subplot(334)
nNetSize = W_initial.shape[0]
tDuration = t[-1]
channels = initial_channels
times = initial_times
channels_id_initial = channels[channels >= 0]
times_initial = times[channels >= 0]
ax3.scatter(times_initial, 1+channels_id_initial, color="k")
ax3.set_yticks([1,nNetSize+1])
ax3.set_xticks([0,tDuration])
ax3.set_ylabel(r"Neuron ID")

ax3.plot([0.0,0.2], [-3,-3], color="k", linewidth=0.5)
ax3.text(x=0.01, y=-6, s="200 ms")


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

axes_mid = [ax4,ax5]

for ax_tmp in axes_mid:
    ax_tmp.spines["top"].set_visible(False)
    ax_tmp.spines["right"].set_visible(False)
    ax_tmp.spines["left"].set_visible(False)
    ax_tmp.spines["bottom"].set_visible(False)
    ax_tmp.set_xticks([])
    ax_tmp.set_yticks([])

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_visible(False)
ax3.set_xticks([])
ax3.set_yticks([1,20])

ax6 = fig.add_subplot(337)
im = plt.matshow(W_initial / 100, fignum=False, cmap='RdBu')
#plt.axis('off')
plt.xticks([])
#plt.yticks([])
plt.ylabel(r"Neuron ID")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

plt.colorbar(ticks=[-0.9,0,0.9])

ax7 = fig.add_subplot(338)
im = plt.matshow(W_optimal / 100, fignum=False, cmap='RdBu')
plt.axis('off')
#plt.colorbar(ticks=[-0.9,0,0.9])

ax8 = fig.add_subplot(339)
im = plt.matshow(W_learned / 100, fignum=False, cmap='RdBu')
plt.tight_layout()
plt.axis('off')
#plt.colorbar(ticks=[-0.9,0,0.9])

# - Save and plot
plt.savefig("figure1.png", dpi=1200)
plt.show()