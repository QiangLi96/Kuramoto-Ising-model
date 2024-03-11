import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import time

time_start = time.time()

N = 2

phis = np.load('KM_sync_vs_params_phis_1.npy')
phis /= math.pi
phis = np.around(phis,4)
# phis = np.mod(phis,2)
Es = np.load('KM_sync_vs_params_Es_1.npy')

Kc_arr = np.around(np.arange(0.1,1.0+0.1/2,0.1),1)
Ks_arr = np.around(np.arange(-0.5,1.0+0.1/2,0.1),1)

Kc_arr = Kc_arr[::-1]

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

xtick_step = 5
ytick_step = 5
# for i in range(N):
#     fig, ax = plt.subplots()
#     phis_tmp = phis[i]
#     phis_tmp = phis_tmp[::-1]
#     im = ax.imshow(phis_tmp,vmin=0.5,vmax=2.0)
#     # Show all ticks and label them with the respective list entries
#     ax.set_xticks(np.arange(len(Ks_arr))[::xtick_step], labels=Ks_arr[::xtick_step],fontsize=15)
#     ax.set_yticks(np.arange(len(Kc_arr))[::ytick_step], labels=Kc_arr[::ytick_step],fontsize=15)
#     ax.set_xlabel(r'$K_s$',fontsize=15)
#     ax.set_ylabel(r'$K$',fontsize=15)
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=0,rotation_mode="anchor")
#     # Create colorbar
#     cbar = ax.figure.colorbar(im, ax=ax,)
#     cbarlabel = r'$\phi_i$ ($\pi$)'
#     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=15)
#     # set fontsize of the colorbar ticks
#     cbar.ax.tick_params(labelsize=15)
#     # Loop over data dimensions and create text annotations.
#     ax.set_title(r"$N=2, t=50, \Delta t=0.01$",fontsize=15)
#     fig.tight_layout()
#     plt.savefig('max_cut_OIM_test/Figs/KM_sync_states_vs_params/sates_vs_params_phi'+str(i+1)+'_1.pdf')



# fig, ax = plt.subplots()
# Es = Es[::-1]
# im = ax.imshow(Es)
# # Show all ticks and label them with the respective list entries
# ax.set_xticks(np.arange(len(Ks_arr))[::xtick_step], labels=Ks_arr[::xtick_step],fontsize=15)
# ax.set_yticks(np.arange(len(Kc_arr))[::ytick_step], labels=Kc_arr[::ytick_step],fontsize=15)
# ax.set_xlabel(r'$K_s$',fontsize=15)
# ax.set_ylabel(r'$K$',fontsize=15)
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=0,rotation_mode="anchor")
# # Create colorbar
# cbar = ax.figure.colorbar(im, ax=ax,)
# cbarlabel = r'Energy'
# cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=15)
# # set fontsize of the colorbar ticks
# cbar.ax.tick_params(labelsize=15)
# # Loop over data dimensions and create text annotations.
# ax.set_title(r"$N=2, t=50, \Delta t=0.01$",fontsize=15)
# fig.tight_layout()
# plt.savefig('max_cut_OIM_test/Figs/KM_sync_states_vs_params/energy_vs_params_phi'+str(i+1)+'_1.pdf')


fig, ax = plt.subplots(2,1)
for i in range(N):
    phis_tmp = phis[i]
    phis_tmp = phis_tmp[::-1]
    im = ax[i].imshow(phis_tmp,vmin=0.5,vmax=2.0)
    ax[i].set_yticks(np.arange(len(Kc_arr))[::ytick_step], labels=Kc_arr[::ytick_step],fontsize=15)
ax[i].set_ylabel(r'$K$',fontsize=15)
# Show all ticks and label them with the respective list entries
ax[i].set_xticks(np.arange(len(Ks_arr))[::xtick_step], labels=Ks_arr[::xtick_step],fontsize=15)
ax[i].set_yticks(np.arange(len(Kc_arr))[::ytick_step], labels=Kc_arr[::ytick_step],fontsize=15)
ax[i].set_xlabel(r'$K_s$',fontsize=15)
ax[i].set_ylabel(r'$K$',fontsize=15)
# Create colorbar
cbar = ax[i].figure.colorbar(im, ax=ax,)
cbarlabel = r'$\phi_i$ ($\pi$)'
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=15)
# set fontsize of the colorbar ticks
cbar.ax.tick_params(labelsize=15)

# plt.savefig('max_cut_OIM_test/Figs/KM_sync_states_vs_params/sates_vs_params_phi'+str(i+1)+'_1.pdf')



plt.show()

print()