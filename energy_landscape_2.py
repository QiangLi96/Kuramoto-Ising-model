import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import math
import time

time_start = time.time()





class Plot():
    def __init__(self,fig_path:str) -> None:
        self.fig_path = fig_path
        
        
    def plot_2dArr(self,x_arr,xtick:str,ytick:str,cmap:str='Reds',x_label:str='',y_label:str='',title:str='',new_fig:bool=True,center:bool=False,xtick_top:bool=False,cbarlabel:str='',alpha:float=0.6,save_fig:bool=False,file_name:str=''):
        df = pd.DataFrame(data=x_arr,columns=xtick,index=ytick)
        if new_fig:
            plt.figure(figsize=[6.4*1.0, 4.8*1.0])
        if center:
            ax1 = sns.heatmap(df, cbar=True, linewidths=0, square=True, cmap=cmap,center=0,alpha=alpha)
        else:
            ax1 = sns.heatmap(df, cbar=True, linewidths=0, square=True, cmap=cmap,alpha=alpha)
        if len(title)>0:
            ax1.set_title(r''+title,fontsize=20)
        if xtick_top:
            # show xticks on the top
            ax1.xaxis.tick_top()
        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbarlabel = r''+cbarlabel
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=20)
        # to solve xticks and yticks too dense
        plt.xticks(np.arange(0, df.shape[1], 10)+0.5, xtick[::10],fontsize=20)
        plt.yticks(np.arange(0, df.shape[0], 10)+0.5, ytick[::10],fontsize=20)

        # # Rotate the tick labels and set their alignment.
        if xtick_top:
            plt.setp(ax1.get_xticklabels(), rotation=0, rotation_mode="anchor")
        #     plt.setp(ax1.get_yticklabels(), rotation=0, rotation_mode="anchor")
        else:
            plt.setp(ax1.get_xticklabels(), rotation=0, rotation_mode="anchor")
        #     plt.setp(ax1.get_yticklabels(), rotation=0, rotation_mode="anchor")
        plt.xlabel(r''+x_label,fontsize=20)
        plt.ylabel(r''+y_label,fontsize=20)

        if save_fig:
            plt.savefig(self.fig_path+file_name)

    def plot_2dArrSubplot(self,x_list:list,xtick:str,ytick:str,sub_list:list,cmap:str='Reds',x_label:str='',y_label:str='',title:str='',center:bool=False,xtick_top:bool=False,cbarlabel:str='',alpha:float=0.6,save_fig:bool=False,file_name:str=''):
        for i,x_arr in enumerate(x_list):
            df = pd.DataFrame(data=x_arr,columns=xtick,index=ytick)
            sub_params = sub_list[i]
            plt.subplot(sub_params[0],sub_params[1],sub_params[2])
            if center:
                if i==len(x_list)-1:
                    ax1 = sns.heatmap(df, cbar=True, linewidths=0, square=True, cmap=cmap,center=0,alpha=alpha)
                    cbar = ax1.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=20)
                    cbarlabel = r''+cbarlabel
                    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=20)
                else: 
                    ax1 = sns.heatmap(df, cbar=False, linewidths=0, square=True, cmap=cmap,center=0,alpha=alpha)
            else:
                if i==len(x_list)-1:
                    ax1 = sns.heatmap(df, cbar=True, linewidths=0, square=True, cmap=cmap,alpha=alpha)
                    cbar = ax1.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=20)
                    cbarlabel = r''+cbarlabel
                    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=20)
                else:
                    ax1 = sns.heatmap(df, cbar=False, linewidths=0, square=True, cmap=cmap,alpha=alpha)

            if xtick_top:
                # show xticks on the top
                ax1.xaxis.tick_top()
            # to solve xticks and yticks too dense
            plt.xticks(np.arange(0, df.shape[1], 10)+0.5, xtick[::10],fontsize=20)
            plt.yticks(np.arange(0, df.shape[0], 10)+0.5, ytick[::10],fontsize=20)

            # # Rotate the tick labels and set their alignment.
            if xtick_top:
                plt.setp(ax1.get_xticklabels(), rotation=0, rotation_mode="anchor")
            #     plt.setp(ax1.get_yticklabels(), rotation=0, rotation_mode="anchor")
            else:
                plt.setp(ax1.get_xticklabels(), rotation=0, rotation_mode="anchor")
            #     plt.setp(ax1.get_yticklabels(), rotation=0, rotation_mode="anchor")
            plt.xlabel(r''+x_label,fontsize=20)
            plt.ylabel(r''+y_label,fontsize=20)
        if len(title)>0:
            ax1.set_title(r''+title,fontsize=20)
        if save_fig:
            plt.savefig(self.fig_path+file_name)


    def plot_2darrImshow(self,x_arr,xtick:str,ytick:str,cmap:str='Reds',x_label:str='',y_label:str='',title:str='',xtick_top:bool=False,cbarlabel:str='',cbar_ext:bool=False,save_fig:bool=False,file_name:str=''):
        fig, ax = plt.subplots()
        im = ax.imshow(x_arr,cmap=cmap)
        if len(title)>0:
            ax.set_title(r''+title,fontsize=20)
        if xtick_top:
            # show xticks on the top
            ax.xaxis.tick_top()
        # Show all ticks and label them with the respective list entries
        if len(xtick)>10:
            xtick = ['']*len(xtick)
            for i,xt in enumerate(xtick):
                if i%3 == 0:
                    xtick[i] = xt
        if len(ytick)>10:
            ytick = ['']*len(ytick)
            for i,yt in enumerate(ytick):
                if i%3 == 0:
                    ytick[i] = yt

        ax.set_xticks(np.arange(len(xtick)), labels=xtick,fontsize=20)
        ax.set_yticks(np.arange(len(ytick)), labels=ytick,fontsize=20)
        ax.set_xlabel(r''+x_label,fontsize=20)
        ax.set_ylabel(r''+y_label,fontsize=20)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-45, rotation_mode="anchor")
        # Create colorbar
        if cbar_ext:
            # Add colorbar, make sure to specify tick locations to match desired ticklabels
            cbar = fig.colorbar(im,
                                ticks=[-1, 0, 1],
                                format=ticker.FixedFormatter(['< -1', '0', '> 1']),
                                extend='both',
                            )

            labels = cbar.ax.get_yticklabels()
            labels[0].set_verticalalignment('top')
            labels[-1].set_verticalalignment('bottom')
        else:
            cbar = fig.colorbar()
        cbarlabel = r''+cbarlabel
        labels = cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=20)
        # set fontsize of the colorbar ticks
        cbar.ax.tick_params(labelsize=20)
        fig.tight_layout()

        if save_fig:
            plt.savefig(self.fig_path+file_name)



x = np.arange(0,2*math.pi+0.05/2,0.05*math.pi)
y = np.arange(0,2*math.pi+0.05/2,0.05*math.pi)
X,Y = np.meshgrid(x,y)
# energy function of two connected nodes
def cal_energy(J_12,J_21,X=X,Y=Y):
    E1 = -J_12*np.cos(X-Y)-J_21*np.cos(Y-X)
    E2 = -J_12*np.cos(X-Y)-J_21*np.cos(Y-X)-np.cos(2*X)-np.cos(2*Y)
    return E1, E2

E_dict = {}
J_12 = 1
J_21 = 1
E_dict['KM_Aij'], E_dict['KM_Aij_sync'] = cal_energy(J_12=J_12,J_21=J_21)
J_12 = -1
J_21 = -1
E_dict['KM'], E_dict['KM_sync'] = cal_energy(J_12=J_12,J_21=J_21)

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

### plot energy heatmap
fig_path = 'F:/Latex_work/work_4_2020_5_20/conference_Sep/presentation/figs/'
plot_fig = Plot(fig_path=fig_path)
save_fig = False

# plot_fig.plot_2dArrSubplot(x_list=[E_dict['KM_Aij'],E_dict['KM']],sub_list=[(2,1,1),(2,2,2)],xtick=np.around(x/math.pi,decimals=4),ytick=np.around(y/math.pi,decimals=4),center=True,cmap='bwr',x_label='$\\phi_1 (\pi)$',y_label='$\\phi_2 (\pi)$',cbarlabel='Energy',alpha=0.8,save_fig=save_fig,file_name='')
# plot_fig.plot_2dArrSubplot(x_arr=E_dict['KM_Aij_sync'],sub_params=(1,2,2),xtick=np.around(x/math.pi,decimals=4),ytick=np.around(y/math.pi,decimals=4),center=True,cmap='bwr',x_label='$\\phi_1 (\pi)$',y_label='$\\phi_2 (\pi)$',cbarlabel='Energy',alpha=0.8,save_fig=save_fig,file_name='')


# for J_ij=-A_ij
figname_KM = 'energy_KM_heatmap_1.pdf'
figname_KMsync = 'energy_KM_with_sync_heatmap_1.pdf'
E1,E2 = E_dict['KM'],E_dict['KM_sync']
# for J_ij=A_ij
# figname_KM = 'energy_KM_Aij_heatmap_1.pdf'
# figname_KMsync = 'energy_KM_Aij_with_sync_heatmap_1.pdf'
# E1,E2 = E_dict['KM_Aij'], E_dict['KM_Aij_sync']
plot_fig.plot_2dArr(x_arr=E1,xtick=np.around(x/math.pi,decimals=4),ytick=np.around(y/math.pi,decimals=4),center=True,cmap='bwr',x_label='$\\phi_1 (\pi)$',y_label='$\\phi_2 (\pi)$',cbarlabel='Energy',alpha=0.8,save_fig=save_fig,file_name=figname_KMsync)
plot_fig.plot_2dArr(x_arr=E2,xtick=np.around(x/math.pi,decimals=4),ytick=np.around(y/math.pi,decimals=4),center=True,cmap='bwr',x_label='$\\phi_1 (\pi)$',y_label='$\\phi_2 (\pi)$',cbarlabel='Energy',alpha=0.8,save_fig=save_fig,file_name=figname_KM)




plt.show()

print()
# plt.figure(figsize=[6.4 * 1.2, 4.8 * 1.2])
# # plt.plot(t_arr[int(Total_steps/2):], phi_mat[:,int(Total_steps/2):].T, alpha=0.4)
# # plt.plot(t_arr.cpu()[-5000:], (phi_mat.cpu()[:,-5000:]/math.pi).T, alpha=0.6)
# plt.plot(t_arr.cpu()[:], (phi_mat.cpu()[:,:]/math.pi).T, alpha=0.6)
# plt.title(r'$K_s$='+str(Ks)+r', $\sigma$='+str(sigma),fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel(r"time (s)", fontsize=20)
# plt.ylabel(r"$\phi_i$ ($\pi$)", fontsize=20)
# plt.show()
