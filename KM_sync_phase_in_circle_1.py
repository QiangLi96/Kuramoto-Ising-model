import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import time

time_start = time.time()

class Simulation():
    def __init__(self,t_start,t_end,step_size:float=0.01) -> None:
        self.t_arr = np.arange(t_start,t_end,step_size)
        self.step_size = step_size
        self.Total_step = len(self.t_arr)
        pass

    def fun_dotx(self,phi_tmp,J_tmp,omega,Kc_tmp,Ks_tmp,v:int=2):
        k_temp = omega-Kc_tmp*(np.multiply(J_tmp,np.sin(phi_tmp[:,None]-phi_tmp[None,:]))).sum(axis=1)-Ks_tmp*np.sin(v*phi_tmp)
        return k_temp

    def RK4(self,phis_tmp,J_mat,omega,Kc,Ks):
        for i in range(0,self.Total_step-1,1):
            k1 = self.step_size*self.fun_dotx(phis_tmp[:,i],J_mat,omega,Kc, Ks)
            k2 = self.step_size*self.fun_dotx(phis_tmp[:,i]+k1/2.0,J_mat,omega,Kc, Ks)
            k3 = self.step_size * self.fun_dotx(phis_tmp[:,i] + k2 / 2.0,J_mat,omega,Kc, Ks)
            k4 = self.step_size * self.fun_dotx(phis_tmp[:,i]+k3,J_mat,omega,Kc, Ks)
            phis_tmp[:,i+1] = phis_tmp[:,i]+(k1+2.0*k2+2.0*k3+k4)/6.0
        
        return phis_tmp

    def simulate(self,N,J_mat,omega,Kc,Ks):
        phi_mat = np.zeros((N,self.Total_step))
        np.random.seed(0)
        phi_mat[:,0] = np.random.uniform(0,1,N)*math.pi
        phi_mat = self.RK4(phis_tmp=phi_mat,J_mat=J_mat,omega=omega,Kc=Kc,Ks=Ks)

        return phi_mat
    
    def get_orderParams(self,phi_mat,N:int):
        """
        Calculate order parameters.
        """
        r = np.sqrt(np.sum(np.cos(phi_mat),axis=0)**2+np.sum(np.sin(phi_mat),axis=0)**2)/N

        return r
    
    def fun_energy(self,phi_tmp,J_tmp,Kc_tmp,Ks_tmp,v=2):
        """
        energy expression.
        """
        E_tmp = -Kc_tmp*np.sum(np.multiply(J_tmp,np.cos(phi_tmp[:,None]-phi_tmp[None,:])))-Ks_tmp*np.sum(np.cos(v*phi_tmp))
        return E_tmp
    def cal_energy(self,phi_mat,J_tmp,Kc_tmp,Ks_tmp,v=2):
        """
        Calculate energy.
        """
        E_tmp = []
        for phi in phi_mat.T:
            E_tmp.append(self.fun_energy(phi_tmp=phi,J_tmp=J_tmp,Kc_tmp=Kc_tmp,Ks_tmp=Ks_tmp,v=v))
        
        return E_tmp



class Plot():
    def __init__(self) -> None:
        pass

    def plot_2Dphases(self,t_arr,phi_mat,color:str='#1f77b4',title:str='',alpha:float=0.8,save_fig:bool=False,save_path:str='',file_name:str=''):
        plt.figure()
        plt.plot(t_arr,phi_mat.T/math.pi,color=color,linewidth=2,alpha=alpha)
        if len(title)>0:
            plt.title(r''+title,fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r'$t$',fontsize=20)
        plt.ylabel(r'$\phi_i (\pi)$',fontsize=20)
        # plt.legend(loc='best',fontsize=20)
        if save_fig:
            plt.savefig(save_path+file_name)

    def plot_phisIncircle(self,phis,mark:str='o',markersize:int=10,color:str='#1f77b4',title:str='',alpha:float=0.8,color_cir:str='black',alpha_cir:float=0.5,save_fig:bool=False,save_path:str='',file_name:str=''):
        x_arr = np.cos(phis)
        y_arr = np.sin(phis)
        plt.figure(figsize=[3,3])
        plt.plot(x_arr,y_arr,mark,markersize=markersize,color=color,alpha=alpha)
        theta = np.arange(0,math.pi*2+0.01/2,0.01)
        # plot circle
        x_cir = np.cos(theta)
        y_cir = np.sin(theta)
        plt.plot(x_cir,y_cir,'-',color=color_cir,linewidth=2,alpha=alpha_cir)
        plt.plot([0],[0],'.',color=color_cir,linewidth=2,alpha=alpha_cir)
        if len(title)>0:
            plt.title(r''+title,fontsize=20)
        plt.xticks([-1,-0.5,0,0.5,1],fontsize=20)
        plt.yticks([-1,-0.5,0,0.5,1],fontsize=20)
        plt.xlabel(r'$\cos(\phi_i)$',fontsize=20)
        plt.ylabel(r'$\sin(\phi_i)$',fontsize=20)
        # plt.legend(loc='best',fontsize=20)
        if save_fig:
            plt.savefig(save_path+file_name)
    
    def plot_1dArr(self,x_arr,y_arr,color:str='red',title:str='',label:str='',x_label:str=r'',y_label=r'',alpha:float=0.8,new_fig:bool=True,save_fig:bool=False,save_path:str='',file_name:str=''):
        if new_fig:
            plt.figure()
        if len(label)>0:
            plt.plot(x_arr,y_arr,color=color,linewidth=2,label=label,alpha=alpha)
        else:
            plt.plot(x_arr,y_arr,color=color,linewidth=2,alpha=alpha)
        if len(title)>0:
            plt.title(r''+title,fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(r''+x_label,fontsize=20)
        plt.ylabel(r''+y_label,fontsize=20)
        if len(label)>0:
            plt.legend(loc='best',fontsize=20)
        if save_fig:
            plt.savefig(save_path+file_name)
        



### ER with connection probability is 0.1
data_path = 'F:/python_work_PyCharm/work_3_20200527/max_cut_OIM_test/'
A_mat = np.load(data_path+'KM_sync_ER_pis_0dot1_Amat_1.npy')

J_mat = -A_mat
N = J_mat.shape[0]



omega = 0
KcKs_arr = [(1,0),(1,2)]
km_sync = Simulation(t_start=0,t_end=100,step_size=0.01)
phis_arr = []
for i,KcKs in enumerate(KcKs_arr):
    phis_arr.append(km_sync.simulate(N=100,J_mat=J_mat,omega=omega,Kc=KcKs[0],Ks=KcKs[1]))



plot_fig = Plot()
fig_path = 'F:/python_work_PyCharm/work_3_20200527/max_cut_OIM_test/Figs/KM_sync_phase_in_circle/'
save_fig = False
t_ends = [1500,1500]
### plot order parmas, some pre-setting
# color_r = ['blue','red']
# color_r = ['#1f77b4','#d62728']
color_r = ['#009ade','#ff1f5b']
label_r = [r'without $K_s\sin(2\phi)$',r'with $K_s\sin(2\phi)$']
file_phi = ['KM_phase_t_ER_Nis100_1.pdf','KM_sync_phase_t_ER_Nis100_1.pdf']
file_cir = ['KM_phase_circle_ER_Nis100_1.pdf','KM_sync_phase_circle_ER_Nis100_1.pdf']

for i,phi_mat in enumerate(phis_arr):
    title_phis = r'ER, $\phi_i$ vs $t$, $K_g$='+(str(KcKs_arr[i][0]))+r',$K_s$='+str(str(KcKs_arr[i][1]))+r',$N=$'+str(N)
    title_cir = r'ER, $\phi_i$ distribution, $K_g$='+(str(KcKs_arr[i][0]))+r',$K_s$='+str(str(KcKs_arr[i][1]))+r',$N=$'+str(N)
    t_arr = km_sync.t_arr[:t_ends[i]]
    plot_fig.plot_2Dphases(t_arr=t_arr,phi_mat=phi_mat[:,:t_ends[i]],title=title_phis,save_fig=save_fig,save_path=fig_path,file_name=file_phi[i])
    phis = phi_mat[:,t_ends[i]]
    plot_fig.plot_phisIncircle(phis=phis,mark='o',markersize=10,color_cir='black',title=title_cir,alpha_cir=0.5,save_fig=save_fig,save_path=fig_path,file_name=file_cir[i])

file_name = 'KM_sync_orderParams_t_ER_Nis100_1.pdf'
### order params
for i,phi_mat in enumerate(phis_arr):
    r_arr = km_sync.get_orderParams(phi_mat=phi_mat[:,:t_ends[i]],N=N)
    if i==0:
        plot_fig.plot_1dArr(x_arr=t_arr,y_arr=r_arr,color=color_r[i],x_label=r't',y_label='Order Parameter',label=label_r[i])
    else:
        plot_fig.plot_1dArr(x_arr=t_arr,y_arr=r_arr,color=color_r[i],x_label=r't',y_label='Order Parameter',label=label_r[i],new_fig=False,save_fig=save_fig,save_path=fig_path,file_name=file_name)
file_name = 'KM_sync_E_t_ER_Nis100_1.pdf'
### energy
for i,phi_mat in enumerate(phis_arr):
    E_arr = km_sync.cal_energy(phi_mat=phi_mat[:,:t_ends[i]],J_tmp=J_mat,Kc_tmp=KcKs_arr[i][0],Ks_tmp=KcKs_arr[i][1])
    if i==0:
        plot_fig.plot_1dArr(x_arr=t_arr,y_arr=E_arr,color=color_r[i],x_label=r't',y_label='Energy',label=label_r[i])
    else:
        plot_fig.plot_1dArr(x_arr=t_arr,y_arr=E_arr,color=color_r[i],x_label=r't',y_label='Energy',label=label_r[i],new_fig=False,save_fig=save_fig,save_path=fig_path,file_name=file_name)
plt.show()
time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")

### plot res


colors = ['#B1CE46','#F1D77E']

# ### plot moving phase
# ### omega = 0.8
# ### KcKs_arr = [(0.5,0),[0,0.5]]
# plt.figure()
# labels = ['$\\frac{1}{2}\,sin(\\phi)$','$\\frac{1}{2}\,sin(2\\phi)$']
# for i,KcKs in enumerate(KcKs_arr):
#     phis = phi_arr[i].copy()
#     phis /= math.pi
#     plt.plot(km_sync.t_arr,phis[0]%2,linewidth=2,color=colors[i],alpha=0.8,label=r''+labels[i])
# plt.title(r'$\omega=$'+str(omega),fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlabel(r'$t$',fontsize=20)
# plt.ylabel(r'$\phi (\pi)$',fontsize=20)
# plt.legend(loc='best',fontsize=20)

plt.show()
print()