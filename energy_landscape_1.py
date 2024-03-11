import numpy as np
import matplotlib.pyplot as plt
import math
import time

time_start = time.time()


x = np.arange(0,2*math.pi,0.01)
y = np.arange(0,2*math.pi,0.01)
X,Y = np.meshgrid(x,y)
# energy function of two connected nodes
J_12 = 1
J_21 = 1
E1 = -J_12*np.cos(X-Y)-J_21*np.cos(Y-X)-np.cos(2*X)-np.cos(2*Y)

E2 = -J_12*np.cos(X-Y)-J_21*np.cos(Y-X)

time_terminal = time.time()
print('totally cost', str("{:.2f}".format(time_terminal - time_start)) + "s")


fig = plt.figure()  #定义新的三维坐标轴
ax1 = plt.axes(projection='3d')

#作图
ax1.plot_surface(X/math.pi,Y/math.pi,E1,cmap='rainbow')
# ax1.plot_surface(X,Y,E1,rstride = 1, cstride = 1,cmap='rainbow')
# ax1.contour(X,Y,E1, zdim='z',offset=-2, cmap='rainbow')  #等高线图，要设置offset，为Z的最小值
ax1.set_xlabel(r'$\phi_1(\pi)$',fontsize=15)
ax1.set_ylabel(r'$\phi_2(\pi)$',fontsize=15)
ax1.set_zlabel(r'$E(\phi)$',fontsize=15)
ax1.tick_params(axis='both',labelsize=15)


fig = plt.figure()  #定义新的三维坐标轴
ax2 = plt.axes(projection='3d')

#作图
ax2.plot_surface(X/math.pi,Y/math.pi,E2,cmap='rainbow')
# ax2.plot_surface(X,Y,E1,rstride = 1, cstride = 1,cmap='rainbow')
# ax2.contour(X,Y,E2, zdim='z',offset=-2, cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
ax2.set_xlabel(r'$\phi_1(\pi)$',fontsize=15)
ax2.set_ylabel(r'$\phi_2(\pi)$',fontsize=15)
ax2.set_zlabel(r'$E(\phi)$',fontsize=15)
ax2.tick_params(axis='both',labelsize=15)
plt.show()

print()
# plt.figure(figsize=[6.4 * 1.2, 4.8 * 1.2])
# # plt.plot(t_arr[int(Total_steps/2):], phi_mat[:,int(Total_steps/2):].T, alpha=0.4)
# # plt.plot(t_arr.cpu()[-5000:], (phi_mat.cpu()[:,-5000:]/math.pi).T, alpha=0.6)
# plt.plot(t_arr.cpu()[:], (phi_mat.cpu()[:,:]/math.pi).T, alpha=0.6)
# plt.title(r'$K_s$='+str(Ks)+r', $\sigma$='+str(sigma),fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xlabel(r"time (s)", fontsize=15)
# plt.ylabel(r"$\phi_i$ ($\pi$)", fontsize=15)
# plt.show()
