import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import erf,erfinv
from scipy.interpolate import griddata,interp2d
from scipy.stats import truncnorm
import sys
import os

rcParams.update({'font.size': 26})
rcParams.update({'mathtext.fontset': 'cm'})

def colorbar(mappable,orient):
    last_axes = plt.gca()
    ax = mappable.axes
    fig1 = ax.figure
    divider = make_axes_locatable(ax)
    if orient == 'right':
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig1.colorbar(mappable, cax=cax,orientation='vertical')
        cbar.ax.yaxis.set_ticks_position('right')
    else:
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig1.colorbar(mappable, cax=cax,orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
    cbar.set_ticks([10,100])
    cbar.set_ticklabels(['10','100'])
    cbar.ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    cbar.ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)
    plt.sca(last_axes)
    return cbar

G = 4.*np.pi**2
M_st = int(sys.argv[1])
if M_st == 0:
    M_A = 1.2
    ymax = [45,45,65,65]
elif M_st == 1:
    M_A = 1.
    ymax = [30,30,50,50]
else:
    M_A = 0.8
    ymax = [20,20,30,30]

e_p = 0.
i_m = 0. # change to i_m = 180. for retrograde moons
m_p = 3.0035e-6 #Earth mass
Q_p = 10.
k2_p = 0.299
R_p = 4.26352e-5 #Earth Radius in AU
T = 1e10 #Moon lifetime in yr

data_dir = "../data/Fig11/"
fname = data_dir + "MC_sample_%1.2f.txt" % M_A

cmap = cm.nipy_spectral_r
vmin = 10
vmax = 5500
my_cmap=cm.get_cmap(cmap)
norm = colors.LogNorm(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))
cmap.set_under('w')

fs = 'x-large'
width = 10.
aspect = 2.
ms = 6.
lw = 5

fig = plt.figure(figsize=(aspect*width,width),dpi=300)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

data_MC = np.genfromtxt(fname,delimiter=',',comments='#')
ax_list = [ax1,ax2,ax3,ax4]
sublbl = ['a','b','c','d']

for m in range(0,2):
    if m ==1:
        i_m = 180
    else:
        i_m = 0
    
    host = ['A','B']
    color = ['k','r']
    for k in range(0,2):
        ax = ax_list[2*m+k]

        set_cut = np.where(np.logical_and(np.abs(data_MC[:,0]-k)<1e-6,np.abs(data_MC[:,1]-i_m)<1e-6))[0]
        xbins = np.arange(0.001,0.1005,0.0005)

        ybins = np.arange(0.1,ymax[2*m+k]+0.1,0.1)
        h,xedges,yedges = np.histogram2d(data_MC[set_cut,2],data_MC[set_cut,3],bins=[xbins,ybins])
        CS = ax.pcolormesh(xedges[:-1],yedges[:-1],h.T,vmin=vmin,vmax=vmax,cmap=cmap,norm=norm)
        
        ax.minorticks_on()
        ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
        ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0)
        
        ax.set_xlim(0.001,0.1)
        ax.set_ylim(0.1,ymax[2*m+k]+10)
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_yticks([1,10])
        ax.set_yticklabels(['1','10'])
        ax.set_xticks([0.001,0.01,0.1])
        ax.set_xticklabels(['0.001','0.01','0.1'])
        if m==0:
            ax.text(0.99,0.9,"prograde" , color='k',fontsize='large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
        else:
            ax.set_xlabel("$a_{\\rm p}/a_{\\rm bin}$",fontsize=fs)
            ax.text(0.99,0.9,"retrograde" , color='k',fontsize='large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
        ax.text(0.02,0.88,sublbl[2*m+k], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)

        if k == 0:
            ax.set_ylabel("TTV (min)",fontsize=fs)
        if 2*m+k == 0:
            ax.text(0.5,1.05,"Star A" , color='k',fontsize='large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
        elif 2*m+k == 1:
            ax.text(0.5,1.05,"Star B" , color='k',fontsize='large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
        if (2*m+k) < 2:
            ax.set_xticklabels([])

color_label = 'Number of Systems'
cax = fig.add_axes([0.92,0.11,0.015,0.77])
cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
cbar.set_label(color_label,fontsize=fs)

cbar.set_ticks([10,100,1000])
cbar.set_ticklabels(['10','100','1000'])
cbar.ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
cbar.ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)

fig.subplots_adjust(hspace=0.1)
fig.savefig("../Figs/Fig11_MonteCarlo.png",bbox_inches='tight',dpi=300)
plt.close()
