import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker
import os


rcParams.update({'font.size': 26})
rcParams.update({'mathtext.fontset': 'cm'})

data_dir = "../data/Fig8/"
Q_rng = [10,100]
Mf_rng = [0.0123,0.3]
tau = [10,20,50,100,200,600]
m_sat = [0.001,0.002,0.005,0.01,0.02,0.05]
star = ['A','B','A','B']
sublabel = ['a','b','c','d']
color = ['red','orange','cyan','blue','purple','violet']

G = 4.*np.pi**2
M_E = 3.0035e-6
R_E = 4.26352e-5 #Radius of Earth in AU

aspect = 2*16./9.
width = 10.
lw = 8
ms = 20
fs = 72#'x-large'

fig = plt.figure(1,figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax_list = [ax1,ax2,ax3,ax4]

for s in range(0,4):
    ax = ax_list[s]
    ax_top = ax.twiny()

    ax_top.minorticks_on()
    ax_top.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=45)
    ax_top.tick_params(which='minor',axis='both', direction='out',length = 8.0, width = 6.0)

    ax_top.set_xscale('log')

    ax_top.set_xlim(1e2,1e10)
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax_top.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0,1.1,0.1),numticks=12)
    ax_top.xaxis.set_minor_locator(locmin)
    ax_top.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    
    if s == 1 or s == 3:
        M_star = 0.972
        R_H = np.sqrt(0.5)*(1.0123*M_E/(3.*M_star))**(1./3.)
        n_p = np.sqrt(G*M_star/np.sqrt(0.5)**3)
        
    elif s == 0 or s == 2:
        M_star = 1.133
        R_H = np.sqrt(1.519)*(1.0123*M_E/(3.*M_star))**(1./3.)
        n_p = np.sqrt(G*M_star/np.sqrt(1.519)**3)
    
    n_crit = np.sqrt(3./0.4**3)*n_p
    n_Roche = np.sqrt(G*M_E/(3*R_E)**3)
    Omega_p = 2.*np.pi/(10./24./365.25)
    cnt = 0
    if s <2:
        for t in tau:
            data = np.genfromtxt(data_dir+"acen%s_secular_%03i.txt" % (star[s],t),delimiter=',',comments='#')
            ax.plot(data[:,0],(G*M_E*1.0123/data[:,1]**2)**(1./3.)/R_H,'-',color=color[cnt],lw=lw,label='%i s' % t,zorder=5)
            cnt += 1
        
    else:
        for m in m_sat:
            data = np.genfromtxt(data_dir+"acen%s_secular_%1.1e.txt" % (star[s],m),delimiter=',',comments='#')
            ax.plot(data[:,0],(G*M_E*1.0123/data[:,1]**2)**(1./3.)/R_H,'-',color=color[cnt],lw=lw,label='%s M$_\oplus$' % (str(m)),zorder=5)
            cnt += 1
        ax_top.set_xticklabels([])
        ax.set_xlabel("Time (yr)",fontsize=fs)
    ax.fill_between(10**np.arange(2,10.5,0.5),0.38,0.45,color='gray')
    ax.legend(loc='lower right',fontsize='large',ncol=3,frameon=False) #bbox_to_anchor=(1., 0.32)

    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=45)
    ax.tick_params(which='minor',axis='both', direction='out',length = 8.0, width = 6.0)
    ax.text(0.02,0.88,sublabel[s], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.text(0.02,0.78,'$\\alpha$ Cen %s' % star[s], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.set_xlim(1e2,1e10)

    ax.set_ylim(0.01,0.45)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yticklabels(['','0.01','0.1'])
    
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0,1.1,0.1),numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    if s == 0 or s == 2:
        ax.set_ylabel("$a_{\\rm sat}$ (R$_{\\rm H}$)",fontsize=fs)
    if s == 1 or s == 3:
        ax.set_yticklabels([])
    if s < 2:
        ax.set_xticklabels([])
fig.subplots_adjust(wspace=0.1,hspace=0.1)
fig.savefig("../Figs/Fig8_secular.png",bbox_inches='tight',dpi=300)
plt.close()
