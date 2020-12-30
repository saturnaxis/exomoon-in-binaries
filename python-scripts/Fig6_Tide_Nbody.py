import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker
import os

rcParams.update({'font.size': 24})
rcParams.update({'mathtext.fontset': 'cm'})

home = os.getcwd() + "/"


star = ['N-body','Secular']
color = ['c','b','m','r']

G = 4.*np.pi**2
M_E = 3.0035e-6
R_E = 4.26352e-5 #Radius of Earth in AU

aspect = 16./9.
width = 8.
lw = 6
ms = 20
fs = 'x-large'

fig = plt.figure(1,figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax_list = [ax1,ax2]
xmin = 100
xmax = 1.5e7
sub_lbl = ['a','b']

for i in range(0,2):
    ax = ax_list[i]
    ax_top = ax.twiny()

    ax_top.minorticks_on()
    ax_top.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    ax_top.tick_params(which='minor',axis='both', direction='out',length = 8.0, width = 6.0)

    ax_top.set_xscale('log')

    ax_top.set_xlim(xmin,xmax)
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax_top.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0,1.1,0.1),numticks=12)
    ax_top.xaxis.set_minor_locator(locmin)
    ax_top.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    if i == 1:
        fname = 'acen_B'
        M_star = 0.972
        R_H = np.sqrt(0.5)*(1.0123*M_E/(3.*M_star))**(1./3.)
        n_p = np.sqrt(G*M_star/np.sqrt(0.5)**3)
        ax.set_xlabel("Time (yr)",fontsize=fs)
        ax_top.set_xticklabels([])
    else:
        fname = 'acen_A'
        M_star = 1.133
        R_H = np.sqrt(1.519)*(1.0123*M_E/(3.*M_star))**(1./3.)
        n_p = np.sqrt(G*M_star/np.sqrt(1.519)**3)
        
        
    n_crit = np.sqrt(3./0.4**3)*n_p
    n_Roche = np.sqrt(G*M_E/(3*R_E)**3)
    Omega_p = 2.*np.pi/(10./24./365.25)

    data = np.genfromtxt(home+fname+"_out_100.txt",delimiter=',',comments='#')
    ax.plot(data[:,0],(G*M_E*1.0123/data[:,3]**2)**(1./3.)/R_H,'-',color='k',label='N-body',lw=lw,zorder = 2)

    data = np.genfromtxt(home+fname+"_secular_100.txt",delimiter=',',comments='#')
    ax.plot(data[:,0],(G*M_E*1.0123/data[:,1]**2)**(1./3.)/R_H,':',color='r',lw=lw,label='Secular',zorder=5)
    ax.axhline((G*M_E*1.0123/n_Roche**2)**(1./3.)/R_H,linestyle='-.',lw=lw,color='c',label='Roche Limit')

    ax.text(0.99,0.9,"$\\alpha$ Cen %s" % fname[-1] , color='k',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
    ax.text(0.02,0.9,sub_lbl[i], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.legend(loc='lower right',fontsize='large',ncol=3,frameon=False)
    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    ax.tick_params(which='minor',axis='both', direction='out',length = 8.0, width = 6.0)


    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0.01,0.4)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yticklabels(['','0.01','0.1'])
    #ax.set_yticklabels(['','1','10','100'])
    #ax.set_xticks([1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10])

    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0,1.1,0.1),numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


    
    ax.set_ylabel("$a_{\\rm sat}$ (R$_{\\rm H}$)",fontsize=fs)

fig.savefig("acen_tide_compare.png",bbox_inches='tight',dpi=300)
plt.close()
