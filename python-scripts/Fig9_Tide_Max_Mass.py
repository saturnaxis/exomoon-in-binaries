import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from scipy.optimize import curve_fit, root_scalar
import sys
import os

rcParams.update({'font.size': 26})
rcParams.update({'mathtext.fontset': 'cm'})


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax,orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    plt.sca(last_axes)
    return cbar

def f_ecc(f_idx,ecc):
    if f_idx == 1:
        return (1. + 31./2.*ecc**2 + 255./8.*ecc**4 + 185./16.*ecc**6 + 25./64.*ecc**8)/(1.-ecc**2)**(15./2.)
    elif f_idx == 2:
        return (1. + 15./2.*ecc**2 + 45./8.*ecc**4 + 5./16.*ecc**6)/(1.-ecc**2)**6
    elif f_idx == 3:
        return (1. + 15./4.*ecc**2 + 15./8.*ecc**4 + 5./64.*ecc**6)/(1.-ecc**2)**(13./2.)
    elif f_idx == 4:
        return (1. + 3./2.*ecc**2 + 1./8.*ecc**4)/(1.-ecc**2)**5
    elif f_idx == 5:
        return (1. + 3.*ecc**2 + 3./8.*ecc**4)/(1.-ecc**2)**(9./2.)

def func(x,f_c,a_p,k2p,M_p,R_p,Q_p,M_star,Tmax):
    a_c = f_c*a_p*(M_p*(1+x)/(3.*M_star))**(1./3.)
    a_o = 3.*2.88*R_p
    Q_fact = Q_p/(3.*k2p*R_p**5*Tmax*np.sqrt(G*M_p))
    return x*np.sqrt(1.+x) - (2./13.)*(a_c**(13./2.)-a_o**(13./2.))*Q_fact

data_dir = "../data/Fig9/"

fs = 50#'xx-large'
width = 10.
aspect = 2.1
ms = 6.5
lw=5
#sublbl = [r'$\textbf{a}$',r'$\textbf{b}$',r'$\textbf{c}$',r'$\textbf{d}$']
sublbl = ['a','b','c','d']
star = ['A','B','A','B']

G = 4.*np.pi**2
M_E = 3.0035e-6
R_E = 4.2635e-5 #AU

fig = plt.figure(figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax_list = [ax1,ax2,ax3,ax4]

cmap = cm.jet
vmin = 0.02/0.4
vmax = 1.
my_cmap=cm.get_cmap(cmap)
norm = colors.Normalize(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))
cmap.set_under('gray')

ep_rng = np.arange(0,0.61,0.01)
xi = np.arange(0.16,0.383,0.003)
yi = np.concatenate((np.arange(0.001,0.01,0.00025),np.arange(0.01,0.301,0.001)))
Mass_max = np.zeros(len(ep_rng))


for s in range(0,4):
    ax = ax_list[s]
    ax_top = ax.twiny()
    ax_top.set_xlim(0.6,0)
    ax_top.minorticks_on()
    ax_top.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,colors='r')
    ax_top.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 8.0,colors='r')
    if s < 2:
        ax_top.set_xlabel("$e_{\\rm p}$",fontsize=fs,labelpad=25,color='r')
        
    
    if star[s] == 'A':
        a_p = np.sqrt(1.519)
        M_star = 1.133
    else:
        a_p = np.sqrt(0.5)
        M_star = 0.972
    eps = 1.25*a_p/23.78*(0.524/(1.-0.524**2))
    ax_top.plot(eps,0.0123,marker='$\oplus$',color='k',ms=35)
    
    if s ==0:
        Q_p = 33*f_ecc(2,0.1)
    elif s==1:
        Q_p = 33*f_ecc(2,0.15)
    else:
        Q_p = 33
    f_crit = np.zeros(len(ep_rng))
    for ep in range(0,len(ep_rng)):
        f_crit[ep] = 0.4*(1.-eps-np.abs(ep_rng[ep]-eps))
        try:
            sol = root_scalar(func,args=(f_crit[ep],a_p,0.299,M_E,R_E,Q_p,M_star,1e10),method='bisect',bracket=[1e-3,0.3],xtol=1e-6)
            Mass_max[ep] = sol.root
        except ValueError:
            Mass_max[ep] = -1
    
    ax.plot(f_crit[Mass_max>0],Mass_max[Mass_max>0],'k-',lw=8)
    if s < 2:
        fname = "mass_stab_%s_t.txt" % star[s]
    else:
        fname = "mass_stab_%s_Q.txt" % star[s]
    data = np.genfromtxt(data_dir+fname,delimiter=',',comments='#')
    X = data[:,0]
    Y = data[:,1]
    
    
    Z = data[:,2]/X
    a_crit = 0.4*(1.-eps-np.abs(data[:,4]-eps))
    bound_cut = np.where(np.logical_and(data[:,2]*(1+data[:,3])>=a_crit,data[:,2]*(1-data[:,3])<2.88*R_E))[0]
    Z[bound_cut] *= -1
    unstab_cut = np.where(np.logical_and(Z>0,data[:,-1]/1e10<1))[0]
    Z[unstab_cut] *=- 1

    zi = griddata((X,Y),Z,(xi[None,:],yi[:,None]),method = 'linear') #,fill_value=0
    CS = ax.pcolormesh(xi,yi,zi,cmap = cmap,vmin=vmin,vmax=vmax)

    if s > 1:
        ax.set_xlabel(r"$a_{\rm sat}^{\rm crit}$ (R$_{\rm H})$",fontsize=fs)
    if star[s] == 'A':
        ax.set_ylabel(r"M$_{\rm sat}/$M$_{\rm p}$" ,fontsize=fs)

    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0)
    ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 8.0)
    ax.text(0.02,0.92,sublbl[s], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)

    ax.text(0.99,0.9,"$\\alpha$ Cen %s" % star[s] , color='w',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
    
    ax.set_ylim(0.001,0.1)
    ax.set_xlim(0.16,0.38)
    ax.set_yscale('log')
    xticks = np.arange(0.16,0.4,0.04)

    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    if s == 1  or s==3:
        ax.set_yticklabels([])
    if s > 1:
        xticklabels = ['%1.2f' % x for x in xticks]
        ax.set_xticklabels(xticklabels)
        ax_top.set_xticklabels([])

    if s == 3:
        color_label=r'$a_{\rm fin}/a_{\rm sat}^{\rm crit}$'
        cax = fig.add_axes([0.92,0.11,0.015,0.77])

        cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
        cbar.set_label(color_label,fontsize=fs)
        cbar.set_ticks(np.arange(0.1,1.1,0.1))
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0)


fig.subplots_adjust(wspace=0.12,hspace=0.12)
fig.savefig("../Figs/Fig9_Max_Mass.png",bbox_inches='tight',dpi=300)
plt.close()
