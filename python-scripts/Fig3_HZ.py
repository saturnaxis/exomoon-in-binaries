import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc,rcParams
import sys
import os

rcParams.update({'font.size': 22})
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

def plot_ep(ax,data_dir,data_file,lbl,idx,fname,tscale):
    pos = ax.figbox.get_points()
    cmap = cm.gnuplot_r
    vmin = 0.
    vmax = 1.
    my_cmap=cm.get_cmap(cmap)
    norm = colors.Normalize(vmin,vmax)
    cmmapable =cm.ScalarMappable(norm,my_cmap)
    cmmapable.set_array(range(0,1))
    cmap.set_under('w')

    data = np.genfromtxt(data_dir+data_file,delimiter=',',comments='#')
    stab = np.where(np.abs(data[:,-1]-tscale)<1e-6)[0]
    #print(len(stab),len(data))

    xi = np.arange(0,0.61,0.01)
    yi = np.arange(0.25,0.71,0.01)
    X = []
    Y = []
    Z = []
    out = open(data_dir+fname,'w')
    out.write("#e_p,R_H,f_stab\n")
    for xp in xi:
        for yp in yi:
            X.append(xp)
            Y.append(yp)
            xy_cut = np.where(np.logical_and(np.abs(data[:,2]-xp)<1e-6,np.abs(data[:,0]-yp)<1e-6))[0]
            stab_frac = 0.
            if len(xy_cut)>0:
                for cnt in range(0,len(xy_cut)):
                    if np.abs(data[xy_cut[cnt],-1]-tscale)<1e-6:
                        stab_frac += 1.
                if stab_frac <= 0.05:
                    stab_frac -= 1.
                stab_frac /= float(len(xy_cut))
            else:
                stab_frac = -1.
            Z.append(stab_frac)
            out.write("%1.3f,%1.3f,%1.5f\n" % (xp,yp,stab_frac))
    out.close()
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    Z[Z>=0.95] = 1.

    zi = griddata((X,Y),Z,(xi[None,:],yi[:,None]),method = 'linear') #,fill_value=0
    CS = ax.pcolormesh(xi,yi,zi,cmap = cmap,vmin=vmin,vmax=vmax)
    x_data = []
    y_data = []
    
    eps = 1.25*a_p[idx-1]/23.78*(0.524/(1.-0.524**2))
    #print(eps)
    eta = np.abs(xi-eps)
    y_stab = (1. - eps - eta)
    if idx < 3:
        ax.plot(xi,0.4*y_stab,'--',color='gray',lw=lw)
        ax.plot(xi,0.5*y_stab,'-',color='gray',lw=lw)
    else:
        ax.plot(xi,0.65*y_stab,'--',color='gray',lw=lw)
        ax.plot(xi,0.95*y_stab,'-',color='gray',lw=lw)
        N_min = 3
        N_max = 15
        data = np.genfromtxt("../data/Fig1/res_width_N1_A.txt",delimiter=',',comments='#')
        intersect = []
        for N in range(N_min,N_max):
            N_idx = np.where(np.abs(data[:,0]-N)<1e-6)[0]
            if N>=5:
                intersect.append((data[N_idx[0],5],data[N_idx[0],6]))
        intersect = np.asarray(intersect)
        spl = UnivariateSpline(intersect[:,0],intersect[:,1])
        ep_rng = np.arange(0.289,0.41,0.001)
        ax.plot(ep_rng,spl(ep_rng),'-',color='lightgreen',lw=lw,zorder=5)

    
    if idx > 2:
        ax.set_xlabel(r"initial $e_{\rm p}$",fontsize=50)
    if idx == 1 or idx == 3:
        ax.set_ylabel(r"initial $a_{\rm %s}\;(R_{\rm H})$" % lbl,fontsize=50)

    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 8.0)

    ax.text(0.99,0.82,"%1.3f au" % a_p[idx-1], color='k',fontsize='xx-large',horizontalalignment='right',transform=ax.transAxes)
    if idx > 2:
        ax.set_ylim(0.25,0.7)
        ax.set_yticks(np.arange(0.25,0.75,0.05))
        ax.text(0.99,0.9,"retrograde", color='k',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
    else:
        ax.set_ylim(0.25,0.55)
        ax.set_yticks(np.arange(0.25,0.6,0.05))
        ax.text(0.5,1.02,"%s" % star_lbl[idx-1], color='k',fontsize='xx-large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
        ax.text(0.99,0.9,"prograde", color='k',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)

    ax.set_xlim(0.,0.6)
    ax.set_xticks(np.arange(0,0.7,0.1))
    if idx == 4:
        color_label=r'$f_{\rm stab}$'
        cax = fig.add_axes([0.92,0.11,0.02,0.77])
        cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
        cbar.set_label(color_label,fontsize=fs)
        cbar.set_ticks(np.arange(0.,1.25,0.25))
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

data_dir = "../data/Fig3/"

fs = 'xx-large'
width = 10.
aspect = 2.1
ms = 6.5
lw=8
sublbl = [r'$\textbf{a}$',r'$\textbf{b}$',r'$\textbf{c}$',r'$\textbf{d}$']
a_p = [1.232,0.707,1.232,0.707]
star_lbl = ['$\\alpha$ Cen A','$\\alpha$ Cen B']


fig = plt.figure(figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)



plot_ep(ax1,data_dir,"Stab_moon_HZ_A.txt","sat",1,"Stab_frac_moon_A.txt",1e5)
plot_ep(ax2,data_dir,"Stab_moon_HZ_B.txt","sat",2,"Stab_frac_moon_B.txt",1e5)
plot_ep(ax3,data_dir,"Stab_moon_HZ_AR.txt","sat",3,"Stab_frac_moon_AR.txt",1e6)
plot_ep(ax4,data_dir,"Stab_moon_HZ_BR.txt","sat",4,"Stab_frac_moon_BR.txt",1e6)


fig.subplots_adjust(wspace=0.25,hspace=0.15)

fig.savefig("../Figs/Fig3_HZ.png",bbox_inches='tight',dpi=300)
plt.close()
