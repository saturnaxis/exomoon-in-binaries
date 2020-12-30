import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import matplotlib.ticker
import os

rcParams.update({'font.size': 22})
rcParams.update({'mathtext.fontset': 'cm'})

def get_params(x1,x2):
    r_m = [x1[0]-x2[0],x1[1]-x2[1],x1[2]-x2[2]] #position of moon  w/ respect to planet
    v_m = [x1[3]-x2[3],x1[4]-x2[4],x1[5]-x2[5]] #velocities of moon w/ respect to planet
    h_m = [r_m[1]*v_m[2]-r_m[2]*v_m[1],r_m[2]*v_m[0]-r_m[0]*v_m[2],r_m[0]*v_m[1]-r_m[1]*v_m[0]] #angular momentum of moon w/ respect to planet
    #Murray & Dermott Chp 2, pg 53 (pg 67 in pdf)
    a_m = 1./(2./np.sqrt(np.dot(r_m,r_m)) - np.dot(v_m,v_m)/(G2*1.0123*M_E)) #semi of moon 
    e_m = np.sqrt(1.-np.dot(h_m,h_m)/(G2*1.0123*M_E*a_m))#eccentricity of moon
    n_m = np.sqrt(G2*1.0123*M_E/a_m**3)*365.25 
    return a_m,e_m,n_m

def fit_power(x,C,a,k,b):
    return C*a**(k*x) + b

data_dir = "../data/Fig6/"


star = ['N-body','Secular']
color = ['c','b','m','r']

G = 4.*np.pi**2
G2 = 4*np.pi**2/365.25**2
M_E = 3.0035e-6
R_E = 4.26352e-5 #Radius of Earth in AU

aspect = 16./9.
width = 12.
lw = 6
ms = 20
fs = 'x-large'

fig = plt.figure(1,figsize=(aspect*width,width),dpi=300)
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
        
    star_dir = '../data/Fig6/aCen%s/' % fname[-1]

    data_p = np.genfromtxt(star_dir+"Earthmoo_history_1.txt",delimiter='\t',skip_header=1)
    data_m = np.genfromtxt(star_dir+"Earthmoo_history_2.txt",delimiter='\t',skip_header=1)
    data_sec = np.genfromtxt(data_dir+"acen_%s_secular_100.txt" % fname[-1],delimiter=',',comments='#')
    time = data_m[:,0]/365.25
    data_moon = np.zeros((len(data_p[:,0]),3))
    for k in range(0,len(time)):
        temp_pl = np.concatenate((data_p[k,2:5],data_p[k,8:11]))
        temp_moon = np.concatenate((data_m[k,2:5],data_m[k,8:11]))
        data_moon[k,:] = get_params(temp_pl,temp_moon)
        
    n_crit = np.sqrt(3./0.4**3)*n_p
    n_Roche = np.sqrt(G*M_E/(3*R_E)**3)
    Omega_p = 2.*np.pi/(10./24./365.25)

    ax.plot(time,data_moon[:,0]/R_H,'k-',lw=lw,label='Posidonius')
    ax.plot(data_sec[:,0],(G*M_E*1.0123/data_sec[:,1]**2)**(1./3.)/R_H,':',color='r',lw=lw,label='Secular',zorder=5)
    ax.axhline((G*M_E*1.0123/n_Roche**2)**(1./3.)/R_H,linestyle='-.',lw=lw,color='c',label='Roche Limit')

    ax.text(0.99,0.9,"$\\alpha$ Cen %s" % fname[-1] , color='k',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
    ax.text(0.02,0.9,sub_lbl[i], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.legend(loc='lower right',fontsize='medium',ncol=3,frameon=False)
    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    ax.tick_params(which='minor',axis='both', direction='out',length = 8.0, width = 6.0)


    ax.set_xlim(xmin,xmax)
    ax.set_ylim(0.01,0.4)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yticklabels(['','0.01','0.1'])

    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0,1.1,0.1),numticks=12)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_ylabel("$a_{\\rm sat}$ (R$_{\\rm H}$)",fontsize=fs)

fig.savefig("../Figs/Fig6_Tide_Nbody.png",bbox_inches='tight',dpi=300)
plt.close()
