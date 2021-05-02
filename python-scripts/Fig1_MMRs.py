import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar, curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import rcParams
import sys

rcParams.update({'font.size': 22})
rcParams.update({'mathtext.fontset': 'cm'})

G = 4.*np.pi**2
data_dir = "../data/Fig1/"

#constants
M_A = 1.133
L_A = 1.519
M_B = 0.972
L_B = 0.5
M_p = 9.54e-4
a_p = np.sqrt(L_A)
M_sat = 3.0035e-6

M_123 = M_A + M_p + M_sat
M_12 =  M_p + M_sat
e_i = 1e-4

R_H = a_p*(M_12/(3.*M_A))**(1./3.)

def sigma_N(x,k,M_st):
    x = np.abs(x)
    if x <1e-3:
        x = 1e-3    
    xi = np.arccosh(1./x) - np.sqrt(1.-x**2)
    M_123 = M_st + M_p + M_sat
    dsN = (6.*np.sqrt(0.71)/(2.*np.pi)**0.25)*np.sqrt(M_st/M_123 + (k*M_12/M_123)**(2./3)*(M_p*M_sat/M_12**2))
    dsN = dsN*np.sqrt(e_i)/x*(1.-x**2)**(3./8.)*k**0.75*np.exp(-k*xi/2.)
    return dsN

def int_func(x,k,M_st):
    return (3./k**2)**(1./3.) - (3./(k+1)**2)**(1./3.) - sigma_N(x,k,M_st) - sigma_N(x,k+1,M_st)

def fit_intersect(x,a,b):
    return a*x**b

#Resonance widths for a N:1 MMR between moon and planet around star
N_min = 3
N_max = 15
ecc_rng = np.concatenate(([0.001],np.arange(0.01,0.61,0.01)))
star = ['A','B']
for st in star[:1]:
    M_st = M_A
    if st == 'B':
        M_st = M_B
    out = open(data_dir+"res_width_N1_%s.txt" % st,'w')
    out.write("#N,e_p,a_res,k_left,k_right,e_int,a_int\n")
    for N in range(N_min,N_max):
        for ecc in ecc_rng:
            a_res = (3./N**2)**(1./3.)#*R_H
           
            #Resonance width (Mardling 2013)
            delta_sigma_N = sigma_N(ecc,N,M_st)
            e_int = 0
            if N >= N_min:
                intersect = root_scalar(int_func,args=(N,M_st),x0=0.25,x1=0.4)
                e_int = intersect.root
                a_int = a_res-sigma_N(e_int,N,M_st)
            else:
                a_int = 0
            out.write("%i,%1.2f,%1.5e,%1.5e,%1.5e,%1.5e,%1.5e\n" % (N,ecc,a_res,a_res+delta_sigma_N,a_res-delta_sigma_N,e_int,a_int))
    out.close()


aspect = 1.
width = 10.
lw = 4
fs = 'x-large'

cmap = cm.Set3
vmin = 3
vmax = 15
my_cmap=cm.get_cmap(cmap)
norm = colors.Normalize(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))



fig = plt.figure(1,figsize=(aspect*width,width),dpi=300)
ax = fig.add_subplot(111)
ax.set_facecolor('k')

data = np.genfromtxt(data_dir+"res_width_N1_A.txt",delimiter=',',comments='#')
intersect = []
for N in range(N_min,N_max):
    N_idx = np.where(np.abs(data[:,0]-N)<1e-6)[0]
    ax.fill_between(data[N_idx,1],data[N_idx,4],data[N_idx,3],color='w',zorder=2)
    ax.fill_between(data[N_idx,1],data[N_idx,4],data[N_idx,3],color=my_cmap(norm(N)),alpha=0.65,zorder=3)
    ax.plot(data[N_idx,1],data[N_idx,3],'-',color='w',lw=3,zorder=4)
    ax.plot(data[N_idx,1],data[N_idx,4],'-',color='w',lw=3,zorder=4)
    if N>=5:
        intersect.append((data[N_idx[0],5],data[N_idx[0],6]))

eps = 1.25*a_p/23.78*(0.524/(1.-0.524**2))
intersect = np.asarray(intersect)
spl = UnivariateSpline(intersect[:,0],intersect[:,1])
ep_rng = np.arange(0.289,0.41,0.001)

popt, pcov = curve_fit(fit_intersect, intersect[:,1],intersect[:,0],method='lm')
#print(popt,np.sqrt(np.diag(pcov)))
ax.plot(ep_rng,spl(ep_rng),'-',color='lightgreen',lw=lw,zorder=5)


ep_rng = np.arange(0.0,0.61,0.01)
eps = 1.25*a_p/23.78*(0.524/(1.-0.524**2))
eta = np.abs(ep_rng-eps)
y_stab = (1.- eps - eta)


ax.minorticks_on()
ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 8.0)

ax.set_xlabel(r"initial $e_{\rm p}$",fontsize=50)
ax.set_ylabel(r"$a_{\rm sat}\; (R_{\rm H}$)",fontsize =50)
ax.set_ylim(0.25,0.7)
ax.set_yticks(np.arange(0.25,0.75,0.05))
ax.set_xlim(0.0,0.6)
ax.set_xticks(np.arange(0.,0.7,0.1))


color_label=r'N:1 MMR'
cax = fig.add_axes([0.92,0.11,0.015,0.77])
cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
cbar.set_label(color_label,fontsize=fs)
cbar.set_ticks(np.arange(3.5,16.5,1.))
cbar.set_ticklabels(['%i' % n for n in range(3,16)])
cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

fig.savefig("../Figs/Fig1_MMR.png",bbox_inches='tight',dpi=300)
plt.close()