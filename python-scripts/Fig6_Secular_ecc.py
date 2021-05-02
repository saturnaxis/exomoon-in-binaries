import rebound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar,curve_fit
import matplotlib.colors as colors
import matplotlib.cm as cm
import time
import sys
import os
from matplotlib import rcParams


rcParams.update({'font.size': 22})
rcParams.update({'mathtext.fontset': 'cm'})

def Orb2Cart(x,m,mp,com):
    temp_sim = rebound.Simulation()
    temp_sim.integrator = "ias15"
    temp_sim.units = ('days', 'AU', 'Msun')
    temp_sim.add(m=m)
    temp_sim.add(m=mp,a=x[0],e=x[1],inc=np.radians(x[2]),omega=np.radians(x[3]),Omega=np.radians(x[4]),M=np.radians(x[5]))
    temp_ps = temp_sim.particles
    if com == "F":
        temp = [mp,temp_ps[1].x,temp_ps[1].y,temp_ps[1].z,temp_ps[1].vx,temp_ps[1].vy,temp_ps[1].vz]
        return temp
    else:
        temp_sim.move_to_com()
        temp_0 = [m,temp_ps[0].x,temp_ps[0].y,temp_ps[0].z,temp_ps[0].vx,temp_ps[0].vy,temp_ps[0].vz]
        temp_1 = [mp,temp_ps[1].x,temp_ps[1].y,temp_ps[1].z,temp_ps[1].vx,temp_ps[1].vy,temp_ps[1].vz]
        return temp_0, temp_1


def simulation(a_sat,e_p,idx):

    if star == 'A':
        M_A = 1.133
        M_B = 0.972
    elif star == 'B':
        M_A = 0.972
        M_B = 1.133
    else:
        M_A = 1.

    ms = M_J + M_E
    R_H = a_p*(ms/(3.*M_A))**(1./3.)
    R_p = 0.000477894503 #Radius of Jupiter in AU
    MA_sat = 0#np.random.uniform(0,360)
    

    AU = 1.495978707e13
    M_sol = 1.99e33

    ap = 0.25*R_H
    dt_tp = np.sqrt(ap**3/ms)/40. * 365.25

    i_m = 0.
    if retro == 'T':
        i_m = 180.
    temp_state = [a_sat*R_H,0,i_m,0,0,MA_sat]
    Cart_Jup, Cart_moon = Orb2Cart(temp_state,ms,M_E,"T")

    temp_state = [a_p,e_p,0.,0.,0.,0.]
    Cart_JCOM = Orb2Cart(temp_state,M_A,ms,"F")
    if star == 'A' or star == 'B':
        Cart_B = Orb2Cart([23.7,0.524,0,0,0,0],M_A,M_B,"F")


    for i in xrange(1,len(Cart_moon)):
        Cart_Jup[i] += Cart_JCOM[i]
        Cart_moon[i] += Cart_JCOM[i]

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.units = ('days', 'AU', 'Msun')
    sim.dt = dt_tp
    sim.automateSimulationArchive("../data/Fig6/aCen%s_archive_%i.bin" % (star,idx),interval=(10*365.25),deletefile=True)

    sim.add(m=M_A)
    sim.add(m=ms,x=Cart_Jup[1],y=Cart_Jup[2],z=Cart_Jup[3],vx=Cart_Jup[4],vy=Cart_Jup[5],vz=Cart_Jup[6]) #Jupiter
    sim.add(m=M_E,x=Cart_moon[1],y=Cart_moon[2],z=Cart_moon[3],vx=Cart_moon[4],vy=Cart_moon[5],vz=Cart_moon[6]) #Moon
    if star == 'A' or star == 'B':
        sim.add(m=M_B,x=Cart_B[1],y=Cart_B[2],z=Cart_B[3],vx=Cart_B[4],vy=Cart_B[5],vz=Cart_B[6]) #Star B
    sim.move_to_com()
    ps = sim.particles

    t0 = time.time()

    tcol = tscale
    t_sim = sim.dt
    stop_run = True

    while stop_run:
        sim.integrate(t_sim)
        deltax, deltay = ps[2].x-ps[1].x,ps[2].y-ps[1].y
        if np.sqrt(deltax**2+deltay**2)>=R_H:
            tcol = sim.t/365.25
            stop_run = False
        elif np.sqrt(deltax**2+deltay**2)<=R_p:
            tcol = sim.t/365.25
            stop_run = False
        else:
            t_sim += sim.dt
        if t_sim > (tscale*365.25):
            stop_run = False



def write2file(fname,f_str,head):
    lock.acquire() # thread blocks at this line until it can obtain lock
    if head == "head":
            f = open(output_fldr+fname,'w')
    else:
        f = open(output_fldr+fname,'a')
    f.write(f_str)
    lock.release()

def get_semi(ps):
    pl_idx = 1
    moon_idx = 2
    tot_mass = ps[moon_idx].m + ps[pl_idx].m
    r_m = [ps[moon_idx].x-ps[pl_idx].x,ps[moon_idx].y-ps[pl_idx].y,ps[moon_idx].z-ps[pl_idx].z]
    v_m = [ps[moon_idx].vx-ps[pl_idx].vx,ps[moon_idx].vy-ps[pl_idx].vy,ps[moon_idx].vz-ps[pl_idx].vz]
    h_m = [r_m[1]*v_m[2]-r_m[2]*v_m[1],r_m[2]*v_m[0]-r_m[0]*v_m[2],r_m[0]*v_m[1]-r_m[1]*v_m[0]]
    
    a_m = 1/(2./np.sqrt(np.dot(r_m,r_m)) - np.dot(v_m,v_m)/(G*tot_mass))
    e_m = np.sqrt(1.-np.dot(h_m,h_m)/(G*tot_mass*a_m))
    return a_m, e_m
def int_func(x,k,M_st):
    return (3./k**2)**(1./3.) - (3./(k+1)**2)**(1./3.) - sigma_N(x,k,M_st) - sigma_N(x,k+1,M_st)
def fit_intersect(x,a,b):
    return a*x**b
def sigma_N(x,k,M_st):
    x = np.abs(x)
    if x <1e-3:
        x = 1e-3
    if x > 1:
        return -np.inf
    xi = np.arccosh(1./x) - np.sqrt(1.-x**2)
    M_123 = M_st + M_p + M_sat
    dsN = (6.*np.sqrt(0.71)/(2.*np.pi)**0.25)*np.sqrt(M_st/M_123 + (k*M_12/M_123)**(2./3)*(M_p*M_sat/M_12**2))
    dsN = dsN*np.sqrt(e_i)/x*np.sqrt(1.-(13./24.)*e_i**2)*(1.-x**2)**(3./8.)*k**0.75*np.exp(-k*xi/2.)
    return dsN

def get_MMRs(idx,M_A):
    #Resonance widths for a N:1 MMR between moon and planet around star
    N_min = 3
    N_max = 15
    ecc_rng = np.concatenate(([0.001],np.arange(0.01,0.61,0.01)))
    out = open("../data/Fig6/res_width_N%i_%s.txt" % (idx,star),'w')
    out.write("#N,e_p,a_res,k_left,k_right,e_int,a_int\n")
    for N in range(N_min,N_max):
        for ecc in ecc_rng:
            a_res = (3./N**2)**(1./3.)#*R_H
           
            #Resonance width (Mardling 2013)
            delta_sigma_N = sigma_N(ecc,N,M_A)
            e_int = 0
            if N >= N_min:
                intersect = root_scalar(int_func,args=(N,M_A),x0=0.2,x1=0.4)
                e_int = intersect.root
                a_int = a_res-sigma_N(e_int,N,M_A)
            else:
                a_int = 0
            out.write("%i,%1.2f,%1.5e,%1.5e,%1.5e,%1.5e,%1.5e\n" % (N,ecc,a_res,a_res+delta_sigma_N,a_res-delta_sigma_N,e_int,a_int))
    out.close()

def get_MMR_curve(idx,star):
    data = np.genfromtxt("../data/Fig6/res_width_N%i_%s.txt" % (idx,star),delimiter=',',comments='#')
    intersect = []
    for N in range(N_min,N_max):
        N_idx = np.where(np.abs(data[:,0]-N)<1e-6)[0]
        if N>=5:
            intersect.append((data[N_idx[0],5],data[N_idx[0],6]))
    intersect = np.asarray(intersect)
    spl = UnivariateSpline(intersect[:,0],intersect[:,1])
    return spl

G = (2.*np.pi/365.25)**2.
M_J = 9.54e-4
M_E = 3.0035e-6
M_N = 15.*M_E
M_p = M_J
M_sat = M_E

home = os.getcwd() + "/"

tscale = float(sys.argv[1]) #yrs

star = sys.argv[2] #A, B, or S
retro = sys.argv[3] # T (i_m = 180 deg); F (i_m = 0 deg)
run_sim = sys.argv[4]

if star == 'A':
    M_A = 1.133
    M_B = 0.972
elif star == 'B':
    M_A = 0.972
    M_B = 1.133
else:
    M_A = 1.
a_p = 2.25
e_po = [0,0.082,0.25]
M_123 = M_A + M_p + M_sat
M_12 =  M_p + M_sat

if run_sim == 'T':
    simulation(0.3,0.,1)
    simulation(0.3,0.082,2)
    simulation(0.3,0.25,3)


cmap = cm.Set3
vmin = 3
vmax = 15
my_cmap=cm.get_cmap(cmap)
norm = colors.Normalize(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))


aspect = 1.6
width = 12
lw = 5
ms = 8
fs = 'x-large'

fig = plt.figure(1,figsize=(aspect*width,width),dpi=300)
spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, wspace=0.3, hspace = 0.25)
ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[1,0])
ax3 = fig.add_subplot(spec[0,1])
ax4 = fig.add_subplot(spec[1,1])
ax5 = fig.add_subplot(spec[0,2])
ax6 = fig.add_subplot(spec[1,2])
ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]
sublbl = ['a','d','b','e','c','f']

for j in range(0,3):
    sa = rebound.SimulationArchive("../data/Fig6/aCen%s_archive_%i.bin" % (star,j+1))
    time = np.zeros(len(sa))
    ep_t = np.zeros(len(sa))
    asat_t = np.zeros(len(sa))
    esat_t = np.zeros(len(sa))
    mtot = M_J + M_E
    for i, sim in enumerate(sa):
        time[i] = sim.t/365.25
        ep_t[i] = sim.particles[1].e
        R_H = sim.particles[1].a*(mtot/(3.*M_A))**(1./3.)
        asat_t[i],esat_t[i] = get_semi(sim.particles)
        asat_t[i] /= R_H

    e_f = 5./4.*a_p/23.78*0.524/(1.-0.524**2)
    e_f = 0.085

    ep_var = np.abs(np.max(ep_t)-np.min(ep_t))
    esat_vals = [np.min(esat_t),np.max(esat_t)]
    colors = ['lightgreen','m','b']
    ls = ['-','--']
    for i in range(0,2):
        e_i = esat_vals[i]
        if e_i < 0.001:
            e_i = 0.0001
        get_MMRs(i,M_A)
        data = np.genfromtxt("res_width_N%i_%s.txt" % (i,star),delimiter=',',comments='#')
        N_min = 3
        N_max = 15
        intersect = []
        for N in range(N_min,N_max):
            N_idx = np.where(np.abs(data[:,0]-N)<1e-6)[0]
            if N>=5:
                intersect.append((data[N_idx[0],5],data[N_idx[0],6]))

        intersect = np.asarray(intersect)
        spl = UnivariateSpline(intersect[:,0],intersect[:,1])
        ep_rng = np.arange(0.05,0.41,0.001)

        MMR_a = spl(ep_rng)
        y_cut = np.where(MMR_a<0.46)[0]
        ax_list[2*j+1].plot(ep_rng[y_cut],MMR_a[y_cut],linestyle=ls[i],color=colors[i],lw=lw,zorder=5)
        
    eta = np.abs(ep_t-e_f)

    ecc_rng = np.arange(0,0.5,0.01)
    LCO = 0.4*(1-e_f-np.abs(e_f-ecc_rng))

    ax_list[2*j].plot(time/1000,ep_t,'k-',lw=3)
    #ax_list[2*j].plot(time,esat_t,'b-',lw=lw)


    ax_list[2*j+1].plot(ecc_rng,LCO,'--',color='gray',lw=lw,zorder=5)
    ax_list[2*j+1].plot(ep_t,asat_t*(1.+esat_t),'.',color='k',ms=ms,alpha=0.1,zorder=5)

    ep_var = np.abs(np.max(ep_t)-np.min(ep_t))
    print(np.max(ep_t),np.min(ep_t),ep_var)

    ax_list[2*j].set_xlim(0,20)
    ax_list[2*j+1].set_ylim(0.25,0.55)
    ax_list[2*j+1].set_xlim(0,0.5)
    
    ax_list[2*j+1].text(0.99,0.91,'$e_{\\rm sat}$ = %1.3f' % np.round(np.min(esat_t),3), color='lightgreen',fontsize='large',weight='bold',horizontalalignment='right',transform=ax_list[2*j+1].transAxes)
    ax_list[2*j+1].text(0.99,0.81,'$e_{\\rm sat}$ = %1.3f' % np.round(np.max(esat_t),3), color='m',fontsize='large',weight='bold',horizontalalignment='right',transform=ax_list[2*j+1].transAxes)

    ax_list[j].text(0.02,0.91,sublbl[j], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax_list[j].transAxes)
    ax_list[j+3].text(0.02,0.91,sublbl[j+3], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax_list[j+3].transAxes)
    ax_list[2*j].text(0.5,1.05,'initial $e_{\\rm p}$ = %1.3f' % (e_po[j]), color='k',fontsize='large',weight='bold',horizontalalignment='center',transform=ax_list[2*j].transAxes)
    
    ax_list[2*j+1].set_xlabel("$e_{\\rm p}$",fontsize=fs)
    if j == 0:
        ax_list[2*j].set_ylabel("$e_{\\rm p}$",fontsize=fs)
        ax_list[2*j+1].set_ylabel("$Q_{\\rm sat}$ (R$_{\\rm H}$)",fontsize=fs)
    if j<2:
        ax_list[2*j].set_ylim(0.,0.17)
    ax_list[2*j].set_xlabel("Time (kyr)",fontsize=fs)
    
    ax_list[2*j].minorticks_on()
    ax_list[2*j].tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0)
    ax_list[2*j].tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0)
    
    ax_list[2*j+1].minorticks_on()
    ax_list[2*j+1].tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0)
    ax_list[2*j+1].tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0)
    ax_list[2*j+1].set_yticks(np.arange(0.25,0.6,0.05))
    ax_list[2*j+1].set_xticks(np.arange(0,0.6,0.1))

fig.savefig("../Figs/Fig6_secular_ecc_%s.png" % star,bbox_inches='tight',dpi=300)
plt.close()