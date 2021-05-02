from scipy.integrate import solve_ivp,odeint
from scipy.optimize import minimize,fmin
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import threading
import multiprocessing as mp
import time

home = os.getcwd() + "/"

#constants
G = 4.*np.pi**2
M_sun = 1.989e33 #Mass of Sun in g
AU = 149597870700. #AU in m
AU_cm = AU*100.

M_E = 3.0035e-6 #M_sun
M_J = 9.54e-4 #M_sun
R_E = 4.2635e-5 #AU
rho_E = 5.515 # g/cm^3
rho_Mars = 3.93
rho_Moon = 3.34
rho_N = 1.64 # g/cm^3
rho_J = 1. # g/cm^3
rho_moon = 3.34 #g/cm^3

def write2file(fname,f_str,head):
    lock.acquire() # thread blocks at this line until it can obtain lock
    if head == "head":
            f = open(output_fldr+fname,'w')
    else:
        f = open(output_fldr+fname,'a')
    f.write(f_str)
    lock.release()

def f_ecc(f_idx,ecc):
    if f_idx == 1:
        return 1. + 31./2.*ecc**2 + 255./8.*ecc**4 + 185./16.*ecc**6 + 25./64.*ecc**8
    elif f_idx == 2:
        return 1. + 15./2.*ecc**2 + 45./8.*ecc**4 + 5./16.*ecc**6
    elif f_idx == 3:
        return 1. + 15./4.*ecc**2 + 15./8.*ecc**4 + 5./64.*ecc**6
    elif f_idx == 4:
        return 1. + 3./2.*ecc**2 + 1./8.*ecc**4
    elif f_idx == 5:
        return 1. + 3.*ecc**2 + 3./8.*ecc**4

def deriv_t(t,y,k2p,tau_p,M_p,R_p,M_m,M_star,alpha): 
    #y = [n_m,e_m,n_p,e_p,O_p]
    #secular tidal evolution (Hut 1980)

    #calculate factors
    dnm_fact = 9.*(k2p*tau_p*R_p**5)*(M_m/M_p)/(G*(M_p+M_m))**(5./3.)
    dem_fact = -27.*(k2p*tau_p*R_p**5)*(M_m/M_p)/(G*(M_p+M_m))**(5./3.)
    em_fact = (1.-y[1]**2)
    
    dnp_fact = 9.*(k2p*tau_p*R_p**5)*(M_star/(M_p+M_m))/(G*(M_star+M_p+M_m))**(5./3.)
    dep_fact = -27.*(k2p*tau_p*R_p**5)*(M_star/(M_p+M_m))/(G*(M_star+M_p+M_m))**(5./3.)
    ep_fact = (1.-y[3]**2)
    
    dOp_fact1 = 3.*(k2p*tau_p*R_p**3/alpha)*(G*M_p)*(M_m/M_p)**2/(G*(M_p+M_m))**2
    dOp_fact2 = 3.*(k2p*tau_p*R_p**3/alpha)*(G*(M_p+M_m))*(M_star/(M_p+M_m))**2/(G*(M_star+M_p+M_m))**2

    
    dnmdt = dnm_fact*y[0]**(19./3.)/em_fact**(15./2.)*(f_ecc(1,y[1])-em_fact**1.5*f_ecc(2,y[1])*y[4]/y[0])
    demdt = dem_fact*y[0]**(16./3.)*y[1]/em_fact**1.5*(f_ecc(3,y[1])-(11./18.)*em_fact**1.5*f_ecc(4,y[1])*y[4]/y[0])
    
    dnpdt = dnm_fact*y[2]**(19./3.)/ep_fact**(15./2.)*(f_ecc(1,y[3])-ep_fact**1.5*f_ecc(2,y[3])*y[4]/y[2])
    depdt = dep_fact*y[2]**(16./3.)*y[3]/ep_fact**1.5*(f_ecc(3,y[3])-(11./18.)*ep_fact**1.5*f_ecc(4,y[3])*y[4]/y[2])
    
    dOpdt = dOp_fact1*y[0]**5/em_fact**6*(f_ecc(2,y[1])-em_fact**1.5*f_ecc(5,y[1])*y[4]/y[0])
    dOpdt += dOp_fact2*y[2]**5/ep_fact**6*(f_ecc(2,y[3])-ep_fact**1.5*f_ecc(5,y[3])*y[4]/y[2])

    return [dnmdt,demdt,dnpdt,depdt,dOpdt]

def deriv_Q(t,y,k2p,Q_p,M_p,R_p,M_m,M_star,alpha): 
    #y = [n_m,e_m,n_p,e_p,O_p]
    #secular tidal evolution (Goldriech & Soter 1966)

    #calculate factors
    dnm_fact = -(9./2.)*(k2p*R_p**5/Q_p)*(M_m/M_p)/(G*(M_p+M_m))**(5./3.)
    dem_fact = (171./16.)*(k2p*R_p**5/Q_p)*(M_m/M_p)/(G*(M_p+M_m))**(5./3.)  #Goldreich & Soter 1966

    dnp_fact = -(9./2)*(k2p*R_p**5/Q_p)*(M_star/(M_p+M_m))/(G*(M_star+M_p+M_m))**(5./3.)
    dep_fact = (171./16.)*(k2p*R_p**5/Q_p)*(M_star/(M_p+M_m))/(G*(M_star+M_p+M_m))**(5./3.)

    dOp_fact1 = -(3./2.)*(k2p*R_p**3/alpha/Q_p)*G*M_p*(M_m/M_p)**2/(G*(M_p+M_m))**2
    dOp_fact2 = -(3./2.)*(k2p*R_p**3/alpha/Q_p)*G*(M_p+M_m)*(M_star/(M_p+M_m))**2/(G*(M_star+M_p+M_m))**2
    
    D_m,E_m,F_m = spin_coeff(y[4],y[0])
    D_p,E_p,F_p= spin_coeff(y[4],y[2])
    
    dnmdt = dnm_fact*y[0]**(16./3.)*(np.sign(y[4]-y[0]))
    demdt = dem_fact*y[0]**(13./3.)*np.sign(2*y[4]-3*y[0])
    dOpdt = dOp_fact1*y[0]**4*(np.sign(y[4]-y[0]))#+y[1]**2*D_m)
    
    dnpdt = dnm_fact*y[2]**(16./3.)*(np.sign(y[4]-y[2]))#+y[3]**2*E_p)
    depdt = dep_fact*y[2]**(13./3.)*np.sign(2*y[4]-3*y[2])
    dOpdt += dOp_fact2*y[2]**4*(np.sign(y[4]-y[2]))#+y[3]**2*D_p)

    return [dnmdt,demdt,dnpdt,depdt,dOpdt]

def deriv_ep(t,y,M_p,M_star):
    #y = [np,ep]

    if M_star == 1.133:
        M_B = 0.972
    else:
        M_B = 1.133
    semi_p = (G*(M_star+M_p)/y[0]**2)**(1./3.)
    g_H = 0.75*(M_B/M_star)*(semi_p/23.78)**3*y[0]/(1.-0.524**2)**1.5
    e_H = 1.25*(semi_p/23.78)*(0.524/(1.-0.524**2))
    depdt = e_H*g_H*np.sin(g_H*t)
    
    return depdt


def calc_Tide_evolution(M_sat,a_p,e_p,M_p,M_star,R_p,k2p,alpha,t_fin,star):
    #t_fin = tscale
    times = np.concatenate((np.arange(0,1000,100),np.arange(1000,1e6,1000),np.arange(1e6,t_fin+1e6,1e6)))
    nsteps = len(times)
    
    R_H = a_p*((M_p+M_sat)/(3.*M_star))**(1./3.)
    a_Roche = 2.88*R_p #Roche Limit

    a_m = 3.*a_Roche#initial semimajor axis of moon (AU)
    e_m = 0.001
    
    n_m = np.sqrt(G*(M_p+M_sat)/a_m**3) #mean motion of moon in rad/yr
    a_crit = 0.4061*R_H
    n_crit = np.sqrt(G*(M_p+M_sat)/a_crit**3)
    n_p = np.sqrt(G*(M_star+M_p+M_sat)/a_p**3) #mean motion of planet in rad/yr
    T_p = 5./24./365.25 # rotation period in yr (Kokubo & Ida 2007)
    O_p = 2.*np.pi/T_p  #rotation rate of planet in rad/yr

    eps = 1.25*a_p/23.78*(0.524/(1.-0.524**2))
    
    
    tau_p = 100./3600./24./365.25
    Q_p = 1./(n_m*tau_p)


    moon_escape = False
    for i in range(0,nsteps-1):
        t_i = times[i]
        t_n = times[i+1]

        delta_t = [t_i,t_n]
        if tid_mod == 't':
            temp = solve_ivp(deriv_t,delta_t,[n_m,e_m,n_p,e_p,O_p],method='RK45',atol=1e-12,rtol=1e-12,args=(k2p/2.,tau_p,M_p,R_p,M_sat,M_star,alpha))
        else:
            temp = solve_ivp(deriv_Q,delta_t,[n_m,e_m,n_p,e_p,O_p],method='RK45',atol=1e-12,rtol=1e-12,args=(k2p,Q_p,M_p,R_p,M_sat,M_star,alpha))

        if np.abs(temp.y[0,-1]-n_m)<1e-8:
            moon_escape = True
            break
        n_m,e_m,n_p,e_p,O_p = temp.y[0,-1],temp.y[1,-1],temp.y[2,-1],temp.y[3,-1],temp.y[4,-1]
        a_m = (G*(M_p+M_sat)/n_m**2)**(1./3.)
        a_crit = 0.4*(1.-eps-np.abs(e_p-eps))*R_H

        if a_m*(1.-e_m) < R_p: #check if q < R_p
            moon_escape = True
            break
        if a_m*(1.+e_m) > a_crit: #check if Q > a_crit
            moon_escape = True
            break

    if moon_escape:
        #print(M_sat/M_p,t_i/t_fin)
        out_stg = "%1.2f,%1.4f,%1.4f,%1.4f,%1.5e\n" % (e_p,M_sat/M_p,a_m/R_H,e_m,t_i)
        write2file(fname,out_stg,'foot')
    else:
        out_stg = "%1.2f,%1.4f,%1.4f,%1.4f,%1.5e\n" % (e_p,M_sat/M_p,a_m/R_H,e_m,t_fin)
        write2file(fname,out_stg,'foot')
        
def simulation(par):
    e_pl,m_sat = par
    calc_Tide_evolution(m_sat*M_E,ap,e_pl,M_E,M_A,R_E,0.299,0.3308,T_max,star)

lock = threading.Lock()
T_max = 1e10
star = sys.argv[1] #A or B
tid_mod = sys.argv[2] # constant Q or t
if star == 'A':
    M_A = 1.133
    L_A = 1.519
else:
    M_A = 0.972
    L_A = 0.5
ap = np.sqrt(L_A)

output_fldr = os.getcwd() + "/"
fname = "mass_stab_%s_%s.txt" % (star,tid_mod)

m_range = np.arange(0.001,0.201,0.001) #range in satellite mass
ep_range = np.arange(0.,0.61,0.01) #range in planetary eccentricity
parameters = []
for m in m_range:
    for ep in ep_range:
        parameters.append((ep,m))

header = "#e_p,M_sat (M_p),final a_sat (R_H),final e_sat, lifetime\n"
write2file(fname,header,'head')


pool = mp.Pool(processes=20)
jobs = []
for p in range(0,len(parameters)):
    param = parameters[p]
    job = pool.apply_async(simulation, (param,))
    jobs.append(job)


t0 = time.time()
for j in range(0,len(jobs)): 
    job.get()
tf = time.time()
print("simulation took %1.4f min" % ((tf-t0)/60.))