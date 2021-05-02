import numpy as np
from scipy.interpolate import griddata,interp2d
from scipy.optimize import root_scalar
import sys
import os
import multiprocessing as mp
from rebound.interruptible_pool import InterruptiblePool
import threading

def get_stab_func(incl):
    data = np.genfromtxt("a_crit_Incl[%i].txt" % incl ,delimiter=',',comments='#')
    X_data = data[:,0]
    Y_data = data[:,1]
    Z_data = data[:,2]
    xi_data = np.concatenate(([0.001],np.arange(0.01,1,0.01),[0.999]))
    yi_data = np.arange(0,0.81,0.01)
    zi_data = griddata((X_data,Y_data),Z_data,(xi_data[None,:],yi_data[:,None]),method = 'linear',fill_value=0)

    f_data = interp2d(xi_data, yi_data, zi_data, kind='linear')
    return f_data

def f_stab(mu,ebin,a_frac,incl):
    #retrograde moons, but prograde planets
    temp = f_data_pro(mu,ebin)[0]
    if temp >= a_frac:
        return 1
    else:
        return -1

def get_A1(gamma1,gamma2,FT):
    term1 = (1./(gamma1+1.))*(0.3**(gamma1+1.)-0.1**(gamma1+1.))
    term2 = (0.3**(gamma1-gamma2)/((1.-FT)*(gamma2+1.)))*(1.-0.3**(gamma2+1))
    return 1./(term1+term2)

def get_A2(gamma1,gamma2,FT):
    pdf_A1 = get_A1(gamma1,gamma2,FT)
    return pdf_A1*0.3**(gamma1-gamma2)

def get_Aex(gamma1,gamma2,FT):
    pdf_A2 = get_A2(gamma1,gamma2,FT)
    return (FT*pdf_A2/((1.-FT)*(gamma2+1.)))*(1.-0.3**(gamma2+1.))

def q_pdf(x,gamma1,gamma2,FT):
    pdf = np.zeros(len(x))
    for i in range(0,len(x)):
        if x[i]>=0.1 and x[i]<=0.3:
            pdf[i] = get_A1(gamma1,gamma2,FT)*x[i]**gamma1
        elif x[i]>0.3 and x[i]<=1.:
            pdf[i] = get_A2(gamma1,gamma2,FT)*x[i]**gamma2
            if x[i]>= 0.95:
                pdf[i] += get_Aex(gamma1,gamma2,FT)/0.05
    return pdf

def per_pdf(x,C,mu,sigma):
    pdf = (C/(sigma*np.sqrt(2.*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
    return pdf

def get_Norm_per(mu,sigma,step):
    per = per_pdf(logP,1,mu,sigma)
    return 1./np.sum(step*per[1:])

def per_cdf(C,mu,sigma):
    P_pdf = np.array([per_pdf(p,C,mu,sigma) for p in logP])
    cdf = np.cumsum(P_pdf)/np.sum(P_pdf)
    return cdf

def get_rvs(x,cdf,pdf):
    draw_param = True
    while draw_param:
        cdf_samp = np.round(np.random.uniform(0,1),3)
        samp_idx = np.where(np.abs(cdf-cdf_samp)<0.001)[0]
        if len(samp_idx) > 1:
            samp_idx = samp_idx[0]
            draw_param = False
    return np.round(x[samp_idx],3),cdf[samp_idx],pdf[samp_idx]

def get_power_rvs(gamma,C):
    val = np.random.uniform(0,1)
    return ((gamma+1)/C*val)**(1./(gamma+1))

def mass_func(x,f_c,a_p,k2p,M_p,R_p,Q_p,M_star,Tmax):
    a_c = f_c*a_p*(M_p*(1+x)/(3.*M_star))**(1./3.)
    a_o = 3.*2.88*R_p
    Q_fact = Q_p/(3.*k2p*R_p**5*Tmax*np.sqrt(G*M_p))
    return x*np.sqrt(1.+x) - (2./13.)*(a_c**(13./2.)-a_o**(13./2.))*Q_fact

def get_sample(par_idx):
    i_m,star,n_samp = params[par_idx]
    #ensure unique draws per core
    job_int = (int(n_samp)+1)
    current = mp.current_process()
    cpuid = (current._identity[0]+1)
    init_seed = int(par_idx+cpuid*1e5+job_int)
    np.random.seed(init_seed)
    
    #draw_param = True
    #while draw_param:
    #draw from period distribution
    logP_samp, Pcdf_samp, Ppdf_samp = get_rvs(logP,P_cdf,P_pdf)
    per = 10**logP_samp/365.25
    
    #Ecc distribution from Moe & Di Stefano 2017
    e_gam1 = np.random.normal(0.4,0.3)
    C_ebin = (e_gam1+1)/(eb_max**(e_gam1+1))
    e_bin = get_power_rvs(e_gam1,C_ebin)

    L_star = M_A**4
    M_star = M_A

    #q (mass ratio) distribution from Moe & Di Stefano 2017 (needed for star B)
    q_gam2 = np.random.normal(-0.5,0.3)
    q_gam1 = np.random.normal(0.3,0.4)
    FTwin = np.random.normal(0.1,0.03)
    qpdf = q_pdf(q_rng,q_gam1,q_gam2,FTwin)
    qcdf = np.cumsum(qpdf)/np.sum(qpdf)
    q, qcdf_samp, qpdf_samp = get_rvs(q_rng,qcdf,qpdf)

    z = q/(1.+q) #convert q to mu
    
    if star == 1: #these are needed if orbiting star B
        z = 1.-z
        M_star = M_A*q
        if q > 0.43:
            L_star = M_star**4
        else:
            L_star = 0.23*M_star**2.3
    a_bin = (per**2*M_A*(1.+q))**(1./3.)
    a_p = np.sqrt(L_star) #semimajor axis of 1 Earth Flux
    a_frac = a_p/a_bin

    y2 = f_stab(z,e_bin,a_frac,i_m)

    if y2 > 0 :
        #calculate a_crit,m so that we can determine the maximum satellite mass from tides
        R_H = a_p*(m_p/(3.*M_star))**(1./3.)
        eps = 1.25*a_frac*(e_bin/(1.-e_bin**2))
        eps /= 1.+(25./8.)*z/np.sqrt(1+z)*a_frac**1.5*(3.+2.*e_bin**2)/(1.-e_bin**2)**1.5 #correction (Andrade-Ines & Eggl 2017)
        eta = np.abs(e_p - eps)
        a_crit = (1.- eps - eta)*R_H
        if i_m == 0:
            f_crit = 0.4
        else:
            f_crit = 0.65
        a_crit *= f_crit
        for k in range(0,50):
            a_sat = np.random.uniform(0.05,f_crit)*R_H#f_crit*R_H
            #estimate max satellite mass
            try:
                sol = root_scalar(mass_func,args=(a_sat/R_H,a_p,0.299,m_p,R_p,Q_p,M_star,T),method='bisect',bracket=[1e-3,0.3],xtol=1e-6)
                max_msat = sol.root
            except ValueError:
                max_msat = 0.1*m_p
            if max_msat > 0.1*m_p:
                max_msat = 0.1*m_p

            if max_msat > 0.001*m_p:
                M_frac = np.random.uniform(-3,np.log10(max_msat/m_p))
            else:
                M_frac = np.random.uniform(np.log10(max_msat/m_p),np.log10(max_msat/m_p)+2)
            M_sat = 10**M_frac*m_p

            #calculate TTV amplitude
            P_p = np.sqrt(a_p**3/M_star)*365.25*24.*60.
            TTV_max = np.sqrt(a_p)*a_sat*(M_sat/(m_p+M_sat))/np.sqrt(G*(M_star+m_p+M_sat)) #Kipping 2009 (Eqn A27 circular)
            TTV_max *= 365.25*24.*60. #convert years to minutes

            f_str = "%i,%i,%1.5f,%2.5f,%1.5f,%1.5f,%1.5e,%1.5e,%1.5e\n" % (star,i_m,a_frac,TTV_max,z,e_bin,a_bin,a_sat/R_H,M_sat/m_p)
            write2file(f_str,'foot')



def write2file(f_str,head):
    lock.acquire() # thread blocks at this line until it can obtain lock
    if head == "head":
        f = open(output_fldr+fname,'w')
    else:
        f = open(output_fldr+fname,'a')
    f.write(f_str)
    lock.release()

lock = threading.Lock()
eb_min = 0    #min e_bin
eb_max = 0.8  #max e_bin
q_min = 0.3  #min q
q_max = 1.0  #max q
output_fldr = os.getcwd() + "/"

f_data_pro = get_stab_func(0)
f_data_ret = get_stab_func(180)


#Period distribution for binaries from Raghavan+ 2010
P_min = 4
P_max = 7
logP = np.arange(P_min,P_max+0.001,0.001)
mu = 5.03
sigma = 2.28
P_norm = get_Norm_per(mu,sigma,0.001)
P_pdf = per_pdf(logP,P_norm,mu,sigma)
P_cdf = per_cdf(P_norm,mu,sigma)

#stellar mass ratio (q= M_B/M_A)
q_min = 0.1
q_max = 1.
q_rng = np.arange(q_min,q_max+0.001,0.001)

ebin_rng = np.arange(0,0.8,0.001)

G = 4.*np.pi**2
M_A = float(sys.argv[1])
e_p = 0.
i_m = 0. # change to i_m = 180. for retrograde moons
m_p = 3.0035e-6 #Earth mass
Q_p = 33.
k2_p = 0.299
R_p = 4.26352e-5 #Earth Radius in AU
T = 1e10 #Moon lifetime in yr

fname = "MC_sample_%1.2f.txt" % M_A
write2file("#star,inc,a_HZ/abin, max TTV,mu,e_bin,a_bin,M_sat\n",'head')
a_f = np.arange(0.001,0.1001,0.001)
host = [0,1] #index 0 = A; index 1 = B
params = []
for m in range(0,2): #prograde or retrograde
    for h in range(0,2): #host star A or B
        #for af in a_f:
        for n_samp in range(0,10000): #sample index
            params.append((180*m,h,n_samp))
        
pool = InterruptiblePool(processes=16)

N_steps = len(params)
pool.map(get_sample,xrange(0,N_steps))
pool.close()
