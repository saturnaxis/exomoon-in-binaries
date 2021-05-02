import rebound
import reboundx
import numpy as np
import threading
import os
import sys
import shutil
import time
from scipy.interpolate import interp1d


def get_params(t,ps):
    pl_idx = 0
    moon_idx = 1
    tot_mass = ps[moon_idx].m + ps[pl_idx].m
    r_m = [ps[moon_idx].x-ps[pl_idx].x,ps[moon_idx].y-ps[pl_idx].y,ps[moon_idx].z-ps[pl_idx].z]
    v_m = [ps[moon_idx].vx-ps[pl_idx].vx,ps[moon_idx].vy-ps[pl_idx].vy,ps[moon_idx].vz-ps[pl_idx].vz]
    h_m = [r_m[1]*v_m[2]-r_m[2]*v_m[1],r_m[2]*v_m[0]-r_m[0]*v_m[2],r_m[0]*v_m[1]-r_m[1]*v_m[0]]
    
    a_m = 1/(2./np.sqrt(np.dot(r_m,r_m)) - np.dot(v_m,v_m)/(G*tot_mass))
    e_m = np.sqrt(1.-np.dot(h_m,h_m)/(G*tot_mass*a_m))
    n_m = np.sqrt(G*tot_mass/a_m**3)*365.25
    return t/365.25,a_m,e_m,n_m,ps[2].n*365.25,ps[pl_idx].params["Omega"]*365.25
    
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

def simulation(a_sat,e_p,MA_sat,resume):

    if star == 'A':
        M_A = 1.133
        M_B = 0.972
        a_p = np.sqrt(1.519)
    else:
        M_A = 0.972
        M_B = 1.133
        a_p = np.sqrt(0.5)
    m_p = M_E
    m_sat = M_E/81.

    R_H = a_p*((m_p+m_sat)/(3.*M_A))**(1./3.)
    #R_p = 0.000477894503 #Radius of Jupiter in AU
    R_E = 4.26352e-5 #Radius of Earth in AU
    R_p = R_E
    R_s = 0.00465047 #Radius of Sun in AU
    
    a_Roche = 2.44*R_p*(5.515/3.34)**(1./3.)#8.64*R_p#/R_H
    a_sat = 3.*a_Roche

    out_str = ""
    AU = 1.495978707e13
    M_sol = 1.99e33
    pl_idx = 0
    if resume == 'F':
        dt_tp = np.sqrt((0.05*R_H)**3/(m_p+m_sat))/20. * 365.25
        e_p = 1.25*(a_p/23.78)*(0.524/(1.-0.524**2))
        
        temp_state = [a_sat,0.001,0.,0,0,MA_sat]
        
        Cart_pl, Cart_sat = Orb2Cart(temp_state,m_p,m_sat,"T")

        temp_state = [a_p,e_p,0.,0.,0.,0.]
        Cart_pCOM = Orb2Cart(temp_state,M_A,m_p+m_sat,"F")
        Cart_B = Orb2Cart([23.7,0.524,0,0,0,0],M_A,M_B,"F")
        
        #for i in xrange(1,len(Cart_sat)):
        #    Cart_pl[i] += Cart_pCOM[i]
        #    Cart_sat[i] += Cart_pCOM[i]
        for i in range(1,len(Cart_B)):
            Cart_B[i] += Cart_pCOM[i]

        sim = rebound.Simulation()
        sim.integrator = "ias15"
        
        #sim.integrator = 'whfast'
        sim.units = ('days', 'AU', 'Msun')
        sim.dt = dt_tp
        sim.ri_ias15.min_dt = np.sqrt(a_sat**3/(m_p+m_sat))/20. * 365.25
        sim.automateSimulationArchive("aCen%s_archive.bin" % star,interval=(1e3*365.25),deletefile=True)

        #sim.add(m=M_A)
        sim.add(m=m_p,x=Cart_pl[1],y=Cart_pl[2],z=Cart_pl[3],vx=Cart_pl[4],vy=Cart_pl[5],vz=Cart_pl[6]) #Planet
        sim.add(m=m_sat,x=Cart_sat[1],y=Cart_sat[2],z=Cart_sat[3],vx=Cart_sat[4],vy=Cart_sat[5],vz=Cart_sat[6]) #Moon
        sim.add(m=M_A,x=Cart_pCOM[1],y=Cart_pCOM[2],z=Cart_pCOM[3],vx=Cart_pCOM[4],vy=Cart_pCOM[5],vz=Cart_pCOM[6])
        sim.add(m=M_B,x=Cart_B[1],y=Cart_B[2],z=Cart_B[3],vx=Cart_B[4],vy=Cart_B[5],vz=Cart_B[6]) #Star B
        sim.move_to_com()
        ps = sim.particles
        
        #Add tidal evolution (constant time lag) from Reboundx
        rebx = reboundx.Extras(sim)
        tides = rebx.load_force("tides_constant_time_lag")
        rebx.add_force(tides)
        pl_idx = 0
        
        k2_p = 0.299
        ps[pl_idx].params["tctl_k1"] = k2_p/2.
        
        Op0 = (2.*np.pi)/(5./24.) #5 hr rotation rate
        n_p = np.sqrt(G*(M_A+m_p+m_sat)/a_p**3)

        ps[pl_idx].r = R_p
        ps[pl_idx].params["tctl_tau"] = 100./86400.
        ps[pl_idx].params["Omega"] = Op0

        tcol = tscale
        t_sim = sim.dt
        t_cnt = 1
        run = True
        curr_time,a_m,e_m,n_m,n_p,O_p = get_params(sim.t,ps)
        write2file(out_fname,"%1.5e,%1.5e,%1.5e,%1.5e,%1.5e,%1.5e\n" % (curr_time,a_m/R_H,e_m,n_m,n_p,O_p),'foot')
        times = np.concatenate((np.arange(0,1000,100),np.arange(1000,1e6,1000),np.arange(1e6,tscale+1e6,1e6)))
        times *= 365.25

        #while run:
        for tstep in times[1:]:
            sim.integrate(tstep)
            ps[pl_idx].params["Omega"] = f_rot(sim.t) 
            curr_time,a_m,e_m,n_m,n_p,O_p = get_params(sim.t,ps)
            write2file(out_fname,"%1.5e,%1.5e,%1.5e,%1.5e,%1.5e,%1.5e\n" % (np.round(curr_time,-2),a_m/R_H,e_m,n_m,n_p,O_p),'foot')
    else:
        #create backup file before restart
        shutil.copy("aCen%s_archive_restart.bin" % star,"aCen%s_archive_backup.bin" % star)
        sim = rebound.Simulation("aCen%s_archive_backup.bin" % star)
        #Add tidal evolution (constant time lag) from Reboundx
        rebx = reboundx.Extras(sim)
        tides = rebx.load_force("tides_constant_time_lag")
        rebx.add_force(tides)
        ps = sim.particles
        k2_p = 0.299
        ps[pl_idx].params["tctl_k1"] = k2_p/2.
        ps[pl_idx].r = R_p
        ps[pl_idx].params["tctl_tau"] = 100./86400.
        ps[pl_idx].params["Omega"] = f_rot(sim.t)
        
        curr_time,a_m,e_m,n_m,n_p,O_p = get_params(sim.t,ps)
        dt_tp = (2.*np.pi/n_m)/20.*365.25
        sim.automateSimulationArchive("aCen%s_archive_restart.bin" % star,interval=(1e3*365.25),deletefile=True)

        
        times = np.concatenate((np.arange(0,1000,100),np.arange(1000,1e6,1000),np.arange(1e6,tscale+1e5,1e5)))
        times *= 365.25
        
        t_cut = np.where(times > sim.t)[0]
        times = times[t_cut]

        for tstep in times:
            sim.integrate(tstep)
            ps[pl_idx].params["Omega"] = f_rot(sim.t) 
            curr_time,a_m,e_m,n_m,n_p,O_p = get_params(sim.t,ps)
            write2file(out_fname,"%1.5e,%1.5e,%1.5e,%1.5e,%1.5e,%1.5e\n" % (np.round(curr_time,-2),a_m/R_H,e_m,n_m,n_p,O_p),'foot')
        


def write2file(fname,f_str,head):
    lock.acquire() # thread blocks at this line until it can obtain lock
    if head == "head":
            f = open(output_fldr+fname,'w')
    else:
        f = open(output_fldr+fname,'a')
    f.write(f_str)
    lock.release()


G = (2.*np.pi/365.25)**2.
M_J = 9.54e-4
M_E = 3.0035e-6
M_N = 15.*M_E
star = sys.argv[1]
resume = sys.argv[2]

secular_data = np.genfromtxt("../data/Fig7/acen%s_secular_100.txt" % star,delimiter=',',comments='#')
f_rot = interp1d(secular_data[:,0]*365.25, secular_data[:,5]/365.25)
home = os.getcwd() + "/"
output_fldr = home
lock = threading.Lock()


tscale = 1e7#yrs
#star = 'A'
out_fname = "aCen%s_reboundx.txt" % star
if resume == 'F':
    write2file(out_fname,"#time,a_sat,e_sat,n_sat,n_p,Omega_p\n",'head')
else:
    shutil.copy(out_fname,out_fname[:-4]+"_backup.txt")
    out_fname = "aCen%s_reboundx_backup.txt" % star
simulation(0.07,0.,0.,resume)
