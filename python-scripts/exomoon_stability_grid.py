import rebound
import numpy as np
from rebound.interruptible_pool import InterruptiblePool
import multiprocessing as mp
import time
import sys
import os
import shutil
import random
import subprocess as sb
import threading

def generate_orb_elements(semi):
    ecc = np.abs(np.random.normal(0,0.3))
    incl = np.abs(np.random.normal(0,1))
    omg = np.random.rand()*360.
    RA = np.random.rand()*360.
    MA = np.random.rand()*360.
    return np.array([semi,ecc,incl,omg,RA,MA])

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

#def simulation(par):
def simulation(a_sat,e_p,MA_sat):

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

    out_str = ""
    AU = 1.495978707e13
    M_sol = 1.99e33

    ap = 0.25*R_H
    dt_tp = np.sqrt(ap**3/ms)/40. * 365.25

    #temp_state = generate_orb_elements(ap)
    i_m = 0.
    if retro == 'T':
        i_m = 180.
    temp_state = [a_sat*R_H,0,i_m,0,0,MA_sat]
    Cart_Jup, Cart_moon = Orb2Cart(temp_state,ms,M_E,"T")

    #temp_state = generate_orb_elements(ac)
    temp_state = [a_p,e_p,0.,0.,0.,0.]
    Cart_JCOM = Orb2Cart(temp_state,M_A,ms,"F")
    if star == 'A' or star == 'B':
        Cart_B = Orb2Cart([23.7,0.524,0,0,0,0],M_A,M_B,"F")
    out_str = "%1.3f,%1.8f,%1.2f,%3.5f," % (a_sat,a_sat*R_H,e_p,MA_sat)

    for i in xrange(1,len(Cart_moon)):
        Cart_Jup[i] += Cart_JCOM[i]
        Cart_moon[i] += Cart_JCOM[i]

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.units = ('days', 'AU', 'Msun')
    sim.dt = dt_tp

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

    tf = time.time()
    time_stg = "simulation took %1.4f min\n" % ((tf-t0)/60.)
    write2file("output_moon%s.txt" % jobid,time_stg,'foot')

    out_str += "%2.5f\n" % tcol
    write2file(fname,out_str,'foot')



def write2file(fname,f_str,head):
    lock.acquire() # thread blocks at this line until it can obtain lock
    if head == "head":
            f = open(output_fldr+fname,'w')
    else:
        f = open(output_fldr+fname,'a')
    f.write(f_str)
    lock.release()    

G = (2.*np.pi)**2.
M_J = 9.54e-4
M_E = 3.0035e-6
M_N = 15.*M_E

home = os.getcwd() + "/"


tscale = float(sys.argv[1]) #yrs
jobid = sys.argv[2]
a_p = float(sys.argv[3]) #planet semimajor axis
star = sys.argv[4] #A, B, or S
retro = sys.argv[5] # T (i_m = 180 deg); F (i_m = 0 deg)

fname = "Stab_moon%03i.txt" % int(jobid)
if star == 'A' or star == 'B':
    output_fldr = home + "aCen%s_moon_circ_%1.2f" % (star,a_p)
else:
    output_fldr = home + "GenRuns"
if retro == 'T':
    output_fldr += "_retro/"
else:
    output_fldr += "/"
if not os.path.exists(output_fldr):
    os.makedirs(output_fldr)

data = np.zeros(1)
if os.path.exists(output_fldr+fname):
    data = np.genfromtxt(output_fldr+fname,delimiter=',',comments='#')


np.random.seed(42)
parameters = []

for a in np.arange(0.25,0.56,0.01):
    for e in np.arange(0,0.61,0.01):
        for m in range(0,20):
            MA = np.round(np.random.uniform(0,180.),6)
            if len(data.shape)>1:
                in_data = 0
                e_data = np.where(np.abs(e-data[:,2])<1e-6)
                MA_data = np.where(np.abs(MA-data[e_data,3])<1e-4)[0]
                if len(MA_data)>0:
                    in_data = 1
                if in_data==0:
                    parameters.append((a,e,MA))
            else:
                parameters.append((a,e,MA))

if len(parameters) == 0:
    print("All jobs finished")
    sys.exit(0)

p_list = np.arange(0,len(parameters),dtype='i8')
np.random.shuffle(p_list)

print("Total number of jobs: %i " % len(p_list))


lock = threading.Lock()
if not os.path.exists(output_fldr+fname):
    header = "#nR_H,a_s,e_s,MA_s,tcol\n"
    write2file(fname,header,'head')
    write2file("output_moon%s.txt" % jobid,'','head')

pool = InterruptiblePool(processes=20)
pool.map(simulation,p_list)
pool.close()