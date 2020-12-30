import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import erf,erfinv
from scipy.interpolate import griddata,interp2d
from scipy.stats import truncnorm
import sys
import os

rcParams.update({'font.size': 26})
rcParams.update({'mathtext.fontset': 'cm'})

#get_data = sys.argv[1]

def colorbar(mappable,orient):
    last_axes = plt.gca()
    ax = mappable.axes
    fig1 = ax.figure
    divider = make_axes_locatable(ax)
    if orient == 'right':
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig1.colorbar(mappable, cax=cax,orientation='vertical')
        cbar.ax.yaxis.set_ticks_position('right')
    else:
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig1.colorbar(mappable, cax=cax,orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
    cbar.set_ticks([10,100])
    cbar.set_ticklabels(['10','100'])
    cbar.ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    cbar.ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)
    plt.sca(last_axes)
    return cbar

def get_counts(xdata,ydata,max_y):
    x_bins = np.arange(0.001,0.1+0.001,0.001)
    y_bins = np.arange(0.1,max_y+0.1,0.1)
    X = []
    Y = []
    Z = []
    norm = 0
    for i in range(0,len(x_bins)-1):
        x_cut = np.where(np.logical_and(x_bins[i]<xdata,xdata<x_bins[i+1]))[0]
        for j in range(0,len(y_bins)-1):
            X.append(x_bins[i])
            Y.append(y_bins[j])
            bin_cnt = np.where(np.logical_and(y_bins[j]<ydata[x_cut],xdata[x_cut]<y_bins[j+1]))[0]
            if len(bin_cnt) > 0:
                Z.append(len(bin_cnt))
                norm += len(bin_cnt)
            else:
                Z.append(0)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)/float(norm)
    #print(len(X),len(Y),len(Z))
    return X,Y,Z

def f_stab(mu,ebin,a_frac):
    temp = f_data(mu,ebin)[0]
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
                #pdf_q += FTwin/0.05
                pdf[i] += get_Aex(gamma1,gamma2,FT)/0.05
    return pdf
    
'''def q_cdf(gamma1,gamma2,FT):
    qpdf = np.array([q_pdf(q,gamma1,gamma2,FT) for q in q_rng])
    cdf = np.cumsum(qpdf)/np.sum(qpdf)
    return cdf'''

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
    cdf_samp = np.round(np.random.uniform(0,1),3)
    samp_idx = np.where(np.abs(cdf-cdf_samp)<0.001)[0]
    if len(samp_idx) > 1:
        samp_idx = samp_idx[0]
    return np.round(x[samp_idx],3),cdf[samp_idx],pdf[samp_idx]

def get_power_rvs(gamma,C):
    val = np.random.uniform(0,1)
    return ((gamma+1)/C*val)**(1./(gamma+1))

def get_Gauss_rvs(mu,sigma,C):
    val = np.random.uniform(0,1)
    return (np.sqrt(2)*sigma/C)*erfinv(2.*val-1) + mu

def get_sample(f,star):
    #f = planetary stability function
    
    #draw from period distribution
    logP_samp, Pcdf_samp, Ppdf_samp = get_rvs(logP,P_cdf,P_pdf)
    per = 10**logP_samp/365.25
    
    #Ecc distribution from Moe & Di Stefano 2017
    e_gam1 = np.random.normal(0.4,0.3)
    #e_gam1 = truncnorm.rvs(a=0.1,b=0.7,loc=0.4,scale=0.4)
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
    if not isinstance(q,float):
        if q.size == 0:
            return -1,-1,-1,-1,-1,-1
    z = q/(1.+q) #convert q to mu
    
    if star == 'B': #these are needed if orbiting star B
        z = 1.-z
        M_star = M_A*q
        if q > 0.43:
            L_star = M_star**4
        else:
            L_star = 0.23*M_star**2.3
    a_bin = (per**2*M_A*(1.+q))**(1./3.)
    a_p = np.sqrt(L_star) #semimajor axis of 1 Earth Flux
    a_frac = a_p/a_bin
    #print(z,e_bin,a_frac)
    y2 = f(z,e_bin,a_frac)
    if y2 > 0 :
        #calculate a_crit,m so that we can determine the maximum satellite mass from tides
        R_H = a_p*(m_p/(3.*M_star))**(1./3.)
        eps = 1.25*a_frac*(e_bin/(1.-e_bin**2))
        eta = np.abs(e_p - eps)
        a_crit = (1.- eps - eta)*R_H
        if i_m == 0:
            a_crit *= 0.38
        else:
            a_crit *= 0.55
        #max satellite mass
        M_sat = (4./13.)*a_crit**(13./2.)*Q_p/(3.*k2_p*T*R_p**5*np.sqrt(G/m_p))
        if M_sat > 0.1*m_p:
            M_sat = 0.1*m_p
        #calculate max TTV amplitude
        P_p = np.sqrt(a_p**3/M_star)*365.25*24.*60.
        if i_m == 0:
            f_crit = 0.38
        else:
            f_crit = 0.55
        #TTV_max = f_crit/(2.*np.pi)*P_p*(M_sat/(m_p+M_sat))*(m_p/(3*M_star))**(1./3.)
        a_sat = f_crit*R_H
        TTV_max = np.sqrt(a_p)*a_sat*(M_sat/(m_p+M_sat))/np.sqrt(G*(M_star+m_p+M_sat)) #Kipping 2009 (Eqn A27 circular)
        TTV_max *= 365.25*24.*60. #convert years to minutes
        return a_frac,TTV_max,z,e_bin,a_bin,M_sat/m_p
    else:
        return -1,-1,-1,-1,-1,-1
        


eb_min = 0    #min e_bin
eb_max = 0.8  #max e_bin
q_min = 0.3  #min q
q_max = 1.0  #max q

data = np.genfromtxt("a_crit_Incl[0].txt" ,delimiter=',',comments='#')

X_data = data[:,0]
Y_data = data[:,1]
Z_data = data[:,2]
xi_data = np.concatenate(([0.001],np.arange(0.01,1,0.01),[0.999]))
yi_data = np.arange(0,0.81,0.01)
zi_data = griddata((X_data,Y_data),Z_data,(xi_data[None,:],yi_data[:,None]),method = 'linear',fill_value=0)

f_data = interp2d(xi_data, yi_data, zi_data, kind='linear')

#Period distribution for binaries from Raghavan+ 2010
P_min = 4
P_max = 7
logP = np.arange(P_min,P_max+0.001,0.001)
mu = 5.03
sigma = 2.28
P_norm = get_Norm_per(mu,sigma,0.001)
P_pdf = per_pdf(logP,P_norm,mu,sigma)
P_cdf = per_cdf(P_norm,mu,sigma)
#pdf_P = get_Norm_per(5.03,2.28)

q_min = 0.1
q_max = 1.
q_rng = np.arange(q_min,q_max+0.001,0.001)

ebin_rng = np.arange(0,0.8,0.001)

G = 4.*np.pi**2
M_st = int(sys.argv[1])
if M_st == 0:
    M_A = 1.2
    ymax = [45,45,65,65]
elif M_st == 1:
    M_A = 1.
    ymax = [30,30,50,50]
else:
    M_A = 0.8
    ymax = [20,20,30,30]
#M_A = float(sys.argv[1])
e_p = 0.
i_m = 0. # change to i_m = 180. for retrograde moons
m_p = 3.0035e-6 #Earth mass
Q_p = 10.
k2_p = 0.299
R_p = 4.26352e-5 #Earth Radius in AU
T = 1e10 #Moon lifetime in yr

fname = "MC_sample_%1.2f.txt" % M_A

'''if get_data == 'T':
    out = open(fname,'w')
    out.write("#star,inc,a_HZ/abin, max TTV,mu,e_bin,a_bin,M_sat\n")
    out.close()
    for m in range(0,2):
        if m ==1:
            i_m = 180.
        else:
            i_m = 0.
        host = ['A','B'] #index 0 = A; index 1 = B
        color = ['k','r']
        for k in range(0,2):
            for n_samp in range(0,10000):
                x_samp,y_samp,z,e_bin,a_bin,M_sat = get_sample(f_stab,host[k])
                if x_samp > 0 and y_samp > 0.5:
                    out = open(fname,'a')
                    out.write("%i,%i,%1.5f,%2.5f,%1.5f,%1.5f,%1.5e,%1.5e\n" % (k,i_m,x_samp,y_samp,z,e_bin,a_bin,M_sat))
                    out.close()'''
                    

cmap = cm.nipy_spectral_r
vmin = 10
vmax = 5500
my_cmap=cm.get_cmap(cmap)
norm = colors.LogNorm(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))
cmap.set_under('w')

fs = 'x-large'
width = 10.
aspect = 2.
ms = 6.
lw = 5

fig = plt.figure(figsize=(aspect*width,width),dpi=300)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

data_MC = np.genfromtxt(fname,delimiter=',',comments='#')
ax_list = [ax1,ax2,ax3,ax4]
sublbl = ['a','b','c','d']
#vmax = [10000,2000,600,100]

for m in range(0,2):
    if m ==1:
        i_m = 180
    else:
        i_m = 0
    
    host = ['A','B']
    color = ['k','r']
    for k in range(0,2):
        ax = ax_list[2*m+k]

        set_cut = np.where(np.logical_and(np.abs(data_MC[:,0]-k)<1e-6,np.abs(data_MC[:,1]-i_m)<1e-6))[0]
        #bins=[np.arange(0.001,0.1,0.001),np.arange(0.1,ymax[2*m+k],0.1)]
        '''xi = np.arange(0.001,0.1+0.001,0.001)
        yi = np.arange(0.1,ymax[2*m+k]+0.1,0.1)
        X,Y,Z = get_counts(data_MC[set_cut,2],data_MC[set_cut,3],ymax[2*m+k])
        zi = griddata((X,Y),Z,(xi[None,:],yi[:,None]),method = 'nearest')'''
        #print(np.max(data_MC[set_cut,3]))
        xbins = np.arange(0.001,0.1005,0.0005)
        #xbins = 10**np.arange(-3,-0.8,0.1)
        ybins = np.arange(0.1,ymax[2*m+k]+0.1,0.1)
        h,xedges,yedges = np.histogram2d(data_MC[set_cut,2],data_MC[set_cut,3],bins=[xbins,ybins])#,vmin=vmin,vmax=vmax,norm=colors.LogNorm(),cmap=cmap)
        #h /= np.max(h)
        #print(np.max(h))
        #h /= 3600.
        CS = ax.pcolormesh(xedges[:-1],yedges[:-1],h.T,vmin=vmin,vmax=vmax,cmap=cmap,norm=norm)
        
        #ax.plot(data_MC[set_cut,2],data_MC[set_cut,3],'.',ms=ms,color=color[k],alpha=0.5)
        #print(np.max(data_MC[set_cut,3]))
        #a_rng = np.arange(0.001,0.1,0.001)
        #TTV_max = int(np.max(CS[0]))
        #print(TTV_max)
        #if k == 0 and m == 0:
        #    print(np.sqrt(a_rng))
        #ax.plot(a_rng,10**np.sqrt(np.log10(0.1)**2-np.log10(a_rng)**2),'r-',lw=lw,zorder=5)
        ax.minorticks_on()
        ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
        ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0)
        
        ax.set_xlim(0.001,0.1)
        ax.set_ylim(0.1,ymax[2*m+k]+10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        #if m == 0:
        ax.set_yticks([1,10])
        ax.set_yticklabels(['1','10'])
        ax.set_xticks([0.001,0.01,0.1])
        ax.set_xticklabels(['0.001','0.01','0.1'])
        if m==0:
            ax.text(0.99,0.9,"prograde" , color='k',fontsize='large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
        else:
            ax.set_xlabel("$a_{\\rm p}/a_{\\rm bin}$",fontsize=fs)
            ax.text(0.99,0.9,"retrograde" , color='k',fontsize='large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
        ax.text(0.02,0.88,sublbl[2*m+k], color='k',fontsize='x-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)

            #ax.set_ylim(0.,42)
        if k == 0:
            ax.set_ylabel("TTV (min)",fontsize=fs)
        if 2*m+k == 0:
            ax.text(0.5,1.05,"Star A" , color='k',fontsize='large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
        elif 2*m+k == 1:
            ax.text(0.5,1.05,"Star B" , color='k',fontsize='large',weight='bold',horizontalalignment='center',transform=ax.transAxes)
        if (2*m+k) < 2:
            ax.set_xticklabels([])
        #colorbar(CS[3],'right')
color_label = 'Number of Systems'
cax = fig.add_axes([0.92,0.11,0.015,0.77])
cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
cbar.set_label(color_label,fontsize=fs)
#cbar.set_ticks(np.arange(0.,1.25,0.25))
#cbar.set_ticks([0.001,0.01,0.1,1])
#cbar.set_ticklabels(['0.001','0.01','0.1','1.0'])
cbar.set_ticks([10,100,1000])
cbar.set_ticklabels(['10','100','1000'])
cbar.ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
cbar.ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)

fig.subplots_adjust(hspace=0.1)
fig.savefig("MC_stab_tides_%1.2f.png" % M_A,bbox_inches='tight',dpi=300)
plt.close()




