import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import root_scalar


rcParams.update({'font.size': 28})
rcParams.update({'mathtext.fontset': 'cm'})

G = 4.*np.pi**2
M_A = 1.133
L_A = 1.519
M_B = 0.972
L_B = 0.500
a_bin = 23.78
e_bin = 0.524

M_E = 3.0035e-6 #Earth mass
R_E = 4.2635e-5 #AU
Q_p = 10.
k2_p = 0.299
T = 1e10
R_p = 4.26352e-5 #Earth Radius in AU
rho_p = 5.515 #Earth density
rho_sat = 3.93 #Mars density

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

fs = 50#'x-large'
width = 10.
aspect = 2.*4./3.
ms = 25
lw = 3

fig = plt.figure(figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(221)
ax_top1 = ax1.twiny()
ax2 = fig.add_subplot(222)
ax_top2 = ax2.twiny()
ax3 = fig.add_subplot(223)
ax_top3 = ax3.twiny()
ax4 = fig.add_subplot(224)
ax_top4 = ax4.twiny()

ax_list = [ax1,ax2,ax3,ax4]
ax_list_top = [ax_top1,ax_top2,ax_top3,ax_top4]
sublbl = ['a','b','c','d']

star = ['A','B','A','B']
orb_dir = ['prograde','prograde','retrograde','retrograde']
M_s = [M_A,M_B,M_A,M_B]
L_s = [L_A,L_B,L_A,L_B]
for k in range(0,4):
    ax = ax_list[k]
    ax_top = ax_list_top[k]
    M_star = M_s[k]

    #for single star M_star
    a_p = np.sqrt(L_s[k])
    e_p = 0.05
    n_p = np.sqrt(G*M_star/a_p**3)
    R_H = a_p*(M_E/(3*M_star))**(1./3.)
    R_L = (2.*rho_p/rho_sat)**(1./3.)

    a_sat = np.array([10,30,60,100,150])*R_p/R_H
    a_sat_lbls = ['10','30','60','100','150',]

    if k < 2:
        ax_top.set_xlim(0.005,0.4)
    else:
        ax_top.set_xlim(0.005,0.7)

    if k == 0:
        ax_top.set_xticks(a_sat[:4])
        ax_top.set_xticklabels(a_sat_lbls[:4])
    elif k == 1:
        ax_top.set_xticks(a_sat[:3])
        ax_top.set_xticklabels(a_sat_lbls[:3])
    elif k == 2:
        ax_top.set_xticks(a_sat)
        ax_top.set_xticklabels(a_sat_lbls)
    else:
        ax_top.set_xticks(a_sat[:4])
        ax_top.set_xticklabels(a_sat_lbls[:4])
    

    ax.plot(60*R_p/R_H,0.0123,marker='$\oplus$',color='k',ms=35,zorder=5)
    
    ax.text(0.02,0.92,sublbl[k], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.text(0.02,0.12,"$\\alpha$ Cen %s" % star[k], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    ax.text(0.02,0.02,"%s" % orb_dir[k], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)

    if k == 0:
        ax.fill_betweenx(np.arange(0.0,1.01,0.01),0.38,0.4,color='lightgray')
    elif k == 1:
        ax.fill_betweenx(np.arange(0.0,1.01,0.01),0.38,0.4,color='lightgray')
    elif k==2:
        ax.fill_betweenx(np.arange(0.0,1.01,0.01),0.65,0.7,color='lightgray')
    else:
        ax.fill_betweenx(np.arange(0.0,1.01,0.01),0.65,0.7,color='lightgray')
    
    eps = 1.25*a_p/a_bin*(e_bin/(1.-e_bin**2))
    e_p = eps
    eta = np.abs(e_p-eps)
    y_stab = (1.-eps-eta)

    if k % 2 == 0:
        Q_p = 33*f_ecc(2,0.1)
    elif k % 2 ==1:
        Q_p = 33*f_ecc(2,0.15)

    TTV_sig = [1,2,5,10,20,40]
    color = ['k','c','m','g','y','orange']
    ls = ['-','--']
    for s in range(0,len(TTV_sig)):
        TTV = TTV_sig[s]/525600. #convert minutes to year

        TTV_C = n_p*a_p*TTV*np.sqrt((1.+e_p)/(1.-e_p))
        if k < 2:
            f_crit = 0.38
        else:
            f_crit = 0.65
        a_crit = f_crit*R_H
        a_sat = np.arange(0.002,f_crit,0.001)*R_H
        q_ps = TTV_C/(a_sat - TTV_C)

        try:
            sol = root_scalar(func,args=(f_crit,a_p,0.299,M_E,R_E,Q_p,M_star,1e10),method='bisect',bracket=[1e-3,0.3],xtol=1e-6)
            M_sat = sol.root
        except ValueError:
            M_sat = 0.1
        M_sat = 0.1
        tide_cut = np.where(np.logical_and(q_ps < (M_sat),q_ps>0.001))[0]
        ax.plot(a_sat[tide_cut]/R_H,q_ps[tide_cut],ls='-',color=color[s],lw=12,label='%i min' % TTV_sig[s])
        
        if k < 2:
            f_crit = 0.38
        else:
            f_crit = 0.65
        a_crit = f_crit*R_H#*y_stab
        a_sat = np.arange(0.005,f_crit,0.001)*R_H
        q_ps = TTV_C/(a_sat - TTV_C)
        try:
            sol = root_scalar(func,args=(f_crit,a_p,0.299,M_E,R_E,Q_p,M_star,1e10),method='bisect',bracket=[1e-3,0.3],xtol=1e-6)
            M_sat = sol.root
        except ValueError:
            M_sat = 0.1
        tide_cut = np.where(np.logical_and(q_ps < (M_sat),q_ps>0.001))[0]
    if k == 0:
        ax.legend(loc='upper left',fontsize='large',ncol=6,bbox_to_anchor=(-0.05, 1.25, 2., .102),frameon=False)
    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0)
    ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 8.0)
    ax.set_yscale('log')
    if k < 2:
        ax.set_xlim(0.005,0.4)
    else:
        ax.set_xlim(0.005,0.7)
    ax.set_ylim(0.001,0.2)


    ax_top.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,colors='r')
    if k < 2:
        ax_top.set_xlabel("$a_{\\rm sat}$ (R$_{\\rm p}$)",fontsize=fs,labelpad=15,color='r')
    else:
        ax.set_xlabel("$a_{\\rm sat}$ (R$_{\\rm H}$)",fontsize=fs)
    if k == 0 or k == 2:
        ax.set_ylabel("M$_{\\rm sat}/$M$_{\\rm p}$",fontsize=fs)
    if k == 1 or k == 3:
        ax.set_yticklabels([])
    

fig.subplots_adjust(wspace=0.1,hspace=0.25)
fig.savefig("../Figs/Fig10_TTV.png",bbox_inches='tight',dpi=300)
plt.close()
