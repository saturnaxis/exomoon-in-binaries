import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from scipy.optimize import curve_fit
import sys
import os

rcParams.update({'font.size': 22})
rcParams.update({'mathtext.fontset': 'cm'})
#rc('text', usetex=True)

def func1(x,a,b):
    return  a*(1-b*x)  #

def func1_bin(x,a,b,c):
    return -(a*x-b)**2 + c

def func2(x,a,b,c):
    ep = x[:,0]
    es = x[:,1]
    return  a*(1-b*ep-c*es)

def Combine_files(fn_out,output_fldr):
    if os.path.exists(fn_out):
        os.remove(fn_out)
    filelist = [f for f in os.listdir(output_fldr) if f.startswith('Stab_') and f.endswith('.txt') and f != 'Stab_moon.txt']

    with open(output_fldr + fn_out, "w") as outfile:
        for fn in filelist:
            with open(output_fldr+fn,"r") as infile:
                outfile.write(infile.read())
                
def Combine_files_dup(fn_out,output_fldr):
    if os.path.exists(fn_out):
        os.remove(fn_out)
    filelist = [f for f in os.listdir(output_fldr) if f.startswith('Stab_') and f.endswith('.txt') and f != 'Stab_moon.txt']

    #with open(output_fldr + fn_out, "w") as outfile:
    out_line = []
    for fn in filelist:
        with open(output_fldr+fn,"r") as infile:
            lines = infile.readlines()
        for l in lines:
            if not l in out_line:
                out_line.append(l)
                #outfile.write(infile.read())
    out = open(output_fldr + fn_out, "w")
    for i in range(0,len(out_line)):
        out.write(out_line[i])

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

def plot_ep(ax,data_dir,data_file,lbl,idx,fname,ap):
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
    stab = np.where(np.abs(data[:,-1]-1e5)<1e-6)[0]
    #print(len(stab),len(data))

    xi = np.arange(0,0.51,0.01)
    yi = np.arange(0.25,0.55,0.01)
    X = []
    Y = []
    Z = []
    out = open(fname,'w')
    out.write("#e_p,R_H,f_stab\n")
    for xp in xi:
        for yp in yi:
            X.append(xp)
            Y.append(yp)
            xy_cut = np.where(np.logical_and(np.abs(data[:,2]-xp)<1e-6,np.abs(data[:,0]-yp)<1e-6))[0]
            stab_frac = 0.
            if len(xy_cut)>0:
                for cnt in range(0,len(xy_cut)):
                    if np.abs(data[xy_cut[cnt],-1]-1e5)<1e-6:
                        stab_frac += 1.
                    #else:
                    #    stab_frac -= 1.
                if stab_frac <= 0.05:
                    stab_frac -= 1.
                stab_frac /= float(len(xy_cut))
                #print(float(len(xy_cut)))
            else:
                stab_frac = -1.
            Z.append(stab_frac)
            out.write("%1.3f,%1.3f,%1.5f\n" % (xp,yp,stab_frac))
    out.close()
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    Z[Z>=0.95] = 1.

    zi = griddata((X,Y),Z,(xi[None,:],yi[:,None]),method = 'nearest') #,fill_value=0
    CS = ax.pcolormesh(xi,yi,zi,cmap = cmap,vmin=vmin,vmax=vmax)
    x_data = []
    y_data = []
    ecc_data = []
    prob_cut = 0.#float(sys.argv[1])
    eps_p = 1.25*a_p[idx-1]/23.78*(0.524/(1.-0.524**2))
    for ecc in np.arange(0,0.5,0.01):
        e_cut = np.where(np.logical_and(np.abs(X-ecc)<1e-6,Z>=prob_cut))[0]
        if len(e_cut)>=1:
            ecc_data.append(ecc)
            eta_p = np.abs(ecc-eps_p)
            x_data.append([eps_p,eta_p])
            #print(cut_y)
            #temp_Y = np.copy(Y[e_cut])
            #temp_Y = temp_Y[np.argsort(temp_Y)[-1:]]
            y_data.append(np.max(Y[e_cut]))
            '''cut_y = Y[e_cut]
            cut_y[::-1].sort()
            if len(cut_y)>=2:
                y_data.append(np.mean(cut_y[:2]))
            else:
                y_data.append(np.mean(cut_y))'''
    #if idx == 1:
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    #popt,pcov = curve_fit(func2,x_data,y_data,method='lm') #bounds=([0.35,0.5],[0.55,1.5])
    #elif idx == 3:
    #    popt,pcov = curve_fit(func1,x_data,y_data,method='lm') #bounds=([0.35,0.5],[0.55,1.5])
    #print(popt,np.sqrt(np.diag(pcov)))
    #sigma = np.sqrt(np.diag(pcov))
    ecc = np.arange(0,0.5,0.01)
    #ax.plot(ecc,func(ecc,0.4895,1.0305),'k--',lw=lw,label='Domingos et al. 2006')
    #ax.plot(ecc_data,func2(x_data,*popt),'-',color='gray',lw=lw,label='This work')
    '''for i in range(0,300):
        a_i = np.random.normal(popt[0],sigma[0])
        b_i = np.random.normal(popt[1],sigma[1])
        if idx == 1:
            y_i = func1(ecc,a_i,b_i)
            ax.plot(ecc,y_i,'-',color='gray',lw=3,alpha=0.1)
        #elif idx == 3:
        #    #c_i = np.random.normal(popt[2],sigma[2])
        #    y_i = func1(ecc,a_i,b_i)
        #ax.plot(ecc,y_i,'-',color='gray',lw=3,alpha=0.1)
    if idx==1:
        ax.plot(ecc,func1(ecc,0.4895,1.0305),'k--',lw=lw,label='DWY06')
        ax.plot(ecc,func1(ecc,*popt),'-',color='r',lw=lw,label='This Work')
        ax.legend(loc='upper right',fontsize='x-large')'''
        
    '''if idx == 1:
        eps = -0.03*ap**2 + 0.13*ap - 0.085
    else:
        eps = -0.027*ap**2 + 0.123*ap - 0.080'''
    mu = 0.972/1.133
    if idx == 1:
        mu = 1.133/0.972
    eps = 1.25*ap/23.7*(0.524/(1.-0.524**2))
    print(2.7*eps)
    #eps /= (1. + (25./8.)*(mu/np.sqrt(1.+mu))*(ap/23.78)**1.5*(3.+2*0.524**2)/(1.-0.524**2)**1.5)
    eta = np.abs(xi-eps)
    y_stab = (1.-eps-eta)
    ax.plot(xi,0.4*y_stab,'--',color='gray',lw=lw)
    ax.plot(xi,0.5*y_stab,'-',color='gray',lw=lw)
    N_min = 3
    N_max = 15
    data = np.genfromtxt("res_width_N1.txt",delimiter=',',comments='#')
    intersect = []
    for N in range(N_min,N_max):
        N_idx = np.where(np.abs(data[:,0]-N)<1e-6)[0]
        if N>=5:
            #ax.plot(data[N_idx[0],5],data[N_idx[0],6],'.',color='b',ms=25,zorder=5)
            intersect.append((data[N_idx[0],5],data[N_idx[0],6]))
    intersect = np.asarray(intersect)
    spl = UnivariateSpline(intersect[:,0],intersect[:,1])
    ep_rng = np.arange(0.289,0.41,0.001)
    ax.plot(ep_rng,spl(ep_rng),'-',color='lightgreen',lw=lw,zorder=5)
    ax.plot(ep_rng-0.205,spl(ep_rng),'--',color='lightgreen',lw=lw,zorder=5)
    if idx >2:
        ax.plot(-ep_rng+0.205+2*eps,spl(ep_rng),'--',color='lightgreen',lw=lw,zorder=5)
    
    if idx > 2:
        ax.set_xlabel(r"initial $e_{\rm p}$",fontsize=50)
    if idx == 1 or idx == 3:
        ax.set_ylabel(r"initial $a_{\rm %s}\;(R_{\rm H})$" % lbl,fontsize=50)

    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 12.0, width = 8.0,labelsize=fs)
    ax.tick_params(which='minor',axis='both', direction='out',length = 6.0, width = 8.0)
    ax.text(0.02,0.92,sublbl[idx-1], color='k',fontsize='xx-large',weight='bold',horizontalalignment='left',transform=ax.transAxes)
    if idx == 1:
        ax.text(0.99,0.9,"$\\alpha$ Cen B" , color='k',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
    else:
        ax.text(0.99,0.9,"$\\alpha$ Cen A" , color='k',fontsize='xx-large',weight='bold',horizontalalignment='right',transform=ax.transAxes)
    ax.text(0.99,0.8,"%1.2f au" % ap , color='k',fontsize='xx-large',horizontalalignment='right',transform=ax.transAxes)
    
    ax.set_ylim(0.25,0.55)
    ax.set_xlim(0.,0.5)
    ax.set_yticks(np.arange(0.25,0.6,0.05))
    ax.set_xticks(np.arange(0,0.6,0.1))
    if idx == 3:
        color_label=r'$f_{\rm stab}$'
        cax = fig.add_axes([0.92,0.11,0.015,0.77])
        #cax=plt.axes([pos[1,0]-0.03,pos[0,1]-0.005,0.01,pos[1,1]-pos[0,1]+0.015])
        cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
        #cbar=fig.colorbar(CS,ax=ax,orientation='horizontal')
        #cbar=colorbar(CS)
        cbar.set_label(color_label,fontsize=fs)
        cbar.set_ticks(np.arange(0.,1.25,0.25))
        cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)
        #fig.text(0.3,0.94,color_label, color='black',fontsize=fs,ha='center', va='center')


home = os.getcwd() + "/"
exomoon_home = "/mnt/c/Users/satur/Dropbox/Marialis/exomoon_submoon/"
output_fldr_2 = home + "aCenA_moon_circ_2.00/"
output_fldr_2B = home + "aCenB_moon_circ_2.00/"
output_fldr_225 = home + "aCenA_moon_circ_2.25/"
output_fldr_250 = home + "aCenA_moon_circ_2.50/"

Combine_files('Stab_moon.txt',output_fldr_2)
Combine_files('Stab_moon.txt',output_fldr_2B)
Combine_files('Stab_moon.txt',output_fldr_225)
Combine_files('Stab_moon.txt',output_fldr_250)
fs = 'xx-large'
width = 10.
aspect = 2.1
ms = 6.5
lw=6
sublbl = ['a','b','c','d']
#star_lbl = ['$\\alpha$ Cen A','$\\alpha$ Cen B']
a_p = [2,2,2.25,2.5]


fig = plt.figure(figsize=(aspect*width,2*width),dpi=300)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


#plot_ep(ax1,exomoon_home+"Domingos06_circ/","Stab_moon.txt","sat",1,"Stab_frac_moon.txt")
plot_ep(ax1,output_fldr_2B,"Stab_moon.txt","sat",1,"Stab_frac_moon_2B.txt",2)
plot_ep(ax2,output_fldr_2,"Stab_moon.txt","sat",2,"Stab_frac_moon_2.txt",2)
plot_ep(ax3,output_fldr_225,"Stab_moon.txt","sat",3,"Stab_frac_moon_225.txt",2.25)
plot_ep(ax4,output_fldr_250,"Stab_moon.txt","sat",4,"Stab_frac_moon_250.txt",2.5)

#plot_esat(ax2,exomoon_home+"GenRuns_out/",2,"Contour_moon.txt")
#plot_ep(ax2,home+"aCenA_moon_circ/","Stab_moon.txt","sat",3,"Stab_frac_moon_2.txt")
#plot_esat(ax2,home+"aCenA_moon_circ/",4,"Contour_moon.txt")

fig.subplots_adjust(wspace=0.25,hspace=0.15)
fig.savefig("Stability_Fig_COM.png",bbox_inches='tight',dpi=300)
plt.close()
