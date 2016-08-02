import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Cursor

import sys
import os
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import tkFont

import numpy as np
from math import sqrt,log
from scipy.special import wofz

from x1d_tools import SpectralLine

global PREFACTOR
PREFACTOR=float('2.95e-14')


def Voigt(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)


# Takes a Spectrum1D object, a central wavelength, and optional initial guesses for velocity components.
# Allows users to interactively add components as necessary
class FitAbsorptionLines(object):
    def __init__(self,spec1d_object,cen_wav, vel_guesses=None, iterations=50):
        self.spec1d_object=spec1d_object
        self.xdata=spec1d_object.conv_wav_to_vel(cen_wav).vel_arr
        self.ydata=spec1d_object.flux_arr/spec1d_object.continuum
        self.res=spec1d_object.hdr['SPECRES']
        self.min_b=round(300000./(self.res*2*sqrt(log(2))),3)
        self.profile_fit=np.array([1]*len(self.xdata))
        self.indiv_profiles=[]
        self.residuals=self.ydata-self.profile_fit
        self.rms_error=self.GetRMSError()
        self.cen_wav=cen_wav
        self.fit_params=[]
        self.load_atomic_info('atomic.dat',())
        self.menu_option=None
        self.iterations=iterations
        self.comp_rects=[]
        self.comp_rects_x=[]

        self.init_window()

        self.num_comps=0
        self.vel_comps=[]
        if vel_guesses!=None:
            for vel in vel_guesses:
                self.add_vel_comp(vel, cen_wav)

		#self.AutoLineID()

        self.window.protocol("WM_DELETE_WINDOW", self._quit)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.UpdatePlot()
        Tk.mainloop()

    def init_window(self):
        width_inches=9
        height_inches=6
        dpi=100
        self.window=Tk.Tk()
        self.window.wm_title('Identify Absorption Lines for '+self.spec1d_object.hdr['TARGNAME']+'  around '+str(self.cen_wav))
        self.window.geometry('%dx%d+%d+%d' % (width_inches*dpi, height_inches*dpi, 0, 0))
        self.fig=plt.figure(1,figsize=(width_inches,height_inches),dpi=dpi)
        self.ax2=plt.subplot(111)
        self.ax=self.ax2.twiny()
        plt.subplots_adjust(bottom=0.2)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.window)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1,horizOn=False)
        self.canvas.show()

        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        # Define Buttons
        self.quit_button = Tk.Button(master=self.window, text='Quit', command=self._quit,width=10)
        self.quit_button['font']=tkFont.Font(family='Helvetica', size=18)
        self.quit_button.place(relx=0.99,rely=0.99,anchor='se')

        self.reset_button = Tk.Button(master=self.window, text='Reset', command=self._reset,width=10)
        self.reset_button['font']=tkFont.Font(family='Helvetica', size=18)
        self.reset_button.place(relx=0.845,rely=0.99,anchor='se')

        #self.iterate_text=Tk.StringVar()
        #self.iterate_text.set('Iterate')
        #self.iterate_button = Tk.Button(master=self.window, textvariable=self.iterate_text, command=self.Iterate,width=10)
        #self.iterate_button['font']=tkFont.Font(family='Helvetica', size=18)
        #self.iterate_button.place(relx=0.8,rely=0.85)

        self.output_fits6p_params_button = Tk.Button(master=self.window, text='Output .par', command=self.output_fits6p_params, width=10)
        self.output_fits6p_params_button['font']=tkFont.Font(family='Helvetica', size=18)
        self.output_fits6p_params_button.place(relx=0.7,rely=0.99,anchor='se')

        self.hide_molecules_var=Tk.IntVar()
        self.hide_molecules_box=Tk.Checkbutton(master=self.window,text='Hide Molecules',variable=self.hide_molecules_var,command=self._hide_molecules_changed)
        self.hide_molecules_box.place(relx=0.25,rely=0.91,anchor='ne')

        # Define lock checkbox variables
        self.lock_N_var=Tk.IntVar()
        self.lock_b_var=Tk.IntVar()
        self.lock_v_var=Tk.IntVar()
        self.link_comp_v_var=Tk.IntVar()
        self.link_comp_b_var=Tk.IntVar()
        self.link_ion_N_var=Tk.IntVar()
        self.link_ion_b_var=Tk.IntVar()
        self.link_ion_v_var=Tk.IntVar()

        # Define checkboxes
        lock_N_box=Tk.Checkbutton(master=self.window, text='Lock all N', variable=self.lock_N_var)
        lock_b_box=Tk.Checkbutton(master=self.window, text='Lock all b', variable=self.lock_b_var)
        lock_v_box=Tk.Checkbutton(master=self.window, text='Lock all v', variable=self.lock_v_var)

        link_ion_N_box=Tk.Checkbutton(master=self.window, text='Link ion N', variable=self.link_ion_N_var)
        link_ion_b_box=Tk.Checkbutton(master=self.window, text='Link ion b', variable=self.link_ion_b_var)
        link_ion_v_box=Tk.Checkbutton(master=self.window, text='Link ion v', variable=self.link_ion_v_var)

        link_comp_v_box=Tk.Checkbutton(master=self.window, text='Link component v', variable=self.link_comp_v_var)
        link_comp_b_box=Tk.Checkbutton(master=self.window, text='Link component b', variable=self.link_comp_b_var)

        lock_N_box.place(relx=0.3, rely=0.85)
        lock_b_box.place(relx=0.3, rely=0.88)
        lock_v_box.place(relx=0.3, rely=0.91)

        link_ion_N_box.place(relx=0.4, rely=0.85)
        link_ion_b_box.place(relx=0.4, rely=0.88)
        link_ion_v_box.place(relx=0.4,rely=0.91)

        link_comp_b_box.place(relx=0.5, rely=0.88)
        link_comp_v_box.place(relx=0.5, rely=0.91)


        min_wav,max_wav=[self.cen_wav*(1+x/300000.) for x in (min(self.xdata), max(self.xdata))]
        self.ion_list=[x for x in self.lines if (x.lam > min_wav and x.lam < max_wav)]


        self.menu_option=Tk.StringVar(self.window)
        self.menu_option.set(self.ion_list[0])
        self.menu=Tk.OptionMenu(self.window,self.menu_option,*self.ion_list)
        #self.menu=apply(Tk.OptionMenu, (self.window, self.menu_option)+tuple(self.ion_list))
        self.menu_option.trace('w',self.GetOptionMenuSelection)
        self.menu.place(relx=0.25,rely=0.85,anchor='ne')

	#Loads info from atomic.dat
    def load_atomic_info(self, filename, excluded):
        self.lines=[]
        path=os.path.realpath(__file__).strip(os.path.basename(__file__))
        with open(path+filename,'r') as myfile:
            hdr=myfile.readline()
            for line in myfile:
                include=True
                temp=SpectralLine(line.strip('\n'))
                for ex in excluded:
                    if temp.ion.startswith(ex):
                        include=False
                if include:
                    self.lines.append(SpectralLine(line.strip('\n')))

    def _on_click(self, event):
        if event.inaxes==self.ax:
            for x_marker in self.comp_rects_x:
                x_min,y_min=x_marker.xy
                x_max=x_min+x_marker.get_width()
                y_max=y_min+x_marker.get_height()
                if event.xdata > x_min and event.xdata <x_max and event.ydata>y_min and event.ydata<y_max:
                    idx=self.comp_rects_x.index(x_marker)
                    del self.fit_params[idx]
                    del self.comp_rects[idx]
                    del self.comp_rects_x[idx]
                    self.CalcNewTheorProfile()
                    self.UpdatePlot()
                    return
            if event.ydata<2.8:
                vel=float(event.xdata)
                self.add_vel_comp(vel, self.GetOptionMenuSelection().lam)

    def _hide_molecules_changed(self):
        if self.hide_molecules_var.get()==1:
            excluded=('CO','13CO','H2','HD','OH','H2O','SH','SiO','NO+','CH','CH2','CS','C18O','PtNe','CN+','AlH','NH','CH+','CN','C3','UID')
        else:
            excluded=()
        self.load_atomic_info('atomic.dat',excluded)
        min_wav,max_wav=[self.cen_wav*(1+x/300000.) for x in (min(self.xdata), max(self.xdata))]
        self.ion_list=[x for x in self.lines if (x.lam > min_wav and x.lam < max_wav)]
        self.menu_option=Tk.StringVar(self.window)
        self.menu_option.set(self.ion_list[0])
        self.menu.destroy()
        self.menu=Tk.OptionMenu(self.window,self.menu_option,*self.ion_list)
        self.menu.place(relx=0.25,rely=0.85,anchor='ne')
        #self.canvas.draw()

    def add_vel_comp(self, vel, wav):
        self.num_comps+=1
        self.vel_comps.append(((self.cen_wav-wav)*300000./wav)+vel)
        N=float('8e14')
        b=self.min_b# Voigt Gaussian parameter, need to include Lorentzian as well?
        v0=0#((self.ion_list[0].lam-wav)*300000./wav)+vel
        group=0

        for entry in self.ion_list:
            v0=((entry.lam-wav)*300000./wav)+vel
            #N=float('e14')/self.ydata[min(range(len(self.xdata)), key=lambda i: abs(self.xdata[i]-v0))]
            if entry.m==0 or entry.m == 1:
                group+=1
            self.fit_params.append({'ion':entry, 'N':N,'N_err':0.0, 'b':b,'b_err':0.0, 'v':v0,'v_err':0.0, 'comp':self.num_comps, 'group':group})
            self.comp_rects.append(Rectangle((v0-2*b,0),4*b,3,facecolor='red',edgecolor='red',alpha=0.2))
            self.comp_rects_x.append(Rectangle((v0-2*b,2.8),4*b,0.2,facecolor='k',edgecolor='k',alpha=1))

        self.CalcNewTheorProfile()
        self.UpdatePlot()

    def GetOptionMenuSelection(self, *args):
		for ion in self.ion_list:
			if str(ion) == self.menu_option.get():
				return ion

    def GetRMSError(self):
        sum_sq=np.sum([(self.ydata[i]-self.profile_fit[i])**2 for i in range(len(self.profile_fit))])
        return sqrt(sum_sq/float(len(self.profile_fit)))

    def GetChiSquared(self):
        return np.sum([((self.ydata[i]-self.profile_fit[i])**2)/self.profile_fit[i] for i in range(len(self.profile_fit))])

    def GetResiduals(self):
        self.residuals=self.ydata-self.profile_fit
        return self.residuals

    def CalcNewTheorProfile(self):
        # Reset theoretical profile
        self.profile_fit=np.array([1.]*len(self.xdata))
        self.indiv_profiles=[]
        for line in self.fit_params:
            idxs=np.where(abs(self.xdata-line['v'])<(10*line['b']))
            xdata_cut=self.xdata[idxs]
            #err_arr=self.spec1d_object.flux_err_arr[idxs]/self.spec1d_object.continuum[idxs]
            temp_prof=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
            self.profile_fit[abs(self.xdata-line['v'])<(10*line['b'])]= self.profile_fit[abs(self.xdata-line['v'])<(10*line['b'])]*temp_prof
            self.indiv_profiles.append([xdata_cut,temp_prof])
        self.rms_error=self.GetRMSError()
        self.residuals=self.GetResiduals()

    def CalcStatisticalErrors(self):
        # Uses a Monte Carlo method to determine errors in N,b,v
        nsamples=300
        steps_per_sample=20
        factor=0.05
        N_errs=[]
        b_errs=[]
        v_errs=[]
        for x in self.fit_params:
            Ns=[]
            bs=[]
            vs=[]
            line=deepcopy(x)
            idxs=np.where(abs(self.xdata-line['v'])<(10*line['b']))
            xdata_cut=self.xdata[idxs]
            err_arr=self.spec1d_object.flux_err_arr[idxs]/self.spec1d_object.continuum[idxs]
            profile=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
            for i in range(nsamples):
                line=deepcopy(x) # Resets initial line parameters between realizations
                temp_prof=np.array([profile[i]+err_arr[i]*np.random.randn() for i in range(len(profile))])
                best_rms=np.sqrt(np.mean((temp_prof-profile)**2))
                # Iterate N,b,v for this realization of the profile

                for i in range(steps_per_sample):
                    factor=0.5/float(i+1)
                    # Iterate N
                    line['N']=line['N']*(1.-factor)
                    new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                    new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                    if new_rms<best_rms:
                        best_rms=new_rms
                    else:
                        line['N']=line['N']/(1.-factor)
                        line['N']=line['N']*(1.+factor)
                        new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                        new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                        if new_rms<best_rms:
                            best_rms=new_rms
                        else:
                            line['N']=line['N']/(1.+factor)
                            new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                            new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))

                    # Iterate b
                    line['b']=line['b']*(1.-factor)
                    new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                    new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                    if new_rms<best_rms:
                        best_rms=new_rms
                    else:
                        line['b']=line['b']/(1.-factor)
                        line['b']=line['b']*(1.+factor)
                        new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                        new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                        if new_rms<best_rms:
                            best_rms=new_rms
                        else:
                            line['b']=line['b']/(1.+factor)
                            new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                            new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))

                    # Iterate v
                    line['v']=line['v']-factor
                    new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                    new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                    if new_rms<best_rms:
                        best_rms=new_rms
                    else:
                        line['v']=line['v']+2*factor
                        new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                        new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                        if new_rms<best_rms:
                            best_rms=new_rms
                        else:
                            line['v']=line['v']-factor
                            new_fit=np.exp(-PREFACTOR*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                            new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))


                Ns.append(line['N'])
                bs.append(line['b'])
                vs.append(line['v'])
            x['N_err'] = np.std(np.array(Ns))
            x['b_err'] = np.std(np.array(bs))
            x['v_err'] = np.std(np.array(vs))

    def OldIterate(self):
        self.iterate_text.set('Iterating...')
        init_chi_sq=self.GetRMSError()
        loop=0
        while True:
            init_chi_sq=self.GetRMSError()
            factor=0.1  #**(1+0.2*loop)
            for line in self.fit_params:
                if self.lock_N_var.get()==0:
                    self.Iterate_N(line,factor)
                if self.lock_b_var.get()==0:
                    self.Iterate_b(line,factor)
                if self.lock_v_var.get()==0:
                    self.Iterate_v(line,factor)
                self.CalcNewTheorProfile()
            #self.UpdatePlot()
            new_chi_sq=self.GetRMSError()
            loop+=1
            print new_chi_sq
            if ((1.-new_chi_sq/init_chi_sq) < 0.001) or loop>self.iterations:
                break

        self.iterate_text.set('Iterate')
        self.UpdatePlot()

    def Iterate(self):
        self.iterate_text.set('Iterating...')
        self.UpdatePlot()
        init_chi_sq=self.GetRMSError()
        loop=0
        while True:
            init_chi_sq=self.GetRMSError()
            factor=0.1  #**(1+0.2*loop)
            for line in self.fit_params:
                if self.lock_N_var.get()==0:
                    self.Iterate_N(line,factor)
            for line in self.fit_params:
                if self.lock_b_var.get()==0:
                    self.Iterate_b(line,factor/4.)
            for line in self.fit_params:
                if self.lock_v_var.get()==0:
                    self.Iterate_v(line,factor)
                self.CalcNewTheorProfile()
            #self.UpdatePlot()
            new_chi_sq=self.GetRMSError()
            loop+=1
            if loop>self.iterations or ((1.-new_chi_sq/init_chi_sq) < 0.001):
                break

        self.iterate_text.set('Iterate')
        self.UpdatePlot()

    def Iterate_N(self,line,factor):
        loop_limit=5
        idx=self.fit_params.index(line)
        rms_error=self.GetRMSError()
        self._adjust_N(idx,factor) # Try adjusting N downward
        self.CalcNewTheorProfile()
        new_rms_error=self.GetRMSError()
        if new_rms_error < rms_error: # If new fit is better, keep line
            loop=0
            while loop<loop_limit and (1.-new_rms_error/rms_error) > 0.1: # Keep adjusting N downard as long as fit is improving
                rms_error=new_rms_error
                self._adjust_N(idx,factor)
                self.CalcNewTheorProfile()
                new_rms_error=self.rms_error
                loop+=1

            return self.fit_params[idx]
        else:
            self._adjust_N(idx,-(factor/(1.-factor))) # If fit is worse, return line to original
            self._adjust_N(idx,-factor)

            self.CalcNewTheorProfile()
            new_rms_error = self.GetRMSError()
            if new_rms_error < rms_error:
                loop=0
                while loop<loop_limit and (1.-new_rms_error/rms_error) > 0.1:
                    rms_error=new_rms_error
                    self._adjust_N(idx,-factor)
                    #self.fit_params[idx]['N']=self.fit_params[idx]['N']*(1.+factor)
                    self.CalcNewTheorProfile()
                    new_rms_error=self.rms_error
                    loop+=1

                return self.fit_params[idx]
            else:
                self._adjust_N(idx,(factor/(1.+factor)))
                #self.fit_params[idx]['N']=self.fit_params[idx]['N']*(1.-factor)
                return self.fit_params[idx]

    def Iterate_b(self,line,factor):
        loop_limit=5
        idx=self.fit_params.index(line)
        rms_error=self.GetRMSError()
        self._adjust_b(idx,factor) # Try changing line
        self.CalcNewTheorProfile()
        new_rms_error=self.GetRMSError()
        if new_rms_error < rms_error: # If new fit is better, keep line
            loop=0
            while loop<loop_limit and (1.-new_rms_error/rms_error) > 0.1:
                rms_error=new_rms_error
                self._adjust_b(idx,factor)
                self.CalcNewTheorProfile()
                new_rms_error=self.rms_error
                loop+=1
            return self.fit_params[idx]
        else:
            self._adjust_b(idx,-(factor/(1.-factor))) # Undo initial change
            self._adjust_b(idx,-factor) # Change in other direction
            self.CalcNewTheorProfile()
            new_rms_error = self.GetRMSError()
            if new_rms_error < rms_error:
                loop=0
                while loop<loop_limit and (1.-new_rms_error/rms_error) > 0.1:
                    rms_error=new_rms_error
                    self._adjust_b(idx,-factor)
                    self.CalcNewTheorProfile()
                    new_rms_error=self.rms_error
                    loop+=1
                return self.fit_params[idx]
            else:
                self._adjust_N(idx,(factor/(1.+factor)))
                return self.fit_params[idx]

    def Iterate_v(self,line,shift):
        idx=self.fit_params.index(line)
        rms_error=self.GetRMSError()
        self._shift_v(idx,shift)
        self.CalcNewTheorProfile()
        new_rms_error=self.GetRMSError()
        if new_rms_error < rms_error:
            while (1.-new_rms_error/rms_error) > 0.01:
                rms_error=new_rms_error
                self._shift_v(idx,shift)
                self.CalcNewTheorProfile()
                new_rms_error=self.rms_error
            return self.fit_params[idx]
        else:
            self._shift_v(idx,-2*shift)
            self.CalcNewTheorProfile()
            new_rms_error=self.GetRMSError()
            if new_rms_error < rms_error:
                while (1.-new_rms_error/rms_error) > 0.01:
                    rms_error=new_rms_error
                    self._shift_v(idx,-shift)
                    self.CalcNewTheorProfile()
                    new_rms_error=self.rms_error
                return self.fit_params[idx]
            else:
                self._shift_v(idx,shift)
                return self.fit_params[idx]

    def _adjust_N(self,idx,factor):
        comp=self.fit_params[idx]['comp']
        group=self.fit_params[idx]['group']
        if self.link_ion_N_var.get()==1:
            new_N=self.fit_params[idx]['N']*(1.-factor)
            for x in self.fit_params:
                if x['comp']==comp and x['group']==group:
                    x['N']=new_N

        else:
            self.fit_params[idx]['N']=self.fit_params[idx]['N']*(1.-factor)

    def _adjust_b(self,idx,factor):
        comp=self.fit_params[idx]['comp']
        group=self.fit_params[idx]['group']
        if self.link_comp_b_var.get()==1:
            new_b=max(self.fit_params[idx]['b']*(1-factor), self.min_b)
            for x in self.fit_params:
                if x['comp']==comp:
                    x['b']=new_b
        elif self.link_ion_b_var.get()==1:
            new_b=max(self.fit_params[idx]['b']*(1-factor), self.min_b)
            for x in self.fit_params:
                if x['comp']==comp and x['group']==group:
                    x['b']=new_b
        else:
            self.fit_params[idx]['b']=max(self.fit_params[idx]['b']*(1-factor),self.min_b)

    def _shift_v(self,idx,shift):
        if self.link_comp_v_var.get()==1: # Need to shift vel for all lines
            comp=self.fit_params[idx]['comp']
            for x in self.fit_params:
                if x['comp']==comp:
                    x['v'] = x['v']-shift
        elif self.link_ion_v_var.get()==1: # Need to shift vel for all lines of same species
            comp=self.fit_params[idx]['comp']
            ion=self.fit_params[idx]['group']
            for x in self.fit_params:
                if x['comp']==comp and x['group']==ion:
                    x['v']=x['v']-shift
        else:
            self.fit_params[idx]['v']=self.fit_params[idx]['v']-shift

    def AutoLineID(self):
        vel=min(self.xdata)
        wav=self.ion_list[0].lam
        self.num_comps+=1
        N=200.
        b=1.5# Voigt Gaussian parameter, need to include Lorentzian as well?
        v0=0#((self.ion_list[0].lam-wav)*300000./wav)+vel


        # Only works for 1 velocity component, need to implement for arbitrary num of comps
        best_rms=self.GetRMSError()
        for vel in self.xdata:
            self.fit_params=[]
            for entry in self.ion_list:
                v0=((entry.lam-wav)*300000./wav)+vel
                self.fit_params.append({'ion':entry, 'N':N,'b':b,'v':v0,'comp':1})
            self.CalcNewTheorProfile()
            new_rms=self.rms_error
            if new_rms<best_rms:
                best_fit_params=self.fit_params
                best_rms=new_rms

        self.fit_params=best_fit_params
        for line in self.fit_params:
            line['N']=50.
        self.CalcNewTheorProfile()
        self.UpdatePlot()

    def UpdatePlot(self):
        self.ax.cla()
        self.ax.plot(self.xdata,self.ydata,'k-', linewidth=1)
        self.ax.plot(self.xdata,self.profile_fit,'b-', linewidth=1)
        for entry in self.indiv_profiles:
            self.ax.plot(entry[0],entry[1], 'r--', linewidth=0.75)
        self.ax.plot(self.xdata, self.residuals+2, 'k-', linewidth=1)
        [self.ax.add_patch(x) for x in self.comp_rects]
        [self.ax.add_patch(x) for x in self.comp_rects_x]
        self.ax.set_ylim([-0.1,3])
        self.ax.set_yticklabels(['-0.5','0.0','0.5','1.0'])

        new_tick_locations=self.ax.get_xticks()
        self.ax2.set_xlim(self.ax.get_xlim())
        self.ax2.set_xticks(new_tick_locations)
        self.ax2.set_xticklabels([str(round(self.cen_wav*(1+vel/300000.),2)) for vel in new_tick_locations])
        self.canvas.draw()
        self.fig.sca(self.ax)

    def GetFitParams(self):
        corrected_params=tuple(self.fit_params)
        for line in corrected_params:
            line['v']=self.vel_comps[line['comp']-1]
        return corrected_params

    def _reset(self):
        self.profile_fit=np.array([1]*len(self.xdata))
        self.fit_params=[]
        self.comp_rects=[]
        self.comp_rects_x=[]
        self.CalcNewTheorProfile()
        self.UpdatePlot()

    def _quit(self):
        #self.CalcStatisticalErrors()
        for entry in self.GetFitParams():
            pass#print entry['ion']
            #print 'N:',str('%.3e'%entry['N']), '+-', str('%.3e'%entry['N_err'])
            #print 'b:',round(entry['b'],3), '+-', round(entry['b_err'], 3)
            #print 'v:',round(entry['v'],2), '+-', round(entry['v_err'],2)
            #print
        self.window.quit()     # stops mainloop
        self.window.destroy()# this is necessary on Windows to prevent
                  # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def output_fits6p_params(self):
        params=self.GetFitParams()
        filename=self.spec1d_object.hdr['TARGNAME']+'_'+str(self.cen_wav)+'.par'
        #print filename, len(self.fit_params)
        with open(filename,'w') as myfile:
            #First line - number of lines to fit
            myfile.write('{:>5}'.format(len(params)))
            myfile.write('\n')
            self.load_atomic_info('atomic.dat',())
            for line in self.lines:
                if line.lam == params[0]['ion'].lam:
                    offset=self.lines.index(line)
            # Write entries for each line to be fit
            for comp in range(1,self.num_comps+1):
                n_curr_group=-1
                n_link_counter=2
                b_curr_group=-1
                b_link_counter=2
                v_curr_group=-1
                v_link_counter=2
                for entry in params:
                    if entry['comp']==comp:
                        # Calc relative line ID flags
                        line_id=self.lines.index(entry['ion'])-offset

                        # Determine (N,b,v) vary flags
                        if self.lock_N_var.get() == 1:
                            n_vary=1
                        elif self.link_ion_N_var.get()==1:
                            if entry['group'] == n_curr_group:
                                n_vary=n_link_counter
                                n_link_counter+=1
                            else:
                                n_curr_group=entry['group']
                                n_vary=0
                                n_link_counter=2
                        else:
                            n_vary=0

                        if self.lock_b_var.get() == 1:
                            b_vary=1
                        elif self.link_comp_b_var.get()==1:
                            if b_link_counter==2:
                                b_vary=0
                                b_link_counter+=1
                            else:
                                b_vary=b_link_counter-1
                                b_link_counter+=1

                        elif self.link_ion_b_var.get()==1:
                            if entry['group'] == b_curr_group:
                                b_vary=b_link_counter
                                b_link_counter+=1
                            else:
                                b_curr_group=entry['group']
                                b_vary=0
                                b_link_counter=2
                        else:
                            b_vary=0

                        if self.lock_v_var.get() == 1:
                            v_vary=1
                        elif self.link_comp_v_var.get()==1:
                            if v_link_counter==2:
                                v_vary=0
                                v_link_counter+=1
                            else:
                                v_vary=v_link_counter-1
                                v_link_counter+=1
                        elif self.link_ion_v_var.get()==1:
                            if entry['group'] == v_curr_group:
                                v_vary=v_link_counter
                                v_link_counter+=1
                            else:
                                v_curr_group=entry['group']
                                v_vary=0
                                v_link_counter=2
                        else:
                            v_vary=0

                        # Write everything out
                        myfile.write('{:>5}'.format(line_id))
                        myfile.write('{:>10}'.format('%.2e'%(entry['N']/100.)))
                        myfile.write('{:>10}'.format('%.3f'%entry['b']))
                        myfile.write('{:>10}'.format('%.2f'%entry['v']))
                        for flag in (n_vary,b_vary,v_vary):
                            myfile.write('{:>5}'.format(str(flag)))
                        myfile.write('\n')
        self._hide_molecules_changed()
