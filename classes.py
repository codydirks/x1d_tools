import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Cursor

import time

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import tkFont


import numpy as np
from numpy.polynomial import chebyshev
from math import sqrt,log
from scipy.special import wofz
from scipy.integrate import simps

from copy import copy,deepcopy
from astropy.io import fits

#input format: python profile_fit.py [filename] [comma-separated central wavelengths to fit] [comma-separated initial guesses for velocity components]

plt.style.use('fivethirtyeight')

global prefactor
prefactor=float('2.95e-14')

# Lets user define region of a Spectrum1D to fit and degree of fit, stores resulting
# continuum in the Spectrum1D object for future use
class InteractiveContinuumFit(object):
	def __init__(self, spec1d_object):
	    self.spec1d_object=spec1d_object
	    self.xdata=spec1d_object.vel_arr
	    self.ydata=spec1d_object.flux_arr
	    self.mask=np.array([1]*len(self.xdata))
	    self.rms_error=0
	    self.degree_of_fit=3
	    width_inches=12
	    height_inches=9
	    dpi=100
	    self.window=Tk.Tk()
	    self.window.wm_title('Fit Continuum')
	    self.window.geometry('%dx%d+%d+%d' % (width_inches*dpi, height_inches*dpi, 0, 0))
	    self.fig=plt.figure(1,figsize=(width_inches,height_inches),dpi=dpi)
	    self.ax=plt.subplot(111)
	    plt.subplots_adjust(bottom=0.2)
	    self.canvas=FigureCanvasTkAgg(self.fig,master=self.window)
	    self.canvas.show()
	    self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

	    finish_button = Tk.Button(master=self.window, text='Finish', command=self.finished_clicked,width=10)
	    finish_button['font']=tkFont.Font(family='Helvetica', size=18)
	    finish_button.place(relx=0.8,rely=0.95)

	    output_txt_button = Tk.Button(master=self.window, text='Output .txt', command=self.output_txt_clicked,width=10)
	    output_txt_button['font']=tkFont.Font(family='Helvetica', size=18)
	    output_txt_button.place(relx=0.8,rely=0.85)

	    output_fits_button = Tk.Button(master=self.window, text='Output .fits', command=self.output_fits_clicked,width=10)
	    output_fits_button['font']=tkFont.Font(family='Helvetica', size=18)
	    output_fits_button.place(relx=0.8,rely=0.90)



	    self.fit_increment_ax=plt.axes([0.1,0.12,0.03,0.03])
	    self.fit_decrement_ax=plt.axes([0.1,0.065,0.03,0.03])

	    self.degree_of_fit_text=Tk.StringVar()
	    self.degree_of_fit_text.set(str(self.degree_of_fit))
	    self.degree_readout_box=Tk.Entry(master=self.window, textvariable=self.degree_of_fit_text, width=3)
	    self.degree_readout_box.place(relx=0.1,rely=0.875)


	    self.rms_readout_ax=plt.figtext(0.14,0.03, str(self.rms_error), horizontalalignment='left')
	    plt.figtext(0.14,0.1,'Degree of Fit')
	    plt.figtext(0.13,0.03,'RMS Error:', horizontalalignment='right')

	    self.axes=[self.ax,self.fit_increment_ax,self.fit_decrement_ax]
	    plt.subplots_adjust(bottom=0.2)
	    self.rect = Rectangle((0,0), 0, 0, facecolor='red', edgecolor='red', alpha=0.2)
	    self.previous_rects=[]
	    self.previous_rects_x=[]
	    self.is_pressed=False
	    self.is_drawing_new_rect=False
	    self.x0 = None
	    self.y0 = self.ax.get_ylim()[0]
	    self.x1 = None
	    self.y1 = self.ax.get_ylim()[1]
	    self.lines=[]
	    self.ax.add_patch(self.rect)

	    self.window.protocol("WM_DELETE_WINDOW", self.finished_clicked)
	    self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
	    self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
	    self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
	    self.degree_of_fit_text.trace('w', self.dof_changed)

	    self.inc_fit_button=Button(self.fit_increment_ax, '/\\')
	    self.dec_fit_button=Button(self.fit_decrement_ax, '\\/')
	    self.UpdatePlots()
	    Tk.mainloop()


	def on_press(self, event):
	    if event.inaxes == self.ax:
	        for x_marker in self.previous_rects_x:
	            x_min,y_min=x_marker.xy
	            x_max=x_min+x_marker.get_width()
	            y_max=y_min+x_marker.get_height()
	            if event.xdata > x_min and event.xdata <x_max and event.ydata>y_min and event.ydata<y_max:
	            	for i in [np.where(self.xdata==x)[0][0] for x in self.xdata if (x>x_min and x<x_max)]:
	            		self.mask[i]=1
	            	del self.previous_rects[self.previous_rects_x.index(x_marker)]
	            	del self.previous_rects_x[self.previous_rects_x.index(x_marker)]
	            	return
	        if event.ydata==None:
	        	return
	        self.rect = Rectangle((0,0), 0, 0, facecolor='red', edgecolor='red', alpha=0.2)
	        self.ax.add_patch(self.rect)
	        self.is_pressed=True
	        self.is_drawing_new_rect=True
	        self.x0 = event.xdata
	        self.x1 = event.xdata
	        self.rect.set_width(self.x1 - self.x0)
	        self.rect.set_xy((self.x0, self.y0))
	        self.rect.set_linestyle('dashed')
	        self.ax.figure.canvas.draw()
	    elif event.inaxes == self.fit_increment_ax:
	        self.inc_clicked()
	    elif event.inaxes == self.fit_decrement_ax:
	        self.dec_clicked()

	def on_motion(self,event):
	    #Updating slow due to canvas.draw(), speed up?
	    if self.is_pressed==False:
	    	return
	    self.x1 = event.xdata
	    self.rect.set_width(self.x1 - self.x0)
	    self.rect.set_height(self.y1 - self.y0)
	    self.rect.set_xy((self.x0, self.y0))
	    self.rect.set_linestyle('dashed')
	    self.ax.figure.canvas.draw()

	def on_release(self, event):
	    if event.ydata==None:
	    	return
	    self.is_pressed=False
	    if self.is_drawing_new_rect==True:

	    	self.x1 = event.xdata
	    	self.rect.set_width(self.x1 - self.x0)
	    	self.rect.set_height(self.y1 - self.y0)
	    	self.rect.set_xy((self.x0, self.y0))
	    	self.rect.set_linestyle('solid')
	    	self.previous_rects.append(copy(self.rect))
	    	x_marker=Rectangle((self.x0,0.95*self.y1),max([(self.x1-self.x0),0.01*(self.xdata[-1]-self.xdata[0])]),0.05*self.y1, facecolor='k',alpha=1)
	    	self.previous_rects_x.append(x_marker)
	    	for i in [np.where(self.xdata==x)[0][0] for x in self.xdata if (x>self.x0 and x<self.x1)]:
	    		self.mask[i]=0
	    self.is_drawing_new_rect=False
	    self.UpdatePlots()

	def UpdatePlots(self):
	    self.fit_continuum(self.degree_of_fit)
	    self.get_rms_error()
	    self.norm_flux=np.divide(self.ydata,self.cont_fit)

	    self.ax.cla()
	    self.ax.autoscale(enable=True, axis='y')
	    self.rms_readout_ax.set_text('{:.2e}'.format(float(self.rms_error)))
	    [self.ax.add_patch(x) for x in self.previous_rects]
	    [self.ax.add_patch(x) for x in self.previous_rects_x]
	    self.ax.plot(self.xdata, float('4e-11')+self.norm_flux/float('1e11'), 'k-',linewidth=1)
	    self.ax.plot(self.xdata,self.ydata, 'k-', linewidth=1)
	    self.ax.plot(self.xdata,self.cont_fit, 'b-', linewidth=1)
	    self.ax.plot(self.xdata,self.spec1d_object.flux_err_arr, color='0.5', linestyle='-', linewidth=1)
	    self.ax.set_ylabel('Flux')
	    self.ax.set_xlabel('Velocity')
	    self.y0, self.y1 = self.ax.get_ylim()
	    self.ax.set_xlim([self.xdata[0],self.xdata[-1]])
	    self.ax.figure.canvas.draw()

	def fit_continuum(self, deg):
		inds=np.where(self.mask !=0 )
		x_to_fit=self.xdata[inds]
		y_to_fit=self.ydata[inds]
		self.cont_fit=chebyshev.chebval(self.xdata, chebyshev.chebfit(x_to_fit,y_to_fit,deg))


	def get_rms_error(self):
		sum_sq=np.sum([(self.ydata[i]-self.cont_fit[i])**2 for i in range(len(self.mask)) if self.mask[i] == 1])
		self.rms_error=sqrt(sum_sq/float(len([i for i in self.mask if i==1])))

		#sum_sq=np.sum((self.ydata-self.cont_fit)**2)
		#self.rms_error=sqrt(sum_sq/float(len(self.ydata)))

	def output_txt_clicked(self):
		wav=str(self.spec1d_object.wav_arr[0]/(1.+(self.spec1d_object.vel_arr[0]/300000.)))
		filename=self.spec1d_object.hdr['TARGNAME']+'_'+wav+'_continuum.txt'
		with open(filename, 'w') as myfile:
		    for i in range(len(self.spec1d_object.wav_arr)):
		        myfile.write(str(self.spec1d_object.wav_arr[i])+'\t'+str(self.norm_flux[i])+'\n')

	def output_fits_clicked(self):
		pass

	def finished_clicked(self):
		self.spec1d_object.continuum=self.cont_fit
		self.spec1d_object.cont_mask=self.mask
		self.fig.clf()
		self.canvas.get_tk_widget().delete("all")
		self.window.quit()
		self.window.destroy()

	def inc_clicked(self):
		self.degree_of_fit_text.set(str(self.degree_of_fit+1))
		self.UpdatePlots()


	def dof_changed(self, *args):
		val= self.degree_of_fit_text.get()
		if val != '' and int(val)<100:
			self.degree_of_fit=int(val)
			self.UpdatePlots()

	def dec_clicked(self):
		if self.degree_of_fit>0:
			self.degree_of_fit_text.set(str(self.degree_of_fit-1))
			self.UpdatePlots()

	def get_spec(self):
		return self.spec1d_object

# Takes a Spectrum1D object, a central wavelength, and optional initial guesses for velocity components.
# Allows users to interactively add components as necessary
class FitAbsorptionLines(object):
    def __init__(self,spec1d_object,cen_wav, vel_guesses=None, iterations=50):
        self.spec1d_object=spec1d_object
        self.xdata=spec1d_object.conv_wav_to_vel(cen_wav).vel_arr
        self.ydata=spec1d_object.flux_arr/spec1d_object.continuum
        self.res=spec1d_object.hdr['SPECRES']
        self.min_b=300000./(self.res*2*sqrt(log(2)))
        self.profile_fit=np.array([1]*len(self.xdata))
        self.indiv_profiles=[]
        self.residuals=self.ydata-self.profile_fit
        self.rms_error=self.GetRMSError()
        self.cen_wav=cen_wav
        self.fit_params=[]
        self.load_atomic_info('atomic.dat')
        self.menu_option=None
        self.iterations=iterations

        self.init_window()

        self.num_comps=0
        if vel_guesses!=None:
            for vel in vel_guesses:
                self.add_vel_comp(vel, cen_wav)

		#self.AutoLineID()

        self.window.protocol("WM_DELETE_WINDOW", self._quit)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.UpdatePlot()
        Tk.mainloop()

    def init_window(self):
        width_inches=12
        height_inches=9
        dpi=100
        self.window=Tk.Tk()
        self.window.wm_title('Fit Absorption Lines Around '+str(self.cen_wav))
        self.window.geometry('%dx%d+%d+%d' % (width_inches*dpi, height_inches*dpi, 0, 0))
        self.fig=plt.figure(1,figsize=(width_inches,height_inches),dpi=dpi)
        self.ax=plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.window)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1,horizOn=False)
        self.canvas.show()

        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        # Define Buttons
        self.quit_button = Tk.Button(master=self.window, text='Quit', command=self._quit,width=10)
        self.quit_button['font']=tkFont.Font(family='Helvetica', size=24)
        self.quit_button.place(relx=0.8,rely=0.95)

        self.reset_button = Tk.Button(master=self.window, text='Reset', command=self._reset,width=10)
        self.reset_button['font']=tkFont.Font(family='Helvetica', size=24)
        self.reset_button.place(relx=0.8,rely=0.9)

        self.iterate_text=Tk.StringVar()
        self.iterate_text.set('Iterate')
        self.iterate_button = Tk.Button(master=self.window, textvariable=self.iterate_text, command=self.Iterate,width=10)
        self.iterate_button['font']=tkFont.Font(family='Helvetica', size=24)
        self.iterate_button.place(relx=0.8,rely=0.85)

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

        link_comp_v_box.place(relx=0.5, rely=0.85)
        link_comp_b_box.place(relx=0.5, rely=0.88)

        min_wav,max_wav=[self.cen_wav*(1+x/300000.) for x in (min(self.xdata), max(self.xdata))]
        self.ion_list=[x for x in self.lines if (x.lam > min_wav and x.lam < max_wav)]

        self.menu_option=Tk.StringVar(self.window)
        self.menu_option.set(self.ion_list[0])
        self.menu=apply(Tk.OptionMenu, (self.window, self.menu_option)+tuple(self.ion_list))
        self.menu_option.trace('w',self.GetOptionMenuSelection)
        self.menu.place(relx=0.1,rely=0.84)

	#Loads info from atomic.dat
    def load_atomic_info(self, filename):
        self.lines=[]
        with open(filename,'r') as myfile:
            hdr=myfile.readline()
            for line in myfile:
                #wav=float(line.strip().split(' ')[0])
                #if (wav > self.spec1d_object.wav_arr[0]-10)and(wav < self.spec1d_object.wav_arr[-1]+10):
                self.lines.append(SpectralLine(line.strip('\n')))

    def _on_click(self, event):
        if event.inaxes==self.ax:
            vel=float(event.xdata)
            self.add_vel_comp(vel, self.GetOptionMenuSelection().lam)

    def add_vel_comp(self, vel, wav):
        self.num_comps+=1
        N=float('5e14')
        b=self.min_b# Voigt Gaussian parameter, need to include Lorentzian as well?
        v0=0#((self.ion_list[0].lam-wav)*300000./wav)+vel
        group=0

        for entry in self.ion_list:
            v0=((entry.lam-wav)*300000./wav)+vel
            if entry.m==0 or entry.m == 1:
                group+=1
            self.fit_params.append({'ion':entry, 'N':N,'N_err':0.0, 'b':b,'b_err':0.0, 'v':v0,'v_err':0.0, 'comp':self.num_comps, 'group':group})

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
            err_arr=self.spec1d_object.flux_err_arr[idxs]/self.spec1d_object.continuum[idxs]
            temp_prof=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
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
            profile=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
            for i in range(nsamples):
                line=deepcopy(x) # Resets initial line parameters between realizations
                temp_prof=np.array([profile[i]+err_arr[i]*np.random.randn() for i in range(len(profile))])
                best_rms=np.sqrt(np.mean((temp_prof-profile)**2))
                # Iterate N,b,v for this realization of the profile

                for i in range(steps_per_sample):
                    factor=0.5/float(i+1)
                    # Iterate N
                    line['N']=line['N']*(1.-factor)
                    new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                    new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                    if new_rms<best_rms:
                        best_rms=new_rms
                    else:
                        line['N']=line['N']/(1.-factor)
                        line['N']=line['N']*(1.+factor)
                        new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                        new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                        if new_rms<best_rms:
                            best_rms=new_rms
                        else:
                            line['N']=line['N']/(1.+factor)
                            new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                            new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))

                    # Iterate b
                    line['b']=line['b']*(1.-factor)
                    new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                    new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                    if new_rms<best_rms:
                        best_rms=new_rms
                    else:
                        line['b']=line['b']/(1.-factor)
                        line['b']=line['b']*(1.+factor)
                        new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                        new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                        if new_rms<best_rms:
                            best_rms=new_rms
                        else:
                            line['b']=line['b']/(1.+factor)
                            new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                            new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))

                    # Iterate v
                    line['v']=line['v']-factor
                    new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                    new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                    if new_rms<best_rms:
                        best_rms=new_rms
                    else:
                        line['v']=line['v']+2*factor
                        new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
                        new_rms=np.sqrt(np.mean((temp_prof-new_fit)**2))
                        if new_rms<best_rms:
                            best_rms=new_rms
                        else:
                            line['v']=line['v']-factor
                            new_fit=np.exp(-prefactor*line['ion'].f*line['N']*Voigt(xdata_cut-line['v'],line['b']/(log(2)),0.0))
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
        self.ax.set_ylim([-0.1,3])
        self.canvas.draw()

    def GetFitParams(self):
        corrected_params=tuple(self.fit_params)
        for line in corrected_params:
            line['v']=line['v']-((line['ion'].lam-self.cen_wav)*300000./self.cen_wav)
        return corrected_params

    def _reset(self):
        self.profile_fit=np.array([1]*len(self.xdata))
        self.fit_params=[]
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

class SpectralLine(object):
    """
    Parses and stores individual atomic line info for entries in FITS6P's atomic.dat file.
    """
    def __init__(self, atomic_dat_line):
		line=atomic_dat_line.strip('\n')
		self.lam=float(line[0:10])
		self.ion=line[10:19].strip()
		self.a=int(line[19:21].strip())
		self.w=int(line[21:24].strip())
		self.i=int(line[24:26].strip())
		self.m=int(line[26:28].strip())
		self.f=line[28:38].strip()
		if self.f != '':
			self.f=float(self.f)
		else:
			self.f=0.0

		self.gamma=line[38:48].strip()
		if self.gamma != '':
			self.gamma=float(self.gamma)
		else:
			self.gamma=0.0

		self.notes=line[48:].strip()

    def __repr__(self):
		return '{:<10}'.format(self.lam) + str(self.ion)

    def __str__(self):
		return self.__repr__()

    def __eq__(self,other):
        """
        Overloaded boolean equality operator. Checks that both the ion name and wavelength are identical.
        """
        if (self.lam == other.lam) and (self.ion == other.ion):
        	return True
        else:
        	return False

# Class to represent a single STIS x1d dataset. Stores the header info and the various
# individual spectral orders are stored as individual Spec1D objects
class X1D(object):
    """
    Class to represent a single X1D file from HST archives.
    """
    def __init__(self, filename):
        hdulist=fits.open(filename)
        self.hdr=hdulist[0].header

        #Cleans up header info (removes redundant history and comment entries, which are long and cluttered)
        while 'HISTORY' in self.hdr.keys():
            self.hdr.remove('HISTORY')
        while '' in self.hdr.keys():
            self.hdr.remove('')
        self.keys=self.hdr.keys()

        # Stores the 1D spectra header info, appends the spectral resolution to this
        # header, then uses this header for each of the individual Spectrum1D objects
        hdr=hdulist[1].header
        hdr.append(card=('SPECRES',self.hdr['SPECRES']))
        hdr.append(card=('TARGNAME',self.hdr['TARGNAME']))
        self.spectra = [Spectrum1D(spectrum, hdr) for spectrum in hdulist[1].data]

    def __repr__(self):
        return self.hdr['filename']

    def __str__(self):
        return self.hdr['filename']

    def get_spec_from_wav(self, w):
		nearest=1000
		spec_idx=[]
		spec_count=0
		if hasattr(w, '__len__')==False:
			if w < float(self.hdr['minwave']) or w > float(self.hdr['maxwave']):
				print 'Wavelength must be between', self.hdr['minwave'], 'and', self.hdr['maxwave']
				return None
			for spectrum in self.spectra:
				if spectrum.wav_arr[0] < w and spectrum.wav_arr[-1] > w:
					spec_idx.append(self.spectra.index(spectrum))
			if len(spec_idx)==2:
				coadd='a'
				while coadd[0].lower() not in ('y','n'):
					coadd= raw_input(str(w)+" A covers multiple orders, co-add orders? (This will only return the region covered by both orders)\n(Y/N): ")
				if coadd[0].lower() == 'y':
					spec1=self.spectra[spec_idx[0]]
					spec2=self.spectra[spec_idx[1]]
					interp_flux=np.interp(spec1.wav_arr, spec2.wav_arr, spec2.flux_arr, left=0, right=0)
					interp_wav=np.array([spec1.wav_arr[i] for i in range(len(spec1.wav_arr)) if interp_flux[i]>0])
					interp_flux=np.array(interp_flux[np.where(interp_flux>0)])
					avg_flux=(np.array([spec1.flux_arr[i] for i in range(len(spec1.flux_arr)) if spec1.wav_arr[i] in interp_wav])+interp_flux)/2.


					spec1.flux_arr=interp_flux
					spec1.wav_arr=interp_wav
					spec1.nelem=len(interp_wav)
					return spec1
				else:
					return self.spectra[spec_idx[0]]
			else:
				return self.spectra[spec_idx[0]]
		else:
			return [self.get_spec_from_wav(i) for i in cen_wav]

#Stores individual 1D spectra that make up STIS datasets
class Spectrum1D(object):
    """
    Stores relevant info from a single 1D spectrum object (input parameter) within an X1D file.
    Also has entries for relevant continuum info.
    """
    def __init__(self, x1d_spec_object, header):
        self.hdr=header
        self.sporder=x1d_spec_object[0]
        self.wav_arr=x1d_spec_object[2][7:1020]
        self.vel_arr=np.array([0]*len(self.wav_arr))
        self.flux_arr=x1d_spec_object[6][7:1020]
        self.flux_err_arr=x1d_spec_object[7][7:1020]
        self.background_arr=x1d_spec_object[4][7:1020]
        self.dq_arr=x1d_spec_object[8][7:1020]
        self.nelem=len(self.wav_arr)
        self.continuum=np.array([0]*self.nelem)
        self.cont_mask=np.array([1]*self.nelem)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '<Spectrum1D, '+str(round(self.wav_arr[0],1))+'-'+str(round(self.wav_arr[-1],1))+'>'


    def conv_wav_to_vel(self, cen_wav):
        """
        Converts the current wavelength array to velocity space centered on the given wavelength.
        """
        if hasattr(cen_wav, '__len__'):
            print "Central Wavelength can't be a list!"
            return None
        self.vel_arr=300000.*(self.wav_arr-cen_wav)/float(cen_wav)
        return self

    def get_vel_range(self, cen_wav, vmin, vmax):
        """
        Creates a new Spectrum1D object containing only the specified velocity range.
        """
        vel_arr=self.conv_wav_to_vel(cen_wav).vel_arr
        #min_index=[x for x in vel_arr if x>vmin][0]
        #max_index=[x for x in vel_arr if x<vmax][-1]
        min_index=abs(vel_arr-vmin).argmin()
        max_index=abs(vel_arr-vmax).argmin()

        new_spec=deepcopy(self)
        new_spec.sporder=self.sporder
        new_spec.wav_arr=self.wav_arr[min_index:max_index]
        new_spec.vel_arr=new_spec.conv_wav_to_vel(cen_wav).vel_arr
        new_spec.flux_arr=self.flux_arr[min_index:max_index]
        new_spec.flux_err_arr=self.flux_err_arr[min_index:max_index]
        new_spec.background_arr=self.background_arr[min_index:max_index]
        new_spec.dq_arr=self.dq_arr[min_index:max_index]
        new_spec.nelem=self.nelem
        new_spec.continuum=self.continuum[min_index:max_index]
        return new_spec
		#return (vel_arr[min_index:max_index], self.flux_arr[min_index:max_index])

    def __add__(self,spec2):
        new_spec=deepcopy(self)
        print len(self.wav_arr), len(spec2.wav_arr)
        new_spec.wav_arr=np.concatenate([new_spec.wav_arr,spec2.wav_arr])
        #new_spec.vel_arr=new_spec.conv_wav_to_vel(cen_wav).vel_arr
        new_spec.flux_arr=np.concatenate([new_spec.flux_arr,spec2.flux_arr])
        new_spec.flux_err_arr=np.concatenate([new_spec.flux_err_arr,spec2.flux_err_arr])
        new_spec.background_arr=np.concatenate([new_spec.background_arr,spec2.background_arr])
        new_spec.dq_arr=np.concatenate([new_spec.dq_arr,spec2.dq_arr])
        new_spec.nelem=new_spec.nelem+spec2.nelem
        new_spec.continuum=np.concatenate([new_spec.continuum,spec2.continuum])
        return new_spec
