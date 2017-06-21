import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Cursor
plt.style.use('fivethirtyeight')

import sys
import os.path as pth

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import tkFont

import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import interp1d
from math import sqrt,log
from astropy.io import fits

from copy import copy,deepcopy

from x1d_tools import X1D

class InteractiveContinuumFit(object):
    def __init__(self, spec1d_object):
        self.spec1d_object=spec1d_object
        self.specs_to_sum=[]
        self.xdata=spec1d_object.vel_arr
        self.ydata=spec1d_object.flux_arr
        self.cen_wav=self.spec1d_object.wav_arr[0]/(1.0+(self.spec1d_object.vel_arr[0]/300000.))
        self.mask=np.array([1]*len(self.xdata))
        self.rms_error=0
        self.degree_of_fit=3
        width_inches=9
        height_inches=6
        dpi=100
        self.window=Tk.Tk()
        self.window.wm_title(self.spec1d_object.hdr['TARGNAME']+' @ '+str(self.cen_wav))
        self.window.geometry('%dx%d+%d+%d' % (width_inches*dpi, height_inches*dpi, 0, 0))
        self.fig=plt.figure(1,figsize=(width_inches,height_inches),dpi=dpi)
        self.ax=plt.subplot(111)
        plt.subplots_adjust(bottom=0.2)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.window)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        output_txt_button = Tk.Button(master=self.window, text='Output .dat', command=self.output_txt_clicked,width=10)
        output_txt_button['font']=tkFont.Font(family='Helvetica', size=18)
        output_txt_button.place(relx=0.845,rely=0.99,anchor='se')

        #output_fits_button = Tk.Button(master=self.window, text='Output .fits', command=self.output_fits_clicked,width=10)
        #output_fits_button['font']=tkFont.Font(family='Helvetica', size=18)
        #output_fits_button.place(relx=0.8,rely=0.90)

        finish_button = Tk.Button(master=self.window, text='Finish', command=self.finished_clicked,width=10)
        finish_button['font']=tkFont.Font(family='Helvetica', size=18)
        finish_button.place(relx=0.99,rely=0.99,anchor='se')

        add_spectra_button=Tk.Button(master=self.window, text='Add Spectra', command=self.add_spectra, width=10)
        add_spectra_button['font']=tkFont.Font(family='Helvetica', size=18)
        add_spectra_button.place(relx=0.7,rely=0.99,anchor='se')


        #self.fit_increment_ax=plt.axes([0.1,0.12,0.03,0.03])
        #self.fit_decrement_ax=plt.axes([0.1,0.065,0.03,0.03])
        #self.inc_fit_button=Button(self.fit_increment_ax, '/\\')
        #self.dec_fit_button=Button(self.fit_decrement_ax, '\\/')

        self.degree_of_fit_text=Tk.StringVar()
        self.degree_of_fit_text.set(str(self.degree_of_fit))
        self.degree_readout_box=Tk.Entry(master=self.window, textvariable=self.degree_of_fit_text, width=2)
        self.degree_readout_box.place(relx=0.12,rely=0.9,anchor='center')


        self.rms_readout_ax=plt.figtext(0.13,0.01, str(self.rms_error), horizontalalignment='left')
        plt.figtext(0.14,0.1,'Degree of Fit')
        plt.figtext(0.13,0.01,'RMSE:', horizontalalignment='right')

        #self.axes=[self.ax,self.fit_increment_ax,self.fit_decrement_ax]
        plt.subplots_adjust(bottom=0.2)
        self.rect = Rectangle((0,0), 0, 0, facecolor='red', edgecolor='red', alpha=0.2)
        self.previous_rects=[]
        self.previous_rects_x=[]
        self.is_pressed=False
        self.is_drawing_new_rect=False
        self.x0 = None
        self.x1 = None
        self.y0,self.y1=self.ax.get_ylim()

        self.lines=[]
        self.ax.add_patch(self.rect)

        self.window.protocol("WM_DELETE_WINDOW", self.finished_clicked)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.degree_of_fit_text.trace('w', self.dof_changed)

        #self.inc_fit_button=Button(self.fit_increment_ax, '/\\')
        #self.dec_fit_button=Button(self.fit_decrement_ax, '\\/')
        self.UpdatePlots()
        self.inc_fit_button=Tk.Button(master=self.window,text='/\\',command=self.inc_clicked,width=2)
        self.inc_fit_button['font']=tkFont.Font(family='Helvetica',size=18)
        self.inc_fit_button.place(relx=0.12,y=self.degree_readout_box.winfo_y(),anchor='s')

        self.dec_fit_button=Tk.Button(master=self.window,text='\\/',command=self.dec_clicked,width=2)
        self.dec_fit_button['font']=tkFont.Font(family='Helvetica',size=18)
        self.dec_fit_button.place(relx=0.12,y=self.degree_readout_box.winfo_y()+int(self.degree_readout_box.winfo_height()),anchor='n')
        self.window.bind('<Configure>', self.configure)
        self.UpdatePlots()
        Tk.mainloop()

    def configure(self,event):
        self.inc_fit_button.place(relx=0.12,y=self.degree_readout_box.winfo_y(),anchor='s')
        self.dec_fit_button.place(relx=0.12,y=self.degree_readout_box.winfo_y()+int(self.degree_readout_box.winfo_height()),anchor='n')


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
        #elif event.inaxes == self.fit_increment_ax:
        #    self.inc_clicked()
        #elif event.inaxes == self.fit_decrement_ax:
        #    self.dec_clicked()

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
        #self.ax.plot(self.xdata, float('4e-11')+self.norm_flux/float('1e11'), 'k-',linewidth=1)
        #self.ax.plot(self.xdata, 1.2*np.mean(self.norm_flux/float('1e11'))+self.norm_flux/float('1e11'), 'k-',linewidth=1)
        self.ax.plot(self.xdata, self.norm_flux*np.mean(self.ydata)+np.mean(self.ydata),'k-',linewidth=1)
        self.ax.plot(self.xdata,self.ydata, 'k-', linewidth=1)
        self.ax.plot(self.xdata,self.cont_fit, 'b-', linewidth=1)
        #self.ax.plot(self.xdata,self.spec1d_object.flux_err_arr, color='0.5', linestyle='-', linewidth=1)
        self.ax.set_ylabel('Flux')
        self.ax.set_xlabel('Velocity')
        self.ax.set_xlim(self.xdata[0]-20,self.xdata[-1]+20)
        self.ax.set_ylim([-0.1*np.mean(self.ydata),2.5*np.mean(self.ydata)])
        self.y0, self.y1 = self.ax.get_ylim()
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
    	filename=self.spec1d_object.hdr['TARGNAME']+'_'+wav+'.dat'
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

    def add_spectra(self):
        text_window=Tk.Toplevel(self.window)
        width_inches=5.0
        height_inches=1.5
        dpi=float(100)
        text_window.geometry('%dx%d+%d+%d' % (width_inches*dpi, height_inches*dpi, 0, 0))
        text_window.wm_title('Dataset Entry')

        l = Tk.Label(text_window, text='Enter the datasets to be added to the current one.\nSeparate datasets by commas.\n(\'_x1d.fits\' will automatically be appended)',anchor='center')
        l.place(relx=0.5,y=10,anchor='n')

        datasets=Tk.StringVar()
        datasets.set('')
        box=Tk.Entry(master=text_window, textvariable=datasets, width=50)
        box.place(relx=0.5,y=75,anchor='center')
        load_button=Tk.Button(master=text_window, text='Load', command=lambda: self.load_spectra(text_window,datasets), width=10)
        load_button['font']=tkFont.Font(family='Helvetica', size=18)
        load_button.place(relx=0.3,rely=1.0,anchor='s')

        sum_button=Tk.Button(master=text_window, text='Sum',command=lambda: self.get_summed_spectrum(text_window,datasets),width=10)
        sum_button['font']=tkFont.Font(family='Helvetica',size=18)
        sum_button.place(relx=0.7,rely=1.0,anchor='s')

    def load_spectra(self,text_window,datasets):
        dataset_list=[x.strip() for x in datasets.get().split(',')]
        if dataset_list==['']:
            return
        dataset_list.insert(0,self.spec1d_object.hdr['filename'].lower())
        dataset_files=list(set([x+'_x1d.fits' for x in dataset_list if pth.isfile(x+'_x1d.fits')]))
        min_vel,max_vel=self.spec1d_object.vel_arr[0],self.spec1d_object.vel_arr[-1]
        total_exp_time=0
        self.specs_to_sum=[]
        fls_to_sum=[]
        for fl in dataset_files:
            temp=X1D(fl).get_spec_from_wav(self.cen_wav)
            if temp is not None:
                self.specs_to_sum.append(temp.get_vel_range(self.cen_wav,min_vel,max_vel))
                fls_to_sum.append(fl)

        #self.specs_to_sum=[X1D(fl).get_spec_from_wav(self.cen_wav).get_vel_range(self.cen_wav,min_vel,max_vel) for fl in dataset_files]
        total_exp_time=sum([float(a.hdr['EXPTIME']) for a in self.specs_to_sum])

        fls_to_sum=sorted(fls_to_sum)
        lst_text='\n'.join([fls_to_sum[i]+'\t'+str(self.specs_to_sum[i].hdr['EXPTIME'])+' sec\t'+str(round(self.specs_to_sum[i].hdr['EXPTIME']/total_exp_time,2)) for i in range(len(fls_to_sum))])
        lst_hdr='{:<18}'.format('Dataset')+'\t'+'{:<10}'.format('Exp Time')+'\t'+'Weight'
        lst_text=lst_hdr+'\n'+'-'*len(lst_hdr.expandtabs())+'\n'+lst_text

        lst_label=Tk.Label(text_window, text=lst_text,font='Courier',justify='left')
        text_window.geometry('%dx%d+%d+%d' % (500, text_window.winfo_height()+30*len(dataset_files), 0, 0))
        lst_label.place(relx=0.5,y=100,anchor='n')

    def get_summed_spectrum(self,text_window,datasets):
        low_lim,up_lim=self.xdata[0],self.xdata[-1]
        total_exp_time=sum([float(a.hdr['EXPTIME']) for a in self.specs_to_sum])
        base_grid=self.specs_to_sum[0].wav_arr
        summed_flux=np.zeros(len(base_grid),dtype=float)
        for spec in self.specs_to_sum:
            weight=float(spec.hdr['EXPTIME'])/total_exp_time
            # Interpolate new flux
            idxs=np.where(np.logical_and(base_grid>=spec.wav_arr[0],base_grid<=spec.wav_arr[-1]))[0]
            base_grid=base_grid[idxs]
            summed_flux=summed_flux[idxs]
            f=interp1d(spec.wav_arr,spec.flux_arr,kind='cubic')
            new_flux=f(base_grid)
            for i in range(len(base_grid)):
                summed_flux[i]+=weight*new_flux[i]

        # Store results
        self.spec1d_object.wav_arr=base_grid
        self.spec1d_object.flux_arr=summed_flux
        self.spec1d_object=self.spec1d_object.conv_wav_to_vel(self.cen_wav).get_vel_range(self.cen_wav,low_lim,up_lim)
        self.xdata=self.spec1d_object.vel_arr
        self.ydata=self.spec1d_object.flux_arr

        # Reset mask and rect drawings
        self.mask=np.array([1]*len(self.xdata))
        self.rect = Rectangle((0,0), 0, 0, facecolor='red', edgecolor='red', alpha=0.2)
        self.previous_rects=[]
        self.previous_rects_x=[]
        self.is_pressed=False
        self.is_drawing_new_rect=False
        self.x0 = None
        self.y0 = self.ax.get_ylim()[0]
        self.x1 = None
        self.y1 = self.ax.get_ylim()[1]

        self.UpdatePlots()

        # Take over plot to check indiv and summed spectra
        #self.ax.cla()
        #i=1
        #for spec in self.specs_to_sum:
        #    self.ax.plot(spec.wav_arr,spec.flux_arr+i*float('1e-11'),linewidth=1)
        #    i+=1
        #self.ax.plot(base_grid,summed_flux,linewidth=1,color='k')
        #self.ax.figure.canvas.draw()
        text_window.destroy()
