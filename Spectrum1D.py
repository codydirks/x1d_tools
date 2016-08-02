from copy import deepcopy
import numpy as np

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
