import numpy as np
from astropy.io import fits
from x1d_tools import Spectrum1D

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
        hdr.append(card=('FILENAME',self.hdr['ROOTNAME']))
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
                print self.hdr['ROOTNAME']+': '+'Wavelength must be between', self.hdr['minwave'], 'and', self.hdr['maxwave']
                return None
            for spectrum in self.spectra:
                if spectrum.wav_arr[0] < w and spectrum.wav_arr[-1] > w:
                    spec_idx.append(self.spectra.index(spectrum))
            if len(spec_idx)==2:
                coadd='a'
                while coadd[0].lower() not in ('y','n'):
                    coadd= raw_input(self.hdr['ROOTNAME']+': '+str(w)+" A covers multiple orders, co-add orders? (This will only return the region covered by both orders)\n([Y]/N): ")
                    if coadd=='':
                        coadd='Y'
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
