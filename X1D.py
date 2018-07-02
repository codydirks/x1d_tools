import numpy as np
from astropy.io import fits
from x1d_tools import Spectrum1D
from copy import copy, deepcopy
from scipy.interpolate import interp1d

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
        #self.spectra = [Spectrum1D(spectrum, hdr) for spectrum in hdulist[1].data]
        if len(hdulist)==2:
            self.spectra = [Spectrum1D(spectrum, hdr) for spectrum in hdulist[1].data]
        else:
            # If multiple data exposures, need to co-add individual exposures
            spec_list=[]
            exposures = hdulist[1:]
            total_exp_time=sum([e.header['exptime'] for e in exposures])
            for spec_idx in range(len(exposures[0].data)):
                temp_spec=Spectrum1D(exposures[0].data[spec_idx],hdr)
                base_grid=deepcopy(exposures[0].data[spec_idx][2])
                summed_flux=np.zeros(len(base_grid),dtype=float)
                errs=np.zeros(len(base_grid),dtype=float)
                for exp in exposures:
                    weight=float(exp.header['exptime'])/total_exp_time
                    idxs=np.where(np.logical_and(base_grid>=exp.data[spec_idx][2][0],base_grid<=exp.data[spec_idx][2][-1]))[0]
                    base_grid=base_grid[idxs]
                    summed_flux=summed_flux[idxs]
                    errs=errs[idxs]
                    f=interp1d(exp.data[spec_idx][2],exp.data[spec_idx][6],kind='cubic')
                    ferr=interp1d(exp.data[spec_idx][2],exp.data[spec_idx][7],kind='cubic')
                    new_flux=f(base_grid)
                    new_errs=ferr(base_grid)
                    for elem in range(len(base_grid)):
                        summed_flux[elem] += weight*new_flux[elem]
                        errs[elem] += 1./(new_errs[elem]**2)

                temp_spec.wav_arr=base_grid
                temp_spec.flux_arr=summed_flux
                temp_spec.flux_err_arr=1./np.sqrt(errs)
                temp_spec.nelem=len(base_grid)
                temp_spec.hdr['exptime']=total_exp_time
                spec_list.append(temp_spec)
            self.spectra = spec_list

    def __repr__(self):
        return self.hdr['filename']

    def __str__(self):
        return self.hdr['filename']

    def get_spec_from_wav(self, w):
        nearest=1000
        spec_idx=[]
        spec_count=0
        if hasattr(w, '__len__')==False:
            mn=min(self.spectra[-1].wav_arr)
            mx=max(self.spectra[0].wav_arr)
            if w < mn or w > mx:
                print self.hdr['ROOTNAME']+': '+'Wavelength must be between', round(mn,1), 'and', round(mx,1)
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
