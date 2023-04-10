#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of modules used to extract data from fits files from, 
e.g., model atmospheres from Kurucz to stellar spectra from FIES.

"""
import glob
import numpy as np
import astropy.io.fits as pyfits
from astropy.time import Time
from barycorrpy import get_BC_vel, utc_tdb
from astropy import constants as const
#from scipy.constants import speed_of_light

# =============================================================================
# Templates
# =============================================================================

def readPhoenix(filename, wl_min=3600,wl_max=8000):
	'''Read Phoenix stellar template.

	
	:param filename: Path to template.
	:type filename: str

	:param wl_min: Minimum wavelength to keep.
	:type wl_min: float, optional

	:param wl_max: Maximum wavelength to keep.
	:type wl_max: float, optional

	:return: template wavelength, template flux
	:rtype: array, array

	'''
	fhdu = pyfits.open(filename)
	flux = fhdu[0].data
	fhdu.close()

	whdu = pyfits.open(os.path.dirname(filename)+'/'+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
	wave = whdu[0].data
	whdu.close()

	#from Ciddor (1996) to go from vacuum to air wavelength
	sig2 = (1e4/wave)**2.0
	f = 1.0 + 0.05792105/(238.0185-sig2) + 0.00167917/(57.362-sig2)
	wave /= f
	
	keep = (wave > wl_min) & (wave < wl_max)
	wave, flux = wave[keep], flux[keep]
	flux /= np.amax(flux)

	#flux /= np.median()
	return wave, flux

def readKurucz(filename):
	'''Read Kurucz/ATLAS9 stellar template.

	Extract template wavelength and flux from :cite:t:`Castelli2003`.
	
	Available `here <http://130.79.128.5/ftp/more/splib120/>`_.
	
	:param filename: Path to template.
	:type filename: str

	:return: template wavelength, template flux
	:rtype: array, array

	'''
	temp = pyfits.open(filename)
	th = temp[0].header
	flux = temp[0].data[0]
	temp.close()

	# create wavelengths
	wls = th['CRVAL1']
	wld = th['CDELT1']
	wave = np.arange(len(flux))*wld + wls
	return wave, flux 

# =============================================================================
# FIES 
# =============================================================================

def extractFIES(filename):
	'''Extract FIES data (FIEStool).

	Extract the flux and wavelength from a FIES fits file.

	Adapted and extended from functions written by R. Cardenes, J. Jessen-Hansen, and J. Sinkbaek.

	:param filename: Path to FIES fits file.
	:type filename: str

	:return: wavelength, flux, error, header
	:rtype: array, array, array, fits-header

	'''
	#Read in the FITS file
	hdr = pyfits.getheader(filename)
	flux = pyfits.getdata(filename)
	#Extract the flux
	fts  = pyfits.open(filename)
	flux = fts[0].data
	fts.close()
	#Extract the error -- available for newer FIES data
	if flux.ndim == 3:
		error = np.copy(flux[2,:,:])
		flux = np.copy(flux[0,:,:])
	else:
		error = np.ones(flux.shape)
	
	# Figure out which header cards we want
	cards = [x for x in sorted(hdr.keys()) if x.startswith('WAT2_')]
	# Extract the text from the cards
	# We're padding each card to 68 characters, because PyFITS won't
	# keep the blank characters at the right of the text, and they're
	# important in our case
	text_from_cards = ''.join([hdr[x].ljust(68) for x in cards])
	data = text_from_cards.split('"')

	#Extract the wavelength info (zeropoint and wavelength increment for each order) to arrays
	info = [x for x in data if (x.strip() != '') and ("spec" not in x)]
	zpt = np.array([x.split(' ')[3] for x in info]).astype(np.float64)
	wstep = np.array([x.split(' ')[4] for x in info]).astype(np.float64)
	npix = np.array([x.split(' ')[5] for x in info]).astype(np.float64)
	orders = np.array([x.split(' ')[0] for x in info])
	#no_orders = len(orders)
	wave = np.empty(flux.shape)

	#Create record array names to store the flux in each order
	col_names = [ 'order_'+order for order in orders]
	
	#Create wavelength arrays
	for i,col_name in enumerate(col_names):
		wave[i,:] = zpt[i] + np.arange(npix[i]) * wstep[i]
		#wave[0,i] = zpt[i] + np.arange(npix[i]) * wstep[i]

	return wave, flux, error, hdr

def sortFIES(path):
	'''Sort files in Science/ThAr.

	Function that returns list of science and/or ThAr spectra from FIES.
	
	:param path: Path to files.
	:type path: str
	
	:return: list of science spectra, list of ThAr spectra
	:rtype: list, list
	
	'''
	files = glob.glob(path+'*wave.fits')

	science = []
	thar = []

	for file in files:
		hdr = pyfits.getheader(file)
		star = hdr['OBJECT']
		if star == 'ThAr':
			thar.append(file)
		else:
			science.append(file)

	return science, thar

def getBarycorrs(filename, rvsys):
	'''Barycentric correction.

	Function to correct for the barycentric velocity, 
	and convert the time of the observation to BJD TDB.

	:param filenames: List of filenames.
	:type filenames: list
	
	:param rvs: Array of radial velocities derived from filenames.
	:type rvs: array

	.. note::
		Could be altered to also be able to deal with other instruments.

	'''
	loc = 'Roque de los Muchachos'
	#rvs_cor = np.empty_like(rvs)
	#filenames = list(filenames)
	#bjds = np.empty(len(filenames))#rvs.shape)
	#rvs_cor = np.empty_like(bjds)
	#for i, filename in enumerate(filenames):
	#_, _, jd, _, star, _, _, _, hdr = extractFIESold(filename, return_hdr=True)
	_, _, jd, _, star, _, _, hdr = extractFIESold(filename, return_hdr=True)

	ra = hdr['OBJRA']*15.0		# convert unit
	dec = hdr['OBJDEC']
	#z_meas = rvs[i]*1000 / const.c.value #speed_of_light
	z_meas = rvsys*1000 / const.c.value #speed_of_light
	rv_cor, _, _ = get_BC_vel(jd, ra=ra, dec=dec, starname=star, ephemeris='de432s', obsname=loc, zmeas=z_meas)
	rv_cor = rv_cor / 1000	 # from m/s to km/s
	#rvs_cor[i] = rv_cor
	bjd, _, _, = utc_tdb.JDUTC_to_BJDTDB(jd, ra=ra, dec=dec, starname=star, obsname=loc)
	#bjds[i] = bjd
	return bjd, rv_cor-rvsys


# =============================================================================
# Grids and resolution
# =============================================================================

def velRes(R=67000,s=2.1):
	'''Velocity resolution.
	
	The resolution of the spectrograph in velocity space.

	.. math::
		\Delta v = \\frac{c}{R \cdot s}
		
	where :math:`c` is the speed of light, :math:`R` is the spectral resolution, and :math:`s` is the spectral sampling in pixels per spectral element.
	
	The default values are for the FIES spectrograph, collected from:
	http://www.not.iac.es/instruments/fies/fies-commD.html#wavelengths

	:param R: Spectral resolution. Default is 67000.
	:type R: float
	:param s: Spectral sampling in pixels per spectral element. Default is 2.1.
	:type s: float

	:return: Velocity resolution in km/s.
	:rtype: float
	
	'''
	dv = 1e-3*const.c.value/(R*s)
	return dv

def grids(rvr=401,R=67000,s=2.1):
	'''Grids for CCFs.

	Create the CCF grids for the given radial velocity range and resolution.
	As the velocity grid will be scaled, the radial velocity range is altered slightly.
	Therefore, there is a difference between the desired radial velocity range and the actual one.

	A (typically) finer grid is also created for the CCFs, with a resolution of 250 m/s.

	:param rvr: (Desired) radial velocity range for the CCF in km/s. Default is 401.
	:type rvr: int
	:param R: Spectral resolution. Default is 67000 (FIES, see :py:func:`velRes`).
	:type R: float
	:param s: Spectral sampling in pixels per spectral element. Default is 2.1 (FIES, see :py:func:`velRes`).
	:type s: float

	:return: CCF grid, CCF error grid, actual radial velocity range, finer CCF grid, finer CCF error grid, finer radial velocity range
	:rtype: array, array, int, array, array, int

	'''

	## Get the velocity resolution
	dv = velRes(R=R,s=s)

	## This is the actual RV range
	arvr = int(rvr/dv)
	if arvr % 2 == 0: arvr += 1
	ccfs = np.zeros(arvr)
	ccferrs = np.zeros(arvr)

	## Some quantities are better determined using a finer grid
	dv25 = 0.25 #250 m/s
	## Otherwise the same
	arvr25 = int(rvr/dv25)
	if arvr25 % 2 == 0: arvr25 += 1
	ccfs25 = np.zeros(arvr25)
	ccferrs25 = np.zeros(arvr25)

	return ccfs, ccferrs, arvr, ccfs25, ccferrs25, arvr25

# =============================================================================
# Old way of extracting FIES data
# =============================================================================

def extractFIESold(filename,return_hdr=False,check_ThAr=True):
	'''Extract FIES data (FIEStool).

	Reads a wavelength calibrated spectrum in IRAF (FIEStool) format.
	Returns a record array with wavelength and flux order by order as a two column 
	array in data['order_i'].
	
	Adapted and extended from functions written by R. Cardenes and J. Jessen-Hansen.
	
	:param filename: Name of .fits file.
	:type filename: str
	
	:return: observed spectrum, wavelength and flux order by order, number of orders for spectrum, name of object, date of observations in UTC, exposure time in seconds
	:rtype: array, int, str, str, float 
	
	'''
	try:
		hdr = pyfits.getheader(filename)
		star = hdr['OBJECT']
		if star == 'ThAr' and check_ThAr:
			raise Exception('\n###\nThis looks like a {} frame\nPlease provide a wavelength calibrated file\n###\n'.format(star))

		date = hdr['DATE-OBS']
		date_mid = hdr['DATE-AVG']
		bjd = Time(date_mid, format='isot', scale='utc').jd
		exp = hdr['EXPTIME']
		vhelio = hdr['VHELIO']
	except Exception as e:
		print('Problems extracting headers from {}: {}'.format(filename, e))
		print('Is filename the full path?')

	# Figure out which header cards we want
	cards = [x for x in sorted(hdr.keys()) if x.startswith('WAT2_')]
	# Extract the text from the cards
	# We're padding each card to 68 characters, because PyFITS won't
	# keep the blank characters at the right of the text, and they're
	# important in our case
	text_from_cards = ''.join([hdr[x].ljust(68) for x in cards])
	data = text_from_cards.split('"')

	#Extract the wavelength info (zeropoint and wavelength increment for each order) to arrays
	info = [x for x in data if (x.strip() != '') and ("spec" not in x)]
	zpt = np.array([x.split(' ')[3] for x in info]).astype(np.float64)
	wstep = np.array([x.split(' ')[4] for x in info]).astype(np.float64)
	npix = np.array([x.split(' ')[5] for x in info]).astype(np.float64)
	orders = np.array([x.split(' ')[0] for x in info])
	no_orders = len(orders)

	#Extract the flux
	fts  = pyfits.open(filename)
	data = fts[0].data
	fts.close()
	wave = data.copy()

	#Create record array names to store the flux in each order
	col_names = [ 'order_'+order for order in orders]

	#Create wavelength arrays
	for i,col_name in enumerate(col_names):
		wave[0,i] = zpt[i] + np.arange(npix[i]) * wstep[i]

	#Save wavelength and flux order by order as a two column array in data['order_i']
	data = np.rec.fromarrays([np.column_stack([wave[0,i],data[0,i]]) for i in range(len(col_names))],
		names = list(col_names))
	if return_hdr:
		return data, no_orders, bjd, vhelio, star, date, exp, hdr
	return data, no_orders, bjd, vhelio, star, date, exp

def getFIES(data,order=40):
	'''Extract FIES spectrum.

	Extract calibrated spectrum from FIES at given order.
		
	:params data: Observed spectrum.
	:type data: array
	:param order: Extract spectrum at order. Default 30.
	:type order: int, optional
	
	:return: observed wavelength, observed raw flux
	:rtype: array, array

	'''
	arr = data['order_{:d}'.format(order)]
	wl, fl = arr[:,0], arr[:,1]
	return wl, fl