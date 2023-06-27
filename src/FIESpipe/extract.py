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
from astropy.utils.data import download_file
import pickle
import os
#from scipy.constants import speed_of_light

# =============================================================================
# Templates
# =============================================================================
def getPhoenix(teff,logg=4.5,feh=0.0,alpha=0.0,
	url='ftp://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/',
	cache=True):
	'''Download PHOENIX template

	Download PHOENIX template using the effective temperature, surface gravity, metallicity and alpha enhancement values closest to the values provided.

	Effective temperature range: 2300-7100 K in steps of 100 K, 7200-13200 K in steps of 200 K.
	Surface gravity range: 0.0-6.0 in steps of 0.5 dex.
	Metallicity values: -4.0, -3.0, -2.0, -1.0, 0.0, 0.2, 0.5 dex.
	Alpha enhancement values: -0.2 to 1.2 in steps of 0.2 dex.

	Similar to https://github.com/Hoeijmakers/StarRotator/blob/master/lib/stellar_spectrum.py#L231.

	:param teff: Effective temperature in K.
	:type teff: int

	:param logg: Surface gravity. Default is 4.5.
	:type logg: float

	:param feh: [Fe/H]. Default is '0.0'.
	:type feh: float

	:param alpha: [alpha/Fe]. Default is '0.0'.
	:type alpha: float

	:param url: Path to the template files. Default is 'ftp://phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/'.
	:type path: str

	:param cache: Cache the downloaded file. Default is True.
	:type cache: bool

	:returns: File names of the downloaded templates.
	:rtype: str
	'''

	## Effective temperature, Teff
	teffs = np.arange(2300, 7100, 100,dtype=int)
	teffs = np.append(teffs,np.arange(7200,13200,200,dtype=int))
	teffq = teffs[np.argmin(np.abs(teffs-teff))]
	if teffq < 10000:
		teffq = '0{:d}'.format(teffq)
	else:
		teffq = '{:d}'.format(teffq)

	## Surface gravity, logg
	loggs = np.arange(0, 6.5, 0.5)
	loggq = loggs[np.argmin(np.abs(loggs-logg))]

	## Metallicity, [Fe/H]
	fehs = np.arange(-4, -1, 1)
	fehs = np.append(fehs,np.arange(-1.5, 1.5, 0.5))
	fehq = fehs[np.argmin(np.abs(fehs-feh))]
	if fehq > 0:
		fehq = '+{:.1f}'.format(fehq)
	else:
		fehq = '-{:.1f}'.format(abs(fehq))

	## Alpha enhancement, [alpha/Fe]
	alphas = np.arange(-0.2, 1.4, 0.2)
	alphaq = alphas[np.argmin(np.abs(alphas-alpha))]
	if alphaq < 0:
		alphaq = '.Alpha={:.2f}'.format(alphaq)
	elif alphaq < 0:
		alphaq = '.Alpha=+{:.1f}'.format(alphaq)
	else:
		alphaq = ''

	## Name of wavelength file
	waveurl = url+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

	## Name of template file
	specurl = url+'PHOENIX-ACES-AGSS-COND-2011/Z{:s}{:s}'.format(fehq,alphaq)+\
		'/lte{:s}-{:.2f}{:s}{:s}'.format(teffq,loggq,fehq,alphaq)+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

	## Download files
	spec = download_file(specurl, cache=cache)
	wave = download_file(waveurl, cache=cache)
	
	return spec, wave

def readPhoenix(specname, wavename='', wl_min=3600,wl_max=8000,to_air=True):
	'''Read PHOENIX stellar template.

	Extract template wavelength and flux from :cite:t:`Husser2013`.
	
	Avaliable `here <http://phoenix.astro.physik.uni-goettingen.de/>`__.

	To go from vacuum to air wavelength, the function uses Eq. (9-10) from :cite:t:`Husser2013`.

	:param specname: Path to template.
	:type specname: str

	:param wavename: Path to wavelength file. Default is ``''``, no wavelength.
	:type wavename: str
	
	:param wl_min: Minimum wavelength to keep. Default is 3600 Å.
	:type wl_min: float, optional

	:param wl_max: Maximum wavelength to keep. Default is 8000 Å.
	:type wl_max: float, optional

	:param to_air: Convert wavelength to air. Default is True.
	:type to_air: bool, optional

	:return: template wavelength, template flux
	:rtype: array, array

	'''
	fhdu = pyfits.open(specname)
	flux = fhdu[0].data
	fhdu.close()
	try:
		whdu = pyfits.open(wavename)
		wave = whdu[0].data
		whdu.close()

		if to_air:
			## go from vacuum to air wavelength
			sig2 = np.power(1e4/wave,2)
			ff = 1.0 + 0.05792105/(238.0185-sig2) + 0.00167917/(57.362-sig2)
			wave /= ff
		
		keep = (wave > wl_min) & (wave < wl_max)
		wave, flux = wave[keep], flux[keep]
		flux /= np.amax(flux)

		return wave, flux
	except FileNotFoundError:
		print('No wavelength file found. Returning flux only.')
		return flux

def getKurucz(teff, logg=4.5, feh=0.0, alpha=0.0,
	url ='http://130.79.128.5/ftp/more/splib120/',
	cache=True
	):
	'''Download ATLAS9/Kurucz template

	Download ATLAS9/Kurucz template using the effective temperature, surface gravity, metallicity and alpha enhancement values closest to the values provided.

	Effective temperature range: 3500-7250 K in steps of 250 K.
	Surface gravity range: 0.0-5.0 in steps of 0.5 dex.
	Metallicity values: -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5 dex.
	Alpha enhancement values: 0.0, 0.4 dex.

	:param teff: Effective temperature.
	:type teff: int
	:param logg: Surface gravity. Default is 4.5.
	:type logg: float
	:param feh: [Fe/H]. Default is '0.0'.
	:type feh: float
	:param alpha: [alpha/Fe]. Default is '0.0'.
	:type alpha: float
	:param url: Path to the template files. Default is 'http://130.79.128.5/ftp/more/splib120/'.
	:type path: str
	:param cache: Cache the downloaded file. Default is True.
	:type cache: bool

	:returns: File name of the downloaded template.
	:rtype: str

	'''

	## Effective temperature, Teff
	teffs = np.arange(3500, 7250, 250,dtype=int)
	teffq = teffs[np.argmin(np.abs(teffs-teff))]

	## Surface gravity, logg
	loggs = np.arange(0, 5.5, 0.5)
	loggq = loggs[np.argmin(np.abs(loggs-logg))]*10
	loggq = '{:02d}'.format(int(loggq))

	## Metallicity, [Fe/H]
	fehs = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0,0.2,0.5])
	fehq = fehs[np.argmin(np.abs(fehs-feh))]*10
	if fehq < 0:
		fehq = 'm{:02d}'.format(np.abs(int(fehq)))
	else:
		fehq = 'p{:02d}'.format(np.abs(int(fehq)))
	
	## Alpha enhancement, [alpha/Fe]
	alphas = np.array([0.0, 0.4])
	alphaq = alphas[np.argmin(np.abs(alphas-alpha))]*10
	alphaq = 'p{:02d}'.format(int(alphaq))

	## Download file
	fname = url+'{:d}_{:s}_{:s}{:s}.ms.fits.gz'.format(teffq, loggq, fehq, alphaq)
	file = download_file(fname, cache=True)
	return file

def readKurucz(filename):
	'''Read Kurucz/ATLAS9 stellar template.

	Extract template wavelength and flux from :cite:t:`Castelli2003`.
	
	Available `here <http://130.79.128.5/ftp/more/splib120/>`__.
	
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
	
	#flux = pyfits.getdata(filename)
	
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
		if not 'ThAr' in hdr['OBJECT']:
			print('No flux errors available for {}'.format(filename))
			print('Setting all errors to 1')
	
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
		elif 'ThAr' in star:
			thar.append(file)
		else:
			science.append(file)

	return science, thar

def getBarycorrs(filename, rvmeas):
	'''Barycentric correction.

	Function to correct for the barycentric velocity, 
	and convert the time of the observation to BJD TDB.

	:param filename: FIES fits file.
	:type filename: str
	
	:param rvsys: The measured radial velocity. See :py:mod:`barycorrpy` @ https://github.com/shbhuk/barycorrpy/blob/master/barycorrpy/barycorrpy.py#L99.
	:type rvsys: float

	:return: BJD TDB, barycentric velocity correction
	:rtype: float, float

	.. note::
		- Could be altered to also be able to deal with other instruments.
		- Exception omits star name, defaults to RA/DEC.
		- Split into two functions, one for barycentric correction, one for BJD TDB?

	'''
	loc = 'Roque de los Muchachos'
	
	## REMOVE
	#rvs_cor = np.empty_like(rvs)
	#filenames = list(filenames)
	#bjds = np.empty(len(filenames))#rvs.shape)
	#rvs_cor = np.empty_like(bjds)
	#for i, filename in enumerate(filenames):
	#_, _, jd, _, star, _, _, _, hdr = extractFIESold(filename, return_hdr=True)
	#_, _, jd, _, star, _, _, hdr = extractFIESold(filename, return_hdr=True)

	_, _, _, hdr = extractFIES(filename)
	
	date_mid = hdr['DATE-AVG']
	jd = Time(date_mid, format='isot', scale='utc').jd
	star = hdr['OBJECT']

	ra = hdr['OBJRA']*15.0		# convert unit
	dec = hdr['OBJDEC']
	
	## REMOVE
	#z_meas = rvs[i]*1000 / const.c.value #speed_of_light
	
	z_meas = rvmeas*1000 / const.c.value #speed_of_light
	try:
		rv_cor, _, _ = get_BC_vel(jd, ra=ra, dec=dec, starname=star, ephemeris='de432s', obsname=loc, zmeas=z_meas)
		bjd, _, _, = utc_tdb.JDUTC_to_BJDTDB(jd, ra=ra, dec=dec, starname=star, obsname=loc)
	except ValueError:
		rv_cor, _, _ = get_BC_vel(jd, ra=ra, dec=dec, ephemeris='de432s', obsname=loc, zmeas=z_meas)
		bjd, _, _, = utc_tdb.JDUTC_to_BJDTDB(jd, ra=ra, dec=dec, obsname=loc)

	rv_cor = rv_cor / 1000	 # from m/s to km/s
	
	## REMOVE
	#rvs_cor[i] = rv_cor
	#bjds[i] = bjd

	return bjd, rv_cor[0]-rvmeas

# =============================================================================
# Group by epochs
# =============================================================================

def groupByEpochs(path,
	thresh = 1/24., #hours
	):
	'''Group spectra by epochs.

	Function to group spectra by epochs, in the sense that
	spectra obtained in quick succession are grouped together.
	Intended for observing strategies of the type:

	.. graphviz::

		digraph acq {
			rankdir="LR";
			th1 -> sci1 -> sci2 -> sci3 -> th2;
			th1 [shape=box, label="ThAr"];
			th2 [shape=box, label="ThAr"];
			sci1 [label="Science"];
			sci2 [label="Science"];
			sci3 [label="Science"];
		}
	
	:param path: Path to files.
	:type path: str
	:param thresh: Threshold for grouping epochs, in days. Default  ``1/24`` (1 hour).
	:type thresh: float
	
	:returns: Dictionary with epochs (labeled as 1,2,3,...) as keys, and lists of filenames and timestamps (:math:`\mathrm{BJD}_\mathrm{TDB}`) as values.
	:rtype: dict

	'''

	## Group into science and ThAr frames
	filenames, tharnames = sortFIES(path)
	filenames.sort()

	## Store BJDs and filenames
	bjds = np.array([])
	scifiles = np.array([],dtype='str')

	## Get BJDs (TDB)
	for filename in filenames:
		bjd, _ = getBarycorrs(filename,0)
		bjds = np.append(bjds,bjd)
		scifiles = np.append(scifiles,filename)

	## Group by epochs
	epochs = {}
	booked = []
	ll = 1
	for ii, bjd in enumerate(bjds):
		if bjd in booked: 
			continue
		booked.append(bjd)
		diff = np.abs(bjds - bjd)
		idxs = np.where(diff < thresh)[0]
		for idx in idxs:
			bjdnext = bjds[idx]
			if bjdnext not in booked: booked.append(bjdnext)
		epochs[ll] = {'names':list(scifiles[idxs]),'bjds':bjds[idxs]}
		ll += 1
	return epochs

# =============================================================================
# Grids and resolution
# =============================================================================

def velRes(R=67000,s=2.1,fib=None):
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
	:param fib: Fibre number. Default is ``None``. If ``None``, the default values are used (fibre 4).
	:type fib: int

	:return: Velocity resolution in km/s.
	:rtype: float
	
	'''
	if fib == 1:
		R, s = 25000, 5.9
	elif fib == 2:
		R, s = 45000, 3.2
	elif fib == 3:
		R, s = 45000, 3.2
	elif fib == 4:
		R, s = 67000, 2.1

	dv = 1e-3*const.c.value/(R*s)
	return dv

def grids(rvr=401,R=67000,s=2.1,fib=None):
	'''Grids for CCFs.

	Create the CCF grids for the given radial velocity range and resolution.
	As the velocity grid will be scaled, the radial velocity range is altered slightly.
	Therefore, there is a difference between the desired radial velocity range and the actual one.

	A (typically) finer grid is also created for the CCFs, with a resolution of 250 m/s.

	:param rvr: (Desired) radial velocity range for the CCF in km/s. Default is 401.
	:type rvr: int
	:param R: Spectral resolution. Default is 67000 (FIES fibre 4, see :py:func:`velRes`).
	:type R: float
	:param s: Spectral sampling in pixels per spectral element. Default is 2.1 (FIES, see :py:func:`velRes`).
	:type s: float
	:param fib: Fibre number. Default is ``None``. If ``None``, the default values are used (fibre 4).
	:type fib: int
	
	:return: CCF grid, CCF error grid, actual radial velocity range, finer CCF grid, finer CCF error grid, finer radial velocity range
	:rtype: array, array, int, array, array, int

	'''

	## Get the velocity resolution
	dv = velRes(R=R,s=s,fib=fib)

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

# def getFIES(data,order=40):
# 	'''Extract FIES spectrum.

# 	Extract calibrated spectrum from FIES at given order.
		
# 	:params data: Observed spectrum.
# 	:type data: array
# 	:param order: Extract spectrum at order. Default 30.
# 	:type order: int, optional
	
# 	:return: observed wavelength, observed raw flux
# 	:rtype: array, array

# 	'''
# 	arr = data['order_{:d}'.format(order)]
# 	wl, fl = arr[:,0], arr[:,1]
# 	return wl, fl


# =============================================================================
# Read data product
# =============================================================================

def readDataProduct(filename):
	with open(filename, 'rb') as f:
		loaded_dict = pickle.load(f)
	
	return loaded_dict

# =============================================================================
# ESPRESSO
# =============================================================================

def extractESPRESSO(filename,to_air=True):
	'''Extract ESPRESSO data.
	
	Extract header and flux from ESPRESSO S2D file.

	.. note::
		The wavelength is in vacuum, so it is converted to air following :cite:t:`Husser2013`.

	:param filename: Path to ESPRESSO fits file.
	:type filename: str

	:param to_air: Convert wavelength to air. Default is True.
	:type to_air: bool, optional

	:return: wavelength, flux, error, header
	:rtype: array, array, array, fits-header

	'''

	## Extract the data from the FITS file
	fts = pyfits.open(filename)
	## Header
	hdr = fts[0].header
	
	## Extract the flux, error, wave
	flux = fts[1].data
	err = fts[2].data
	wave = fts[4].data
	fts.close()
	orders = flux.shape[0]
	
	## Orders are repeated, 
	## so only keep every second order
	## which seems to have been filtered
	wl = np.zeros((orders//2,flux.shape[1]))
	fl = np.zeros((orders//2,flux.shape[1]))
	el = np.zeros((orders//2,flux.shape[1]))
	ii = 0
	for order in range(1,orders+1,2):
		w = wave[order,:]
		## Wavelength is in vacuum, convert to air
		if to_air:
			## go from vacuum to air wavelength
			sig2 = np.power(1e4/w,2)
			ff = 1.0 + 0.05792105/(238.0185-sig2) + 0.00167917/(57.362-sig2)
			w /= ff


		wl[ii,:] = w
		fl[ii,:] = flux[order,:]
		el[ii,:] = err[order,:]
		ii += 1
	
	return wl, fl, el, hdr