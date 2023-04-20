#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different workflows that stitch the various modules in :py:mod:`FIESpipe` together to obtain the parameters of interest.

"""
import os
import scipy.signal as scsig
from .extract import *
from .derive import *
path = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Standard/default
# =============================================================================

def standard(filename,
	ins='FIES',
	orders=[],
	exclude=[],
	rvsys=None,
	gauss=False,
	pidx=3,
	rvr=41,	#RV range
	chi2rvr=20, #RV range for chi2 minimization
	crmit=False, #cosmic ray mitigation
	out=True,sigma=3, #RV outlier rejection
	npoly=2, #degree of polynomial fit, normalization
	CaIIH=3968.47,
	CaIIK=3933.664,
	width=1.09,
	Rsection_low=3991.07,
	Rsection_high=4011.07,
	Vsection_low=3891.07,
	Vsection_high=3911.07,
	outdir=None,
	return_data=False,
	verbose=False,
	):
	'''Standard workflow for FIES data extraction.

	A collection of calls to the various modules in :py:mod:`FIESpipe` to obtain the parameters of interest.
	This processing routine should work regardless of how the data has been acquired.
	
	Flowchart of the workflow:

	.. graphviz::

		digraph G {
			size ="4,4";
			filename -> BJD_TDB, BVC;
			filename -> spectrum;
			template -> spectrum;
			spectrum [shape=box]; /* this is a comment */
			spectrum -> order [weight=8,label="Each order"];
			order -> RV;
			order -> Sidx;
			order -> CCF;
			BVC -> RV [style=dotted];
			edge [color=red]; // so is this
			CCF -> BIS [label="Summed CCF all orders"];
			node [shape=box,style=filled,color=".7 .3 1.0"];
			RV -> product [label="Weighted average"];
			BJD_TDB -> product;
			Sidx -> product;
			CCF -> product;
			BIS -> product;
		}
	
	S-index derived from the Ca II H and K lines following the approach in Section 3.1.2 of :cite:t:`BoroSaikia2018`.
	


	.. note::
		- For poor data/low S/N, the Gaussian fits can crash. ``np.nan`` is inserted into the output dictionary.
		- For now, the template is hardcoded. This should be fixed in the future.
		- Oversampled CCFs are hardcoded to be 0.25 km/s.
		- BIS error seems rather large. Determined from oversampled CCFs, but :cite:t:`Lafarga2020` uses the actual velocity resolution and gets a much smaller error.
		- S-index should perhaps be estimated from the merged spectrum instead?
			* ...merged spectrum should perhaps also be used for the first guess for the systemic velocity?
			* ...and to estimate :math:`T_\mathrm{eff}`, :math:`\log g`, and :math:`[\mathrm{Fe/H}]`?
		- RV :math:`\chi^2` minimization can be done by fitting a gaussian. 
			* Default is a second order polynomial fit as in :cite:t:`Zechmeister2018`.
			* This seems more robust, but the gaussian fit seems to have less RV drift across the orders, because the wings are taking into account.
			* The weighted mean and error seem to agree quite well between the two methods.

	:param filename: Name of the file to be processed.
	:type filename: str
	:param ins: Instrument used to acquire the data. Currently only FIES is supported.
	:type ins: str
	:param orders: List of orders to be processed. If empty, all orders are processed.
	:type orders: list
	:param exclude: List of orders to be excluded from the processing.
	:type exclude: list
	:param rvsys: Systemic RV in km/s of the target. If not provided, an estimate is calculated.
	:type rvsys: float
	:param gauss: If ``True``, the :math:`\chi^2` RVs are fitted with a gaussian. If ``False``, a second order polynomial is fitted.
	:type gauss: bool
	:param pidx: Number of indices on either side of peak for polynomial used to fit the :math:`\chi^2` RVs. Default is 3.
	:type pidx: int
	:param rvr: RV range in km/s for the CCFs.
	:type rvr: float
	:param chi2rvr: RV range in km/s for the :math:`\chi^2` minimization.
	:type chi2rvr: float
	:param crmit: Cosmic ray mitigation. If ``False``, no mitigation is performed.
	:type crmit: bool
	:param out: Outlier rejection for :math:`\chi^2` RVs. If ``True``, the :math:`\chi^2` RVs are rejected if they are more than ``sigma`` away from the median.
	:type out: bool
	:param sigma: Number of sigma for the outlier rejection.
	:type sigma: float
	:param npoly: Degree of the polynomial used to normalize the spectrum.
	:type npoly: int
	:param CaIIH: Wavelength of the Ca II H line in Angstroms.
	:type CaIIH: float
	:param CaIIK: Wavelength of the Ca II K line in Angstroms.
	:type CaIIK: float
	:param width: Width of the Ca II lines in Angstroms.
	:type width: float
	:param Rsection_low: Lower limit of the wavelength range for the red continuum :cite:p:`BoroSaikia2018` in Angstroms.
	:type Rsection_low: float
	:param Rsection_high: Upper limit of the wavelength range for the red continuum :cite:p:`BoroSaikia2018` in Angstroms.
	:type Rsection_high: float
	:param Vsection_low: Lower limit of the wavelength range for the blue continuum :cite:p:`BoroSaikia2018` in Angstroms.
	:type Vsection_low: float
	:param Vsection_high: Upper limit of the wavelength range for the blue continuum :cite:p:`BoroSaikia2018` in Angstroms.
	:type Vsection_high: float
	:param outdir: Output directory. If not provided, the current directory is used.
	:type outdir: str
	:param return_data: If ``True``, the data structure is returned.
	:type return_data: bool 
	:param verbose: If ``True``, the processing is verbose.
	:type verbose: bool
	

	:return: Dictionary with the results of the processing.
	:rtype: dict
	
	Example
	-------
	>>> import FIESpipe as fp
	>>> filename = 'FIEf100105_step010_wave.fits'
	>>> data = fp.standard(filename)

	Plots can be made with the :py:func:`FIESpipe.evince` functions,
	for instance, the CCFs:
	
	>>> fp.plotCCFs(data)


	'''
	## Storage
	data = {}


	if ins == 'FIES':
		R=67000
		s=2.1
		wave, flux, err, h = extractFIES(filename)
	
		# ## Get the barycentric correction AND the BJD in TDB
		# ## temporarily set rvsys to 0.0
		# bjd, bvc = getBarycorrs(filename,rvsys=0.0)# Fix this, should not depend on instrument

	## If no orders are specified, use all of them
	if len(orders) == 0:
		orders = [ii + 1 for ii in range(wave.shape[0])]
	## ...unless, of course, some orders are to be excluded
	use_orders = [order for order in orders if order not in exclude]
	data['orders'] = use_orders
	
	## Create a dictionary for each order
	for order in use_orders:
		data['order_{}'.format(order)] = {}

	## Template
	## Hardcoded for now
	tpath = path + '/../../data/temp/kurucz/'
	temp = '6250_40_p00p00.ms.fits'
	twl, tfl = readKurucz(tpath+temp)
	
	## If systemic velocity is not provided
	## do a preliminary RV measurement
	if rvsys is None:
		## to shift template
		bvc = 0.0
		tempwl = twl*(1.0 - bvc*1e3/const.c.value)
		#tempwl = twl*(1.0 + bvc*1e3/const.c.value)
		rvsys_est = []
		for ii in range(40,50):
			key = 'order_{}'.format(ii)
			rwl, rfl, rfle = wave[ii-1], flux[ii-1], err[ii-1]#fp.getFIES(dat,order=ii)
			rel_err = rfle/rfl
			wl, nfl = normalize(rwl,rfl)
			nfle = rel_err*nfl
			lam, resamp_flip_fl, resamp_flip_tfl, resamp_flip_fle = resample(wl,nfl,nfle,tempwl,tfl,dv=1)
			rvs, ccf, errs = getCCF(resamp_flip_fl,resamp_flip_tfl,resamp_flip_fle,rvr=601,dv=1)
			rvsys_est.append(rvs[np.argmax(ccf)])
		
		rvsys = np.median(rvsys_est)
		## Get the barycentric correction AND the BJD in TDB
		#bjd, bvc_temp = getBarycorrs(filename,rvsys)# Fix this, should not depend on instrument
		bjd, _ = getBarycorrs(filename,rvsys)# Fix this, should not depend on instrument
		data['BJD_TDB'] = bjd
		#data['BVC'] = bvc_temp
	
	
	## Shift the template to center the grid on 0 km/s
	bvc_temp = 0.
	#stwl = twl/(1.0 + (rvsys-bvc_temp)*1e3/const.c.value)
	#stwl = twl/(1.0 + (bvc-rvsys)*1e3/const.c.value)
	stwl = twl*(1.0 + (rvsys-bvc)*1e3/const.c.value)

	## Collect data for S-index
	data['CaIIH'] = {
		'orders' : [],
	}
	data['CaIIK'] = {
		'orders' : [],
	}
	data['Rcontinuum'] = {
		'orders' : [],
	}
	data['Vcontinuum'] = {
		'orders' : [],
	}

	## Arrays for CCFs 
	ccfs, ccferrs, arvr, ccfs_ov, ccferrs_ov, arvr_ov = grids(rvr,R,s)
	dv = velRes(R,s)

	## Chi2 array for RVs
	drvs = np.arange(-chi2rvr,chi2rvr,dv)

	## Loop over orders
	for ii in use_orders:
		key = 'order_{}'.format(ii)
		## Extract the data
		rwl, rfl, rfle = wave[ii-1], flux[ii-1], err[ii-1]
		rel_err = rfle/rfl
		## Normalize the data
		wl, nfl = normalize(rwl,rfl,poly=npoly)
		nfle = rel_err*nfl
		## Mitigate cosmic rays
		if crmit:
			wlo, nflo, nfleo, outliers = crm(wl,nfl,nfle)
			data[key]['outliers'] = np.array([wl[outliers],nfl[outliers],nfle[outliers]])
			wl, nfl, nfle = wlo, nflo, nfleo
		data[key]['spectrum'] = np.array([wl.copy(),nfl.copy(),nfle.copy()])

		## Chi2 minimization for RVs
		chi2s = chi2RV(drvs,wl,nfl,nfle,stwl,tfl)
		data[key]['Chi2s'] = np.array([drvs,chi2s])
		## By fitting a gaussian
		if gauss:
			try:
				pars, cov = fitGauss(drvs,chi2s)
				perr = np.sqrt(np.diag(cov))
				## back to systemic velocity
				rv = pars[1]
				erv = perr[1]
				if verbose: print('Gaussian fit order {}: RV = {:.3f} +/- {:.3f} km/s'.format(ii,rv,erv))
			except RuntimeError:
				if verbose: print('Gaussian to get RV fit failed for order {}. \n Probably low S/N.'.format(ii))
				rv = np.nan
				erv = np.nan
		else:
			## ...or by fitting a 2nd order polynomial 
			## to the chi2s to find the minimum

			## Find dip
			peak = np.argmin(chi2s)

			## Don't use the entire grid, only points close to the minimum
			## For bad orders, there might be several valleys
			## in most cases the "real" RV should be close to the middle of the grid
			if (peak >= (len(chi2s) - pidx)) or (peak <= pidx):
				peak = len(chi2s)//2
			keep = (drvs < drvs[peak+pidx]) & (drvs > drvs[peak-pidx])

			pars = np.polyfit(drvs[keep],chi2s[keep],2)

			rv = -pars[1]/(2*pars[0])
			
			## The curvature is taking as the error.
			erv = np.sqrt(2/pars[0])
			if verbose: print('Poly 2 fit: RV = {:.3f} +/- {:.3f} km/s'.format(rv,erv))

		data[key]['RV'] = rv+rvsys
		data[key]['eRV'] = erv

		## Might be a good idea to split the workflow here.
		## Do a second loop where the weighted average RVs
		## are used to shift the template to the systemic velocity

		## Resample the observations and template
		lam, resamp_fl, resamp_tfl, resamp_fle = resample(wl,nfl,nfle,stwl,tfl,dv=dv)
		rvs, ccf, errs = getCCF(resamp_fl,resamp_tfl,resamp_fle,rvr=arvr,dv=dv)
		
		## Shift RVs to the systemic velocity
		rvs += rvsys#+bvc
		## Co-add CCFs
		ccfs += ccf
		## Co-add CCF errors, error propagation
		ccferrs += errs**2
		## Store for each order
		arr = np.array([rvs,ccf,errs])
		data[key]['CCF'] = arr

		## Fit CCF
		## if the SNR is low, we can't get a good fit
		try:
			r, er2, fw, efw, c, ec = getRV(rvs,ccf,flipped=1)
		except RuntimeError:
			print('Gaussian fit for activity indicators for order {} failed.\n Might have low SNR.'.format(ii))
			r, er2, fw, efw, c, ec = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

		data[key]['RV_ccf'] = r
		data[key]['eRV_ccf'] = er2
		data[key]['contrast'] = c
		data[key]['econtrast'] = ec
		data[key]['FWHM'] = fw
		data[key]['eFWHM'] = efw

		## Shift raw wavelength to rest frame
		swl = rwl*(1.0 - data[key]['RV']*1e3/const.c.value)

		## Ca II H and K lines
		## Grab if the order contains the lines
		if len(np.where( (CaIIH > min(swl)) & (CaIIH < max(swl)) )[0]):
			data['CaIIH']['orders'].append(ii)
			arr = np.array([swl,rfl,rfle])
			data['CaIIH']['order_{}'.format(ii)] = arr
		if len(np.where( (CaIIK > min(swl)) & (CaIIK < max(swl)) )[0]):
			data['CaIIK']['orders'].append(ii)
			arr = np.array([swl,rfl,rfle])
			data['CaIIK']['order_{}'.format(ii)] = arr

		## Get the continuum
		## Hardcoded for now -- fix! Maybe just keep?
		full = 1
		critR = False
		## Span full range
		## OR just that 100 points are within the range
		if full:
			if any(swl < Rsection_low) and any(swl > Rsection_high):
				critR = True
		else:	
			Ridxs = np.where((swl > Rsection_low) & (swl < Rsection_high))[0]
			critR = len(Ridxs) > 100
		if critR:
			data['Rcontinuum']['orders'].append(ii)
			arr = np.array([swl,rfl,rfle])
			data['Rcontinuum']['order_{}'.format(ii)] = arr
		critV = False
		if full:
			if any(swl < Vsection_low) and any(swl > Vsection_high):
				critV = True
		else:
			Vidxs = np.where((swl > Vsection_low) & (swl < Vsection_high))[0]
			critV = len(Vidxs) > 100
		if critV:
			data['Vcontinuum']['orders'].append(ii)
			arr = np.array([swl,rfl,rfle])
			data['Vcontinuum']['order_{}'.format(ii)] = arr

		## Same approach for oversampled CCFs
		lam, resamp_fl, resamp_tfl, resamp_fle = resample(wl,nfl,nfle,stwl,tfl,dv=0.25)
		rvs_ov, ccf_ov, errs_ov = getCCF(resamp_fl,resamp_tfl,resamp_fle,rvr=arvr_ov,dv=0.25)

		rvs_ov += rvsys
		ccfs_ov += ccf_ov
		ccferrs_ov += errs_ov**2

		arr = np.array([rvs_ov,ccf_ov,errs_ov])
		data[key]['CCF_ov'] = arr

		try:
			r, er2, fw, efw, c, ec = getRV(rvs_ov,ccf_ov,flipped=1)
		except RuntimeError:
			print('Gaussian fit for activity indicators for order {} failed.\n Might have low SNR.'.format(ii))
			r, er2, fw, efw, c, ec = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
		data[key]['RV_ccf_ov'] = r
		data[key]['eRV_ccf_ov'] = er2
		data[key]['contrast_ov'] = c
		data[key]['econtrast_ov'] = ec
		data[key]['FWHM_ov'] = fw
		data[key]['eFWHM_ov'] = efw
	
	## Summed CCFs
	ccferrs = np.sqrt(ccferrs)
	arr = np.array([rvs,ccfs,ccferrs])
	data['CCFsum'] = arr

	## Fit summed CCF
	r, er2, fw, efw, c, ec = getRV(rvs,ccfs,ccferr=ccferrs,flipped=1)
	pars, cov = fitGauss(rvs,ccfs,flipped=1)

	data['FWHMsum'] = fw
	data['eFWHMsum'] = efw
	data['Contrastsum'] = c
	data['eContrastsum'] = ec
	data['RVsum'] = r
	data['eRVsum'] = er2
	data['fitGausssum'] = [pars, cov]

	## Summed oversampled CCFs
	ccferrs_ov = np.sqrt(ccferrs_ov)
	arr_ov = np.array([rvs_ov,ccfs_ov,ccferrs_ov])
	data['CCFsum_ov'] = arr_ov

	## Get weighted mean RVs 
	rv_all, erv_all = np.array([]), np.array([])
	for order in data['orders']:
		key = 'order_{}'.format(order)
		rv = data[key]['RV']
		erv = data[key]['eRV']
		if np.isfinite(rv) & np.isfinite(erv):
			rv_all = np.append(rv_all,rv)
			erv_all = np.append(erv_all,erv)
	
	## Outlier rejection?
	if out:
		wavg_rv, wavg_err, mu, std  = weightedMean(rv_all,erv_all,out=out,sigma=sigma)
	else:
		wavg_rv, wavg_err = weightedMean(rv_all,erv_all,out=out)
		mu, std = np.nan, np.nan
	_, bvc = getBarycorrs(filename,wavg_rv)
	data['BVC'] = bvc
	data['order_all'] = {
		'RV': wavg_rv+bvc,
		'eRV': wavg_err,
		'mu_out': mu,
		'std_out': std,
	}

	## REMOVE
	# y = ccfs# - np.min(ccfs)
	# x = rvs
	# err, der, xerr = getError(rvs, y, ccferrs)
	# bis, biserr = getBIS(x, y, xerr)
	## BIS estimate from oversampled CCF
	err, der, xerr = getError(data['CCFsum_ov'][0,:], data['CCFsum_ov'][1,:], data['CCFsum_ov'][2,:])
	data['CCFsum_ov_xerr'] = xerr
	y = data['CCFsum_ov'][1,:]
	x = data['CCFsum_ov'][0,:]
	bis, bx, by, biserr = getBIS(x, y, xerr)

	data['BISx'] = bx
	data['BISy'] = by
	data['BIS'] = bis
	data['eBIS'] = biserr

	## Get S-index 
	Korders = data['CaIIK']['orders']
	Ks = []
	for order in Korders:
		key = 'order_{}'.format(order)
		arr = data['CaIIK'][key]
		wl = arr[0]
		fl = arr[1]
		weights = triangle(wl,CaIIK,width*0.5,width*0.5)
		core = weights > 0.0
		K = np.median(fl[core]*weights[core])
		Ks.append(K)

	Horders = data['CaIIH']['orders']
	Hs = []
	for order in Horders:
		key = 'order_{}'.format(order)
		arr = data['CaIIH'][key]
		wl = arr[0]
		fl = arr[1]
		weights = triangle(wl,CaIIH,width*0.5,width*0.5)
		core = weights > 0.0
		H = np.median(fl[core]*weights[core])
		Hs.append(H)

	Vorders = data['Vcontinuum']['orders']
	Vconts = []
	for order in Vorders:
		key = 'order_{}'.format(order)
		arr = data['Vcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		Vcont = np.median(fl[(wl > Vsection_low) & (wl < Vsection_high)])
		Vconts.append(Vcont)

	Rorders = data['Rcontinuum']['orders']
	Rconts = []
	for order in Rorders:
		key = 'order_{}'.format(order)
		arr = data['Rcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		Rcont = np.median(fl[(wl > Rsection_low) & (wl < Rsection_high)])
		Rconts.append(Rcont)

	S = (np.mean(Ks) + np.mean(Hs))/(np.mean(Vconts) + np.mean(Rconts))
	data['S_uncalibrated'] = S
	
	## Create output path
	fn = filename.split('/')[-1]
	name = fn.split('.fits')[0]+'_end_products.pkl'
	if outdir:
		if not outdir.endswith('/'):
			outdir = outdir+'/'
		name = outdir+name
	else:
		name = filename.rsplit('/',1)[0]+'/'+name
	## Dump data to pickle
	with open(name,'wb') as f:
		pickle.dump(data,f)
	
	if return_data:
		return data

# =============================================================================
# SERVAL style template matching
# =============================================================================

def servalish(filenames,
	run_standard=True,
	knot_freq = 1,
	sigma = 5,
	window_length = 51, # savgol window
	sk = 3, # savgol degree
	ck = 3, # degree of spline, cubic
	outer = 3, # iteration, outlier rejection creating master template
	ignore=False,
	norders=range(20,70), # orders to use to extract RVs
	out=True,
	sigma_out=5,
	**std_kwargs,
	):
	'''SERVAL style template matching 

	Radial velocity extraction through template matching in the style of https://github.com/mzechmeister/serval :cite:p:`Zechmeister2018`.

	Cubic B-spline using :py:func:`scipy.interpolate.splrep` and :py:func:`scipy.interpolate.Bspline`.

	When creating the knots for the spline, the attempt is to get as close to the ones returned by the :py:mod:`SERVAL` splines (https://github.com/mzechmeister/serval/blob/master/src/cspline.py#L1),
	this is achieved with inspiration from this answer:
	https://stackoverflow.com/questions/49191877/how-to-set-manual-knots-in-scipy-interpolate-splprep-in-case-of-periodic-boundar

	:param filenames: List of filenames to use for template matching. Should be larger than 3.
	:type filenames: list
	:param run_standard: Run the standard reduction in :py:func:`FIESpipe.workflows.standard` before template matching. Default is True, otherwise give a list of data products already produced by :py:func:`FIESpipe.workflows.standard`.
	:type run_standard: bool
	:param knot_freq: Frequency of knots in the spline. Default is 1, which means that the number of knots are equal to the wavelength spacing in a given order.
	:type knot_freq: float
	:param sigma: Sigma clipping for outlier rejection when creating the master template. Default is 5.
	:type sigma: float
	:param window_length: Window length for Savitzky-Golay filter. Default is 51.
	:type window_length: int
	:param sk: Degree of Savitzky-Golay filter. Default is 3.
	:type sk: int
	:param ck: Degree of cubic B-spline. Default is 3.
	:type ck: int
	:param outer: Number of iterations for outlier rejection when creating the master template. Default is 3.
	:type outer: int
	:param ignore: Ignore the check that the number of files is larger than 3. Default is False.
	:type ignore: bool
	:param norders: Orders to use for RV extraction. Default is range(20,70).
	:type norders: list
	:param out: Use outlier rejection when extracting RVs. Default is True.
	:type out: bool
	:param sigma_out: Sigma clipping for outlier rejection when extracting RVs. Default is 5.
	:type sigma_out: float
	:param std_kwargs: Keyword arguments for :py:func:`FIESpipe.workflows.standard`.
	:type std_kwargs: dict

	:returns: Dictionaries with the data products from the template matching, and the final data products.
	:rtype: dict, dict
	
	.. note::
		- Using cubic B-spline from :py:mod:`scipy` instead of the ones from https://github.com/mzechmeister/serval/blob/master/src/cspline.py#L1.
		- The weights for creating the spline are just taken as :math:`1/\sigma_f`
			- maybe more weighting could improve the results, like deweighting of telluric lines?

			
	Example
	-------
	>>> import FIESpipe as fp
	>>> import glob
	>>> filenames = glob.glob('FI*_wave.fits')
	>>> sdata = fp.servalish(filenames)

	Plots can be made with the :py:func:`FIESpipe.evince` functions,
	for instance, the coadded template:
	
	>>> fp.plotCoaddTemplate(sdata)
			
				
	'''
	if not ignore:
		Nfiles = len(filenames)
		assert Nfiles > 3, print('Only {} files given. To create a proper master template more files are needed.'.format(Nfiles))


	dats = []
	for filename in filenames:
		## Run the standard pipe first?
		if run_standard:
			data = standard(filename,return_data=True,**std_kwargs)
		## or have parameters already been extracted?
		else:
			data = readDataProduct(filename)
		dats.append(data)
	orders = dats[0]['orders']

	## Dict for template
	servalData = {}
	servalData['orders'] = orders
	for order in orders:
		key = 'order_{}'.format(order)
		servalData[key] = {}

	## Create a template for each order
	for order in orders:
		key = 'order_{}'.format(order)
		swl = np.array([])
		fl = np.array([])
		fle = np.array([])
		points = np.array([],dtype=int)
		## REMOVE
		#dWs = np.array([])

		## Subdict for each file/epoch
		servalData[key]['files'] = {}
		for ii, data in enumerate(dats):
			rv = data['order_all']['RV']
			erv = data['order_all']['eRV']
			bvc = data['BVC']
			wl, nfl, nfle = data[key]['spectrum']
			
			points = np.append(points,len(wl))
			## Shift the wavelength to the rest frame
			dv = rv - bvc
			nwl = wl*(1.0 - dv*1e3/const.c.value)
			
			servalData[key]['files'][filename[ii]] = [nwl,nfl,nfle]
			
			swl = np.append(swl,nwl)
			fl = np.append(fl,nfl)
			fle = np.append(fle,nfle)
			## REMOVE
			#dWs = np.append(dWs,max(nwl)-min(nwl))

		## REMOVE
		## Typical wavelength span
		#dW = np.median(dWs)
		
		## How many points are in the wavelength
		points = np.median(points)
		## Sort the arrays by wavelength
		ss = np.argsort(swl)
		swl, fl, fle = swl[ss], fl[ss], fle[ss]

		## REMOVE
		## Total span of the wavelength
		#dWtotal = max(swl)-min(swl)

		## Outlier rejection
		## Several iterations?
		## This is a bit redundant (at least at this point)
		## No need in storing the number of iterations for each order
		## since it is the same for all orders
		servalData[key]['rejection'] = {
			'iterations':outer,
		}
		for ii in range(outer):
			## Savitzky-Golay filter for outlier rejection
			yhat = scsig.savgol_filter(fl, window_length, sk)
			res = fl-yhat

			## Rejection
			mu, sig = scs.norm.fit(res)
			sig *= sigma	
			keep = (res < (mu + sig)) & (res > (mu - sig))
			## Save the filter parameters (for plotting)		
			servalData[key]['rejection']['iteration_{}'.format(ii)] = {
				'filter' : [yhat, res, swl, mu, sig],
				'outliers' : [swl[~keep],fl[~keep],fle[~keep]],
			}

			## Trim the arrays
			swl, fl, fle = swl[keep], fl[keep], fle[keep]
		servalData[key]['spectrum'] = [swl,fl,fle]

		## Weights, maybe more can be done here... Telluric weights?
		w = 1.0/fle

		## Number of knots, 
		## just use the median number of points in wavelength
		Nknots = int(knot_freq*points)

		## The following is a bit hacky
		## The idea is to try to get as close to the knots as returned by the SERVAL spline
		## The knots have to be within the range of the wavelength array
		## ... this "hackiness" includes the knots[4:-4]
		knots = np.linspace(swl[np.argmin(swl)],swl[np.argmax(swl)],Nknots)
		## Get the coefficients for the spline
		t, c, k = sci.splrep(swl, fl, w=w, k=ck, t=knots[4:-4])
		## ...and create the spline
		spline = sci.BSpline(t, c, k, extrapolate=False)
		
		## Save the template wavelength and the spline
		## The spline is a callable function
		## This is to be able to evaluate the spline at any wavelength
		servalData[key]['template'] = [knots,spline]

	## And now loop over all epochs
	## to extract the RVs 
	chi2rvr = 20 
	dv = 0.1
	pidx = 3
	endData = {}
	## Velocity grid for the RV extraction
	all_times = np.array([])
	all_rvs = np.array([])
	all_ervs = np.array([])
	#return servalData, endData

	drvs = np.arange(-chi2rvr,chi2rvr,dv)
	for ii, data in enumerate(dats):
		#orders = data['orders']
		time = data['BJD_TDB']
		bvc = data['BVC']
		mrv = data['order_all']['RV']
		emrv = data['order_all']['eRV']
		rvs, ervs = np.array([]), np.array([])
		## Loop over all orders
		for ii, order in enumerate(norders):
			key = 'order_{}'.format(order)
			wl, nfl, nfle = data[key]['spectrum'][0], data[key]['spectrum'][1], data[key]['spectrum'][2]
			twl, spline = servalData[key]['template'][0], servalData[key]['template'][1]
			chi2s = np.array([])
			## Chi2 approach to find RV
			## Similar to the one used in SERVAL
			## and above in standard workflow
			for drv in drvs:
				## Subtract bvc and add measured RV
				## to center on 0.0 km/s
				ds = drv - bvc + mrv 
				wl_shift = wl/(1.0 + ds*1e3/const.c.value)

				mask = (wl_shift > min(twl)) & (wl_shift < max(twl))
				ys = spline(wl_shift[mask])
				chi2 = np.sum((ys - nfl[mask])**2/nfle[mask]**2)
				chi2s = np.append(chi2s,chi2)
			## Find dip
			peak = np.argmin(chi2s)

			## Don't use the entire grid, only points close to the minimum
			## For bad CCFs, there might be several valleys
			## in most cases the "real" RV should be close to the middle of the grid
			if (peak >= (len(chi2s) - pidx)) or (peak <= pidx):
				peak = len(chi2s)//2
			keep = (drvs < drvs[peak+pidx]) & (drvs > drvs[peak-pidx])
			
			pars = np.polyfit(drvs[keep],chi2s[keep],2)
			## Add the measured RV to get proper value
			rv = -pars[1]/(2*pars[0]) + mrv
			## The curvature is taking as the error.		
			erv = np.sqrt(2/pars[0])
			## Only append if both values are finite
			if np.isfinite(rv) & np.isfinite(erv):
				rvs = np.append(rvs,rv)
				ervs = np.append(ervs,erv)
		
		## Calculate the weighted average RV
		wavg_rv, wavg_err, mu, std  = weightedMean(rvs,ervs,out=out,sigma=sigma_out)
		print('Weighted average RV: {:.3f} +/- {:.3f} km/s'.format(wavg_rv,wavg_err))
		all_times = np.append(all_times,time)
		all_rvs = np.append(all_rvs,wavg_rv)
		all_ervs = np.append(all_ervs,wavg_err)

	endData['times'] = all_times
	endData['rvs'] = all_rvs
	endData['ervs'] = all_ervs

	return servalData, endData
