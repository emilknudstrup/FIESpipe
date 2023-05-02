#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of modules used to derive quantities from the FIES spectra.

"""

import numpy as np
from astropy import constants as const
from scipy.optimize import curve_fit
import scipy.stats as scs
import scipy.interpolate as sci
import scipy.signal as scsig
import lmfit
from .extract import getBarycorrs

# =============================================================================
# Preparation/normalization
# =============================================================================

def normalize(wl,fl,bl=np.array([]),poly=1,gauss=True,lower=0.5,upper=1.5):
	'''Normalize spectrum.

	Nomalization of observed spectrum.


	.. note::
		This is an obvious place to start with improving the results/precision of the RVs:
			-Normalization could probably be improved
			-Outlier rejection more sophisticated
			-Blaze function correction, available in other data products?

	:param wl: Observed wavelength 
	:type wl: array 
	:param fl: Observed raw flux
	:type fl: array
	:param bl: Blaze function. Default ``numpy.array([])``. If empty, no correction for the blaze function.
	:type bl: array, optional
	:param poly: Degree of polynomial fit to normalize. Default 1. Set to ``None`` for no polynomial fit.
	:type poly: int, optional
	:param gauss: Only fit the polynomial to flux within, (mu + upper*sigma) > fl > (mu - lower*sigma). Default ``True``.
	:type gauss: bool, optional
	:param lower: Lower sigma limit to include in poly fit. Default 0.5.
	:type lower: float, optional
	:param upper: Upper sigma limit to include in poly fit. Default 1.5.
	:type upper: float, optional

	:return: observed wavelength, observed normalized flux
	:rtype: array, array
	'''

	if len(bl) > 0:
		fl = fl/bl # normalize w/ blaze function
	
	keep = np.isfinite(fl)
	wl, fl = wl[keep], fl[keep] # exclude nans
	
	if poly is not None:
		if gauss:
			mu, sig = scs.norm.fit(fl)
			mid  = (fl < (mu + upper*sig)) & (fl > (mu - lower*sig))
			pars = np.polyfit(wl[mid],fl[mid],poly)
		else:
			pars = np.polyfit(wl,fl,poly)
		fit = np.poly1d(pars)
		nfl = fl/fit(wl)
	else:
		nfl = fl/np.median(fl)

	return wl, nfl

def crm(wl,nfl,err=np.array([]),iters=1,q=[99.0,99.9,99.99]):
	'''Cosmic ray mitigation.
	
	Excludes flux over qth percentile.

	.. note::
		There might be a more sophisticated way of doing this.
		Also, need to make sure it's working properly without flux errors.


	:param wl: Observed wavelength.
	:type wl: array 
	:param nfl: Observed normalized flux.
	:type: array
	:param err: Observed flux error. Default ``numpy.array([])``. Leave empty, if no error available.
	:type err: array, optional
	:param iters: Iterations of removing upper q[iter] percentile. Default 1.
	:type iters: int, optional
	:param q: Percentiles. Default ``[99.0,99.9,99.99]``.
	:type q: list, optional
	
	:return: observed wavelength, observed normalized flux
	:rtype: array, array
	'''
	assert iters <= len(q), 'Error: More iterations than specified percentiles.'

	ret = True
	if not len(err):
		err = np.ones(len(wl))
		ret = False

	idxs = np.array([],dtype=int)
	for ii in range(iters):
		cut = np.percentile(nfl,q[ii]) # find upper percentile
		cosmic = nfl > cut # cosmic ray mitigation
		wl, nfl = wl[~cosmic], nfl[~cosmic]
		err = err[~cosmic]
		idxs = np.append(idxs,np.where(cosmic)[0])
	if ret:
		return wl, nfl, err, idxs
	return wl, nfl, idxs

def resample(wl,nfl,fle,twl,tfl,dv=1.0,edge=0.0):
	'''Resample spectrum.

	Resample wavelength and interpolate flux and template flux.
	Flips flux, i.e., 1-flux.
	
	:param wl: Observed wavelength.
	:type wl: array
	:param nfl: Observed normalized flux.
	:type nfl: array
	:param twl: Template wavelength.
	:type twl: array
	:param tfl: Template flux.
	:type tfl: array
	:param dv: RV steps in km/s. Default 1.0.
	:type dv: float, optional
	:param edge: Skip edge of detector - low S/N - in Angstrom. Default 0.0.
	:type edge: float, optional
	
	:return: resampled wavelength, resampled and flipped flux, resampled and flipped template flux
	:rtype: array, array, array
	'''

	wl1, wl2 = min(wl) + edge, max(wl) - edge
	nn = np.log(wl2/wl1)/np.log(np.float64(1.0) + dv/(const.c.value/1e3))
	lam = wl1*(np.float64(1.0)  + dv/(const.c.value/1e3))**np.arange(nn,dtype='float64')
	if len(lam)%2 != 0: lam = lam[:-1] # uneven number of elements

	keep = (twl >= lam[0]) & (twl <= lam[-1]) # only use resampled wl interval
	twl, tfl = twl[keep], tfl[keep]
	
	r_fle = np.interp(lam,wl,fle)
	r_fl = np.interp(lam,wl,nfl)
	r_tl = np.interp(lam,twl,tfl)
	return lam, r_fl, r_tl, r_fle

# def resample(wl,nfl,twl,tfl,dv=1.0,edge=0.0):
# 	'''Resample spectrum.

# 	Resample wavelength and interpolate flux and template flux.
# 	Flips flux, i.e., 1-flux.
	
# 	:param wl: Observed wavelength.
# 	:type wl: array
# 	:param nfl: Observed normalized flux.
# 	:type nfl: array
# 	:param twl: Template wavelength.
# 	:type twl: array
# 	:param tfl: Template flux.
# 	:type tfl: array
# 	:param dv: RV steps in km/s. Default 1.0.
# 	:type dv: float, optional
# 	:param edge: Skip edge of detector - low S/N - in Angstrom. Default 0.0.
# 	:type edge: float, optional
	
# 	:return: resampled wavelength, resampled and flipped flux, resampled and flipped template flux
# 	:rtype: array, array, array
# 	'''

# 	wl1, wl2 = min(wl) + edge, max(wl) - edge
# 	nn = np.log(wl2/wl1)/np.log(np.float64(1.0) + dv/(const.c.value/1e3))
# 	lam = wl1*(np.float64(1.0)  + dv/(const.c.value/1e3))**np.arange(nn,dtype='float64')
# 	if len(lam)%2 != 0: lam = lam[:-1] # uneven number of elements

# 	keep = (twl >= lam[0]) & (twl <= lam[-1]) # only use resampled wl interval
# 	twl, tfl_order = twl[keep], tfl[keep]
	
# 	flip_fl, flip_tfl = 1-nfl, 1-tfl_order
# 	rf_fl = np.interp(lam,wl,flip_fl)
# 	rf_tl = np.interp(lam,twl,flip_tfl)
# 	return lam, rf_fl, rf_tl


# =============================================================================
# Cross-correlation
# =============================================================================

def getCCF(fl,tfl,fle,dv=1.0,rvr=401,ccf_mode='full'):
	'''Cross-correlation function.

	Perform the cross correlation 

	.. math:: 
		\sigma^2 (v) = - \\left [ N  \\frac{C^{\prime \prime}(\hat{s})}{C(\hat{s})} \\frac{C^2(\hat{s})}{1 - C^2(\hat{s})} \\right ]^{-1} \, ,

	where :math:`C(\hat{s})` is the cross-correlation function, and :math:`C^{\prime \prime}(\hat{s})` the second derivative. :math:`N` is the number of bins.
	
	Here using :py:func:`numpy.correlate`.
	The arrays are trimmed to only include points over the RV range.
	
	:param fl: Flipped and resampled flux.
	:type  fl: array
	:param tfl: Flipped and resampled template flux.
	:type tfl: array
	:param fle: Resampled flux errors.
	:type fle: array
	:param dv: RV steps in km/s. Default 1.0.
	:type dv: float, optional
	:param rvr: Range for velocity grid in km/s. Default 401.
	:type rvr: int, optional
	:param ccf_mode: Mode for cross-correlation. Default 'full'.
	:type ccf_mode: str, optional
	
	:return: velocity grid, CCF, CCF errors
	:rtype: array, array, array
	'''

	ccf = np.correlate(fl,tfl,mode=ccf_mode)
	ccferr = np.correlate(fle,tfl,mode=ccf_mode)
	## normalize ccf
	ccf = ccf/(np.std(fl) * np.std(tfl) * len(tfl))
	## normalize errors from ccf
	ccferr = ccferr/(np.std(fl) * np.std(tfl) * len(tfl))
	
	## create velocity grid
	rvs = np.arange(len(ccf)) - len(ccf)//2
	rvs = np.asarray(rvs,dtype=float)
	## midpoint
	mid = (len(ccf) - 1) // 2
	lo = mid - rvr//2
	hi = mid + (rvr//2 + 1)

	## trim arrays
	rvs, ccf = rvs[lo:hi], ccf[lo:hi]
	ccferr = ccferr[lo:hi]
	
	## scale velocity grid
	rvs = rvs*dv

	return rvs, ccf, ccferr

# def sumCCF(ccf):
# 	'''Sum CCF.
	
# 	Sum CCFs from different orders.

# 	.. note::
# 		Add option to weight CCFs by S/N of orders
# 		or some other weighting scheme.
# 		BUT the S/N of the CCFs already take the S/N 
# 		of the flux in a given order into account,
# 		so probably not necessary.

# 	:param ccf: CCF.
# 	:type ccf: array

# 	:return: Summed CCF.
# 	:rtype: array
# 	'''
# 	ccfsum = np.zeros(ccf.shape[1])
# 	for ii in range(ccf.shape[0]):
# 		ccfsum += ccf[ii]
	

# 	return ccfsum

def getxError(rv, ccf, ccferr):
	'''RV error from CCF profile.

	Error estimated from Eqs. (8)-(10) in :cite:t:`Lafarga2020`:
	
	.. math::
		\sigma_\\mathrm{CCF}^2 (v) = \sum_o \sigma^2(v)_{\\mathrm{CCF},o} (v) \left (\\frac{\\mathrm{dCCF}(v)}{\\mathrm{d} v} \\right)^{-1} 

	.. math::
		\sigma (v) = \sigma (v)^2_\\mathrm{CCF} (v) \left (\\mathrm{dCCF}(v)/\\mathrm{d} v \\right)^{-1}

	.. math::
		\sigma_\\mathrm{RV} = \left ( \sqrt{\Sigma_v 1/\sigma^2(v)} \\right)^{-1}
		
	where :math:`o` is a given order.

	Implemented similar to :py:func:`raccoon.ccf.computerverr`:
	https://github.com/mlafarga/raccoon/blob/master/raccoon/ccf.py#L89

	:param rv: RV grid.
	:type rv: array
	:param ccf: CCF.
	:type ccf: array
	:param ccferr: CCF errors.
	:type ccferr: array

	:return: RV error, CCF derivative, RV error grid
	:rtype: float, array, array

	'''

	## Gradient
	dx = np.mean(np.diff(rv))
	der = np.abs(np.gradient(ccf,dx))

	## RV error grid
	rverr = np.power(ccferr,2) / der

	## RV error total
	rverrsum = np.sum(np.power(rverr,-2))
	rverrt = 1./np.sqrt(rverrsum)
	return rverrt, der, rverr

# =============================================================================
# Chi-squared minimization
# =============================================================================

def chi2(yo,ym,yerr):
	'''Chi-squared.

	Chi-squared function:
	
	.. math::
		\\chi^2 = \\sum_{i=1}^{N} \\frac{(y_i - \\hat{y}_i)^2}{\\sigma_i^2}

	:param yo: Observed data.
	:type yo: array
	:param ym: Model data.
	:type ym: array
	:param yerr: Data errors.
	:type yerr: array

	:return: :math:`\chi^2`.
	:rtype: float

	'''
	return np.sum(np.power(yo-ym,2)*np.power(yerr,-2))


def chi2RV(drvs,wl,nfl,fle,twl,tfl):
	'''RVs through Chi-squared.

	Compute :math:`\\chi^2` for each RV using :py:func:`chi2` for each RV in `drvs`.

	:param drvs: RV grid in km/s.
	:type drvs: array
	:param wl: Wavelength in Angstrom.
	:type wl: array
	:param nfl: Normalized flux.
	:type nfl: array
	:param fle: Flux errors.
	:type fle: array
	:param twl: Template wavelength in Angstrom.
	:type twl: array
	:param tfl: Template flux.
	:type tfl: array

	:return: :math:`\\chi^2` for each RV.
	:rtype: array

	'''

	chi2s = np.array([])
	for i, rv in enumerate(drvs):
		## shift template
		shift = twl*(1.0 + rv*1e3/const.c.value)
		
		## only use resampled wl interval
		minwl, maxwl = min(wl), max(wl)
		keep = (shift >= minwl) & (shift <= maxwl)
		twl_order, tfl_order = shift[keep], tfl[keep]

		## interpolate template
		fi = sci.interp1d(twl_order,tfl_order,fill_value='extrapolate')
		
		## calculate chi2
		chi2s = np.append(chi2s,chi2(nfl,fi(wl),fle))
	
	return chi2s

# =============================================================================
# RV/activity indicators through Gaussian fitting
# =============================================================================


def Gauss(x, amp, mu, sig, off=0.0):
	'''Gaussian function.
	
	.. math::

		f(x) = A e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}} + B

	:param x: x-values.
	:type x: array
	:param amp: Amplitude, :math:`A`.
	:type amp: float
	:param mu: Mean, :math:`\\mu`.
	:type mu: float
	:param sig: Standard deviation, :math:`\\sigma`.
	:type sig: float
	:param off: y-axis offset of baseline, :math:`B`. Default is 0.0.
	:type off: float

	:return: Gaussian function calculated at x.
	:rtype: array
	'''
	y = amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + off
	return y

def fitGauss(xx,yy,yerr=None,guess=None,flipped=False):
	'''Fit Gaussian to data.

	:param xx: x-values.
	:type xx: array
	:param yy: y-values.
	:type yy: array
	:param yerr: y-value errors. Default is None. 
	:type yerr: array
	:param guess: Guesses for Gaussian parameters. Default is None. If None, guesses are created.
	:type guess: list
	:param flipped: If True, the input is inversed. Default is False.
	:type flipped: bool

	:return: Gaussian parameters.
	:rtype: list
	'''
	
	if guess is None:
		## starting guesses
		## find minimum of yy
		if flipped:
			idx = np.argmax(yy)
		else:
			idx = np.argmin(yy)
		## mu is easy, location of minimum
		mu = xx[idx]

		## offset probably too
		off = np.median(yy)

		## get max value of yy
		a = yy[idx]
		## amplitude must be difference between min and offset
		amp = a - off

		## width is harder
		## find half-maximum
		hm = 0.5*amp + off
		left = np.where(xx < mu)[0]
		right = np.where(xx > mu)[0]
		## When there's very little signal in the CCF
		## it's difficult to find the half-maximum
		## estimate it as somewhere close to the center
		## Probably doesn't matter much
		## since it's going to be a poor fit anyway
		try:
			lidx = np.argmin(np.abs(yy[left]-hm))
			ridx = np.argmin(np.abs(yy[right]-hm))
			## width is difference between left and right half-maximum
			width = xx[right[ridx]] - xx[left[lidx]]
		except ValueError:
			mu = xx[len(yy)//2]
			width = xx[len(yy)//2 + 5] - xx[len(yy)//2 - 5]

		## sigma is width/2.355
		sig = width/(2*np.sqrt(2*np.log(2)))
		guess = [amp,mu,sig,off]
	
	## fit Gaussian
	gau_par, pcov = curve_fit(Gauss,xx,yy,p0=guess,sigma=yerr)

	return gau_par, pcov

def getRV(vel,ccf,ccferr=None,guess=None,flipped=False,return_pars=False):
	'''Extract RVs and activity indicators.
	
	Get radial velocity and activity indicators from CCF by fitting a Gaussian.
	Call to :py:func:`fitGauss` to fit Gaussian to CCF.


	:param vel: Velocity in km/s.
	:type vel: array
	:param ccf: CCF.
	:type ccf: array
	:param ccferr: CCF errors. Default is None.
	:type ccferr: array
	:param guess: Guesses for Gaussian parameters. Default is None. If None, guesses are created.
	:type guess: list
	:param flipped: If True, the input is inversed. Default is False.
	:type flipped: bool

	:return: Radial velocity, radial velocity error, FWHM, FWHM error, contrast, contrast error, continuum, continuum error.
	:rtype: tuple

	.. note::
		Errors might be overestimated when just adopting the formal errors from the fit.

	'''

	gau_par, pcov = fitGauss(vel,ccf,yerr=ccferr,guess=guess,flipped=flipped)
	
	## mu
	rv = gau_par[1]
	
	## fwhm
	fwhm = 2*np.sqrt(2*np.log(2))*gau_par[2]
	
	## contrast is amplitude/offset
	amp = gau_par[0]
	off = gau_par[3]
	cont = 100*amp/off
	if not flipped:
		cont *= -1

	## error estimation from covariance matrix
	perr = np.sqrt(np.diag(pcov))
	erv = perr[1]
	efwhm = perr[2]
	eamp = perr[0]
	eoff = perr[3]
	econt = 100*np.sqrt(np.power(off,-2)*np.power(eamp,2) + np.power(amp*eoff,2)*np.power(off,-4))
	if return_pars:
		return rv, erv, fwhm, efwhm, cont, econt, gau_par, pcov
	return rv, erv, fwhm, efwhm, cont, econt

# =============================================================================
# Bisector
# =============================================================================

def getBIS(x,y,xerr=np.array([]),
	n = 100,
	bybotmin_percent=10.,
	bybotmax_percent=40.,
	bytopmin_percent=60.,
	bytopmax_percent=90.,
	dx = None,
	):
	'''BIS calculation.
	
	Calculate bisector from CCF following the approach in Section 4.6.3 of :cite:t:`Lafarga2020`.

	The BIS is calculated as the difference between the top part in the interval (in percent)
	bytopmin_percent to bytopmax_percent and the bottom part in the interval (in percent)
	bybotmin_percent to bybotmax_percent of the CCF. 

	Implemented identical to :py:func:`raccoon.ccf.computebisector_biserr`:
	https://github.com/mlafarga/raccoon/blob/master/raccoon/ccf.py#L332



		
	:param x: Velocity grid.
	:type x: array
	:param y: CCF.
	:type y: array
	:param xerr: Velocity errors. Default is `numpy.array([])`. If empty, no error is returned.
	:type xerr: array
	:param n: Number of points for bisector. Default is 100.
	:type n: int
	:param bybotmin_percent: Minimum percent for bottom of bisector. Default is 10.
	:type bybotmin_percent: float
	:param bybotmax_percent: Maximum percent for bottom of bisector. Default is 40.
	:type bybotmax_percent: float
	:param bytopmin_percent: Minimum percent for top of bisector. Default is 60.
	:type bytopmin_percent: float
	:param bytopmax_percent: Maximum percent for top of bisector. Default is 90.
	:type bytopmax_percent: float

	:return: bisector, bisector error (if `xerr` is not empty).
	:rtype: float, float

	.. note::
		- Will (currently) only work now if the CCF has a peak (rather than a dip as for an absorption line).
		- Error is a bit large...
		
	'''


	if n < 100:
		print('Sampling for bisector is too low! (n < 100).')

	## Find peak
	## This should be flipped if the CCF has the shape of an absorption line
	peak = np.argmax(y)
	left = np.argmin(y[:peak])
	right = peak + np.argmin(y[peak:])
	if right != len(y): right = right + 1
	y_small = np.max([y[left],y[right-1]])

	## Y span for bisector
	by = np.linspace(y_small,y[peak],n)


	## Interpolate to find x values for bisector
	int_left = sci.interp1d(y[left:peak+1],x[left:peak+1],kind='linear')
	int_right = sci.interp1d(y[peak:right],x[peak:right],kind='linear')
	bx1 = int_left(by)
	bx2 = int_right(by)

	## Bisector
	bx = (bx2 + bx1)/2.

	# Bisector up and down region limits -> absolute value
	by_min = np.min(by)
	by_max = np.max(by)
	dy = by_max - by_min

	bybotmin = by_min + dy*bybotmin_percent/100.
	bybotmax = by_min + dy*bybotmax_percent/100.
	bytopmin = by_min + dy*bytopmin_percent/100.
	bytopmax = by_min + dy*bytopmax_percent/100.

	# Bisector up and down region limits indices
	# Note: Approximate regions, depend on bisector sampling
	ibybotmin = np.argmin(np.abs(by-bybotmin))
	ibybotmax = np.argmin(np.abs(by-bybotmax))
	ibytopmin = np.argmin(np.abs(by-bytopmin))
	ibytopmax = np.argmin(np.abs(by-bytopmax))

	# Compute mean RV in each region
	bxmeantop = np.mean(bx[ibytopmin:ibytopmax+1])
	bxmeanbot = np.mean(bx[ibybotmin:ibybotmax+1])

	# Compute bisector inverse slope BIS
	bis = bxmeantop - bxmeanbot

	if len(xerr):
		## Identical approach to compute bisector error
		## Bisector x error values
		## Interpolate to find x error values for bisector
		int_left_err = sci.interp1d(y[left:peak+1], xerr[left:peak+1], kind='linear')
		int_right_err = sci.interp1d(y[peak:right], xerr[peak:right], kind='linear')

		## Bisector x error values
		bx1err = int_left_err(by)
		bx2err = int_right_err(by)

		## Compute bisector error (error propagation)
		bxerr = np.sqrt(bx1err**2 + bx2err**2)/2.

		## Top and bottom regions x width (use only one side of the line)
		dx_top = np.abs(int_left(bytopmax) - int_left(bytopmin))
		dx_bot = np.abs(int_left(bybotmax) - int_left(bybotmin))

		## Number of points
		if dx == None:
			dx = np.median(np.diff(x))
		ntop = dx_top/dx
		nbot = dx_bot/dx

		## Mean error top and bottom regions
		bxtopmeanerr = np.mean(bxerr[ibybotmin:ibybotmax+1])/np.sqrt(ntop)
		bxbotmeanerr = np.mean(bxerr[ibytopmin:ibytopmax+1])/np.sqrt(nbot)

		## BIS error
		biserr = np.sqrt(bxtopmeanerr**2 + bxbotmeanerr**2)

		return bis, bx, by, biserr
	
	return bis, bx, by

# =============================================================================
# S-index 
# =============================================================================


def triangle(vals,mode,lower,upper):
	'''Triangle function for weighting.
	
	Generate a triangular weighting function for a given mode, lower, and upper.
	Used to estimate the Ca II H and K cores as described in Section 2.1 of :cite:t:`Isaacson2010`.

	:param vals: Values to weight.
	:type vals: array
	:param mode: Mode of the triangle.
	:type mode: float
	:param lower: Lower bound of the triangle.
	:type lower: float
	:param upper: Upper bound of the triangle.
	:type upper: float
	
	:return: Weights for the given values.
	:rtype: array

	'''
	left = mode - lower
	right = mode + upper
	weights = np.zeros(len(vals))
	
	for ii, val in enumerate(vals):
		if val >= left and val <= mode:
			weights[ii] = (val-left)/((right-left)*(mode-left))
		if val > mode and val <= right:
			weights[ii] = (right-val)/((right-left)*(right-mode))

	return weights

# =============================================================================
# Statistics
# =============================================================================
def weightedMean(vals,errs,out=True,sigma=5.0):
	'''Calculate weighted mean and error.

	As in Eqs. (14) and (15) in :cite:t:`Zechmeister2018`:

	.. math::
		v = \\frac{\\sum w_o v_o}{\\sum w_o} \, , 
	
	.. math::
		w = \\left ( \\frac{1}{\sum w_o} \cdot \\frac{1}{N_o-1} \cdot \sum (v_o-v)^2 \cdot w_o \\right )^{1/2}
	
	where :math:`w_o = 1/\\sigma_o^2` and :math:`v_o` is the RV for order :math:`o` and :math:`\sigma_o` is the error.

	:param vals: Values.
	:type vals: array
	:param errs: Errors.
	:type errs: array
	:param out: Outlier rejection? Default is True.
	:type out: bool
	:param sigma: Sigma for outlier rejection. Default is 5.0.
	:type sigma: float
	
	:return: Weighted mean, weighted error.
	:rtype: float, float
	
	'''
	
	## reject outliers
	if out:
		mu, sig = scs.norm.fit(vals)
		sig *= sigma
		keep = (vals < (mu + sig)) & (vals > (mu - sig))
		vals = vals[keep]
		errs = errs[keep]

	## calculate the weights
	weights = 1./errs**2
	
	## calculate the weighted mean
	mean = np.sum(weights*vals)/np.sum(weights)
	
	## calculate the weighted error
	term1 = 1./np.sum(1./errs**2)
	term2 = 1./(len(vals)-1.0)
	term3 = np.sum((vals-mean)**2/errs**2)
	err = np.sqrt(term1*term2*term3)

	if out:
		return mean, err, mu, sig

	return mean, err

def outSavGol(wl,fl,efl,iter=2,window_length=51,k=3):
	'''Outlier rejection using Savitzky-Golay filter.
	
	Outlier rejection using Savitzky-Golay filter from :py:func:`scipy.signal.savgol_filter`.

	:param wl: Wavelength.
	:type wl: array
	:param fl: Flux.
	:type fl: array
	:param efl: Error in flux.
	:type efl: array
	:param iter: Number of iterations. Default is 2.
	:type iter: int
	:param window_length: Window length for Savitzky-Golay filter. Default is 51.
	:type window_length: int
	:param k: Polynomial order for Savitzky-Golay filter. Default is 3.
	:type k: int

	:return: Wavelength, flux, error in flux.
	:rtype: array, array, array
	
	'''
	for jj in range(iter):
		## Savitzky-Golay filter for outlier rejection
		yhat = scsig.savgol_filter(fl, window_length, k)
		res = fl-yhat

		## Rejection
		mu, sig = scs.norm.fit(res)
		sig *= 5	
		keep = (res < (mu + sig)) & (res > (mu - sig))
		
		## Trim the arrays
		wl, fl, efl = wl[keep], fl[keep], efl[keep]

	return wl, fl, efl

# =============================================================================
# Broadening function
# =============================================================================

def getBF(fl,tfl,rvr=401,dv=1):
	'''Broadening function.

	Carry out the singular value decomposition (SVD) of the "design  matrix" following the approach in :cite:t:`Rucinski1999`.

	This method creates the "design matrix" by applying a bin-wise shift to the template and uses ``numpy.linalg.svd`` to carry out the decomposition. 
	The design matrix, :math:`\hat{D}`, is written in the form :math:`\hat{D} = \hat{U} \hat{W} \hat{V}^T`. The matrices :math:`\hat{D}`, :math:`\hat{U}`, and :math:`\hat{W}` are stored in homonymous attributes.

	:param fl: Resampled flux.
	:type fl: array
	:param tfl: Resampled template flux.
	:type tfl: array
	:param rvr: Width (number of elements) of the broadening function. Needs to be odd.
	:type rvr: int
	:param dv: Velocity stepsize in km/s.
	:type dv: int

	:return: velocity in km/s, the broadening function
	:rtype: array, array 

	'''
	bn = rvr/dv
	if bn % 2 != 1: bn += 1
	bn = int(bn) # Width (number of elements) of the broadening function. Must be odd.
	bn_arr = np.arange(-int(bn/2), int(bn/2+1), dtype=float)
	vel = -bn_arr*dv

	nn = len(tfl) - bn + 1
	des = np.matrix(np.zeros(shape=(bn, nn)))
	for ii in range(bn): des[ii,::] = tfl[ii:ii+nn]

	## Get SVD deconvolution of the design matrix
	## Note that the `svd` method of numpy returns
	## v.H instead of v.
	u, w, v = np.linalg.svd(des.T, full_matrices=False)

	wlimit = 0.0
	w1 = 1.0/w
	idx = np.where(w < wlimit)[0]
	w1[idx] = 0.0
	diag_w1 = np.diag(w1)

	vT_diagw_uT = np.dot(v.T,np.dot(diag_w1,u.T))

	## Calculate broadening function
	bf = np.dot(vT_diagw_uT,np.matrix(fl[int(bn/2):-int(bn/2)]).T)
	bf = np.ravel(bf)

	return vel, bf

def smoothBF(vel,bf,sigma=5.0):
	'''Smooth broadening function.

	Smooth the broadening function with a Gaussian.

	:param vel: Velocity in km/s.
	:type vel: array
	:param bf: The broadening function.
	:type bf: array
	:param sigma: Smoothing factor. Default 5.0.
	:type sigma: float, optional

	:return: Smoothed BF.
	:rtype: array

	'''
	nn = len(vel)
	gauss = np.zeros(nn)
	gauss[:] = np.exp(-0.5*np.power(vel/sigma,2))
	total = np.sum(gauss)

	gauss /= total
	
	bfgs = scsig.fftconvolve(bf,gauss,mode='same')

	return bfgs

def rotBFfunc(vel,ampl,vrad,vsini,gwidth,const=0.,limbd=0.68):
	'''Rotational profile. 

	The rotational profile obtained by convolving the broadening function with a Gaussian following :cite:t:`Kaluzny2006`.

	:param vel: Velocity in km/s.
	:type vel: array
	:param ampl: Amplitude of BF.
	:type ampl: float
	:param vrad: Radial velocity in km/s, i.e., position of BF.
	:type vrad: float
	:param vsini: Projected rotational velocity in km/s, i.e., width of BF.
	:type vsini: float
	:param const: Offset for BF. Default 0.
	:type const: float, optional
	:param limbd: Value for linear limb-darkening coefficient. Default 0.68.
	:type limbd: float, optional

	:return: rotational profile
	:rtype: array

	'''

	nn = len(vel)
	#bf = np.zeros(nn)
	bf = np.ones(nn)*const

	a1 = (vel - vrad)/vsini
	idxs = np.where(abs(a1) < 1.0)[0]
	asq = np.sqrt(1.0 - np.power(a1[idxs],2))
	bf[idxs] += ampl*asq*(1.0 - limbd + 0.25*np.pi*limbd*asq)

	gs = np.zeros(nn)
	cgwidth = np.sqrt(2*np.pi)*gwidth
	gs[:] = np.exp(-0.5*np.power(vel/gwidth,2))/cgwidth

	rotbf = scsig.fftconvolve(bf,gs,mode='same')

	return rotbf


def rotBFres(params,vel,bf,wf):
	'''Residual rotational profile.
	
	Residual function for :py:func:`rotBFfit`.

	:param params: Parameters. 
	:type params: ``lmfit.Parameters()``
	:param vel: Velocity in km/s.
	:type vel: array
	:param bf: Broadening function.
	:type bf: array
	:param wf: Weights.
	:type wf: array

	:return: residuals
	:rtype: array

	'''
	ampl  = params['ampl1'].value
	vrad  = params['vrad1'].value
	vsini = params['vsini1'].value
	gwidth = params['gwidth'].value
	cons =  params['cons'].value
	limbd = params['limbd1'].value

	res = bf - rotBFfunc(vel,ampl,vrad,vsini,gwidth,cons,limbd)
	return res*wf

def rotBFfit(vel,bf,fitsize,res=67000,smooth=5.0,vsini=5.0,print_report=True):
	'''Fit rotational profile.

	:param vel: Velocity in km/s.
	:type vel: array
	:param bf: Broadening function.
	:type bf: array
	:param fitsize: Interval to fit within.
	:type fitsize: int
	:param res: Resolution of spectrograph. Default 67000 (FIES).
	:type res: int, optional
	:param vsini: Projected rotational velocity in km/s. Default 5.0.
	:type vsini: float, optional
	:param smooth: Smoothing factor. Default 5.0.
	:type smooth: float, optional
	:param print_report: Print the ``lmfit`` report. Default ``True``
	:type print_report: bool, optional 

	:return: result, the resulting rotational profile, smoothed BF
	:rtype: ``lmfit`` object, array, array

	'''
	bfgs = smoothBF(vel,bf,sigma=smooth)
	c = const.c.value*1e-3
	gwidth = np.sqrt((c/res)**2 + smooth**2)

	peak = np.argmax(bfgs)
	idx = np.where((vel > vel[peak] - fitsize) & (vel < vel[peak] + fitsize+1))[0]

	wf = np.zeros(len(bfgs))
	wf[idx] = 1.0

	params = lmfit.Parameters()
	params.add('ampl1', value = bfgs[peak])
	params.add('vrad1', value = vel[peak])
	params.add('gwidth', value = gwidth,vary = True)
	params.add('cons', value = 0.0)
	params.add('vsini1', value = vsini)
	params.add('limbd1', value = 0.68,vary = False)  

	fit = lmfit.minimize(rotBFres, params, args=(vel,bfgs,wf),xtol=1.e-8,ftol=1.e-8,max_nfev=500)
	if print_report: print(lmfit.fit_report(fit, show_correl=False))

	ampl, gwidth = fit.params['ampl1'].value, fit.params['gwidth'].value
	vrad, vsini = fit.params['vrad1'].value, fit.params['vsini1'].value
	limbd, cons = fit.params['limbd1'].value, fit.params['cons'].value
	model = rotBFfunc(vel,ampl,vrad,vsini,gwidth,cons,limbd)

	return fit, model, bfgs

# =============================================================================
# Template matching, preliminary
# =============================================================================

def makeSplines(
		normalized,
		iter=1,
		window_length=51,
		k=3,
		):
	'''Make splines for template matching.
	
	:param normalized: Normalized spectra.
	:type normalized: dict
	:param iter: Number of iterations for outlier rejection through :py:func:`outSavGol` filter. Default 1.
	:type iter: int, optional
	:param window_length: Length of window for outlier rejection. Default 51.
	:type window_length: int, optional
	:param k: Polynomial degree for outlier rejection. Default 3.

	:return: The spline for each order.
	:rtype: dict

	'''
	norders = normalized['orders']
	filenames = normalized['files']

	splines = {}
	## Loop over orders
	for ii, order in enumerate(norders):
		## Collect wavelength, flux, and error arrays
		swl = np.array([])
		fl = np.array([])
		fle = np.array([])
		## and the number of points in each spectrum
		points = np.array([])
		## Loop over files
		for jj, file in enumerate(filenames):
			wl, nfl, nfle = normalized[file][order]['wave'], normalized[file][order]['flux'], normalized[file][order]['err']
			## Get derived RV
			rv = normalized[file]['rv']
			## Shift the spectrum according to the measured velocity
			nwl = wl*(1.0 - rv*1e3/const.c.value)

			swl = np.append(swl,nwl)
			fl = np.append(fl,nfl)
			fle = np.append(fle,nfle)
			points = np.append(points,len(wl))

		## How many points are in the wavelength
		points = np.median(points)

		## Sort the arrays by wavelength
		ss = np.argsort(swl)
		swl, fl, fle = swl[ss], fl[ss], fle[ss]
		
		## Savitzky-Golay filter for outlier rejection
		swl, fl, fle = outSavGol(swl,fl,fle,iter=iter,window_length=window_length,k=k)
		
		## Weights
		w = 1.0/fle

		## Number of knots, 
		## just use the median number of points in wavelength
		Nknots = int(points)

		## The following is a bit hacky
		## The idea is to try to get as close to the knots as returned by the SERVAL spline
		## The knots have to be within the range of the wavelength array
		## ... this "hackiness" includes the knots[4:-4]
		knots = np.linspace(swl[np.argmin(swl)],swl[np.argmax(swl)],Nknots)

		## Get the coefficients for the spline
		t, c, k = sci.splrep(swl, fl, w=w, k=3, t=knots[4:-4])
		## ...and create the spline
		spline = sci.BSpline(t, c, k, extrapolate=False)
		
		splines[order] = [swl,spline]

	return splines

def matchRVs(
		normalized,
		splines,
		pidx=4):
	'''Match spline template with spectra.
	
	Extract RVs using template matching, where the template is created from a spline fit to the spectra.

	:param normalized: Normalized spectra.
	:type normalized: dict
	:param splines: Spline fits to the spectra.
	:type splines: dict
	:param pidx: The indices around the peak of the :math:`\chi^2` polynomium. Default 4.
	:type pidx: int, optional

	:return: The RVs extracted from the spectra.
	:rtype: dict

	'''
	filenames = normalized['files']
	orders = normalized['orders']
	wrvs = {}
	## Loop over files
	for ii, file in enumerate(filenames):
		## Get derived RV
		wrv = normalized[file]['rv']
		drvs = np.arange(wrv-10,wrv+10,0.1)

		## Loop over orders
		rvs = np.array([])
		ervs = np.array([])
		for jj, order in enumerate(orders):
			## Collect the wavelength and spline
			twl, spline = splines[order]
			## Extract the wavelength, flux, and error arrays
			wl, nfl, nfle = normalized[file][order]['wave'], normalized[file][order]['flux'], normalized[file][order]['err']

			chi2s = np.array([])
			for drv in drvs:
				## Shift the spectrum 
				wl_shift = wl/(1.0 + drv*1e3/const.c.value)
				mask = (wl_shift > min(twl)) & (wl_shift < max(twl))
				## Evaluate the spline at the shifted wavelength
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

			## The minimum of the parabola is the best RV
			rv = -pars[1]/(2*pars[0])
			## The curvature is taking as the error.
			erv = np.sqrt(2/pars[0])
			if np.isfinite(rv) & np.isfinite(erv):
				rvs = np.append(rvs,rv)
				ervs = np.append(ervs,erv)
		wavg_rv, wavg_err, _, _  = weightedMean(rvs,ervs,out=1,sigma=5)
		bjd, bvc = getBarycorrs(file,wavg_rv)
		wrvs[file] = [bjd,wavg_rv,wavg_err,bvc]

	return wrvs

def coaddSpectra(
	filenames,
	normalized,
	orders=[]):
	'''Coadd spectra.
	
	Coadd spectra using the normalized spectra.

	:param filenames: List of filenames.
	:type filenames: list
	:param normalized: Normalized spectra.
	:type normalized: dict
	:param orders: List of orders to coadd. If empty, all orders from ``normalized`` are used.
	:type orders: list
	
	:return: Coadded spectra.
	:type: dict


	'''

	## If no orders are specified, 
	## use all orders from the dict
	if not len(orders):
		orders = normalized['orders']

	coadd = {'files':filenames,'orders':orders}
	## For each order append flux and errors for each file
	for order in normalized['orders']:
		wls = np.array([])
		fls = np.array([])
		fles = np.array([])
		for jj, file in enumerate(filenames):
			wl = normalized[file][order]['wave']
			fl = normalized[file][order]['flux']
			fle = normalized[file][order]['err']
			## Append the arrays
			wls = np.append(wls,wl)
			fls = np.append(fls,fl)
			fles = np.append(fles,fle)

		## Copy the wavelength array
		## To keep track of indices
		wlc = wl.copy()

		## Outlier rejection
		wls, fls, fles = outSavGol(wls,fls,fles,iter=2)

		## Coadd the spectra
		fwls = np.array([])
		ffls = np.array([])
		ferr = np.array([])
		## Loop over the wavelengths
		## if wavelength is still in the array
		## take the mean of the fluxes
		## and append to the arrays
		for jj, wl in enumerate(wlc):
			idxs = np.where(wls == wl)[0]
			#if len(idxs) > 1:
			if len(idxs):
				fwls = np.append(fwls,wl)
				ffls = np.append(ffls,np.mean(fls[idxs]))
				## Propagate the errors
				ferr = np.append(ferr,np.sqrt(np.sum(fles[idxs]**2)/len(idxs)))
		## Append the coadded spectrum to the dict
		coadd[order] = {'wave':fwls,'flux':ffls,'err':ferr}

	return coadd

def splineRVs(
	coadd,
	splines,
	drvs,
	orders=[],
	pidx=3,	
	):
	'''Calculate RVs using splines.

	Calculate RVs using splines on the ``coadded`` spectra.

	:param coadd: Coadded spectra.
	:type coadd: dict
	:param splines: Splines for each order.
	:type splines: dict
	:param drvs: RV grid.
	:type drvs: array
	:param orders: List of orders to extract RVs for. If empty, all orders from ``coadd`` are used.
	:type orders: list
	:param pidx: Number of points to use for the parabola fit. Default is 3.
	:type pidx: int

	:return: RVs and errors.
	:type: array, array

	'''
	if not len(orders):
		orders = coadd['orders']

	## Loop over orders
	rvs = np.array([])
	ervs = np.array([])
	for jj, order in enumerate(orders):
		## Collect the wavelength and spline
		twl, spline = splines[order]
		## Extract the wavelength, flux, and error arrays
		wl, nfl, nfle = coadd[order]['wave'], coadd[order]['flux'], coadd[order]['err']

		chi2s = np.array([])
		for drv in drvs:
			## Shift the spectrum 
			wl_shift = wl/(1.0 + drv*1e3/const.c.value)
			mask = (wl_shift > min(twl)) & (wl_shift < max(twl))
			## Evaluate the spline at the shifted wavelength
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

		## The minimum of the parabola is the best RV
		rv = -pars[1]/(2*pars[0])
		## The curvature is taking as the error.
		erv = np.sqrt(2/pars[0])
		if np.isfinite(rv) & np.isfinite(erv):
			rvs = np.append(rvs,rv)
			ervs = np.append(ervs,erv)
	
	wavg_rv, wavg_err, _, _  = weightedMean(rvs,ervs,out=1,sigma=5)
	return wavg_rv, wavg_err

# =============================================================================
# Deprecated
# =============================================================================

# def getRV(vel,ccf,nbins=0,zucker=False,no_peak=50,poly=True,degree=1):
# 	'''Extract radial velocities.
	
# 	Get radial velocity from CCF by fitting a Gaussian and collecting the location, respectively.
# 	Error estimation follows that of :cite:t:`Zucker2003`:
	
# 	.. math:: 
# 		\sigma^2 (v) = - \\left [ N  \\frac{C^{\prime \prime}(\hat{s})}{C(\hat{s})} \\frac{C^2(\hat{s})}{1 - C^2(\hat{s})} \\right ]^{-1} \, ,

# 	where :math:`C(\hat{s})` is the cross-correlation function, and :math:`C^{\prime \prime}(\hat{s})` the second derivative. :math:`N` is the number of bins.
	

# 	:param vel: Velocity in km/s.
# 	:type vel: array
# 	:param ccf: CCF.
# 	:type ccf: array
# 	:param zucker: Error estimation using :cite:t:`Zucker2003`. Default ``True``, else covariance.
# 	:type zucker: bool, optional
# 	:param ccf: CCF.
# 	:type ccf: array
# 	:param nbins: Number of bins.
# 	:type nbins: int, optional IF zucker=False
# 	:param no_peak: Range with no CCF peak, i.e, a range that can constitute a baseline. Default 50.
# 	:type no_peak: float, optional
# 	:param poly: Do a polynomial fit to set the CCF baseline to zero. Default ``True``.
# 	:type poly: bool
# 	:param degree: Degree of polynomial fit. Default 1.
# 	:type degree: int
	
# 	:return: position of Gaussian--radial velocity in km/s, uncertainty of radial velocity in km/s
# 	:rtype: float, float

# 	'''


# 	## starting guesses
# 	idx = np.argmax(ccf)
# 	amp, mu1 = ccf[idx], vel[idx]# get max value of CCF and location
	
# 	## fit for offset in CCF
# 	if poly:
# 		no_peak = (vel - mu1 > no_peak) | (vel - mu1 < -no_peak)
# 		pars = np.polyfit(vel[no_peak],ccf[no_peak],degree)
# 		for ii, par in enumerate(pars):
# 			ccf -= par*vel**(degree-ii)

# 	gau_par, pcov = curve_fit(Gauss,vel,ccf,p0=[amp,mu1,1.0])
# 	#cont = 100*gau_par[0]/np.median(ccf[no_peak])
# 	rv = gau_par[1]
# 	fwhm = 2*np.sqrt(2*np.log(2))*gau_par[2]

# 	perr = np.sqrt(np.diag(pcov))
# 	erv = perr[1]
# 	efwhm = perr[2]
# 	#econt = 100*perr[0]/np.median(ccf[no_peak])
	
# 	if zucker:
# 		assert nbins > 0, print('To estimate uncertainties from Zucker (2003) the number bins must be provided.')
# 		y = ccf
# 		dx = np.mean(np.diff(vel))
# 		## derivatives
# 		yp = np.gradient(y,dx)
# 		ypp = np.gradient(yp,dx)
# 		peak = np.argmax(y)
# 		y_peak = y[peak]
# 		ypp_peak = ypp[peak]
		
# 		sharp = ypp_peak/y_peak
		
# 		snr = np.power(y_peak,2)/(1 - np.power(y_peak,2))
		
# 		erv = np.sqrt(np.abs(1/(nbins*sharp*snr)))
# 	#else:
# 		#vsini = gau_par[2]
# 		#esini = perr[2]
