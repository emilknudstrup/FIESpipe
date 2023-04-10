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

#from scipy.constants import speed_of_light

# =============================================================================
# Preparation/normalization
# =============================================================================

def normalize(wl,fl,bl=np.array([]),poly=1,gauss=True,lower=0.5,upper=1.5):
	'''Normalize spectrum.

	Nomalization of observed spectrum.


	.. note::
		Normalization could probably be improved.

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

	for ii in range(iters):
		cut = np.percentile(nfl,q[ii]) # find upper percentile
		cosmic = nfl > cut # cosmic ray mitigation
		wl, nfl = wl[~cosmic], nfl[~cosmic]
		err = err[~cosmic]
	
	if ret:
		return wl, nfl, err
	return wl, nfl

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

def sumCCF(ccf):
	'''Sum CCF.
	
	Sum CCFs from different orders.

	.. note::
		Add option to weight CCFs by S/N of orders
		or some other weighting scheme.
		BUT the S/N of the CCFs already take the S/N 
		of the flux in a given order into account,
		so probably not necessary.

	:param ccf: CCF.
	:type ccf: array

	:return: Summed CCF.
	:rtype: array
	'''
	ccfsum = np.zeros(ccf.shape[1])
	for ii in range(ccf.shape[0]):
		ccfsum += ccf[ii]
	

	return ccfsum

def getError(rv, ccf, ccferr):
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
	rverr = ccferr / der

	## RV error total
	rverrsum = np.sum(np.power(rverr,-2))
	rverrt = 1./np.sqrt(rverrsum)
	return rverrt, der, rverr

# =============================================================================
# RV/activity indicators through Gaussian fitting
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

def Gauss(x, amp, mu, sig, off):
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
	:param off: y-axis offset of baseline, :math:`B`.
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

def getRV(vel,ccf,ccferr=None,guess=None,flipped=False):
	'''Extract RVs and activity indicators.
	
	Get radial velocity and activity indicators from CCF by fitting a Gaussian.
	Call to :py:func:`fitGauss` to fit Gaussian to CCF.

	.. note::
		Errors might be overestimated when just adopting the formal errors from the fit.

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

	return rv, erv, fwhm, efwhm, cont, econt

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


	.. note::
		Will (currently) only work now if the CCF has a peak (rather than a dip as for an absorption line).

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


	#ax2.plot(x[left],y[left],marker='o')
	#ax2.plot(x[peak],y[peak],marker='o')
	#ax2.plot(x[right-1],y[right-1],marker='o')

	## Y span for bisector
	by = np.linspace(y_small,y[peak],n)

	#ax2.plot(x[left:peak+1],y[left:peak+1])
	#ax2.plot(x[peak:right],y[peak:right])

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

		return bis, biserr
	
	return bis

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
		return mean, err, mu, sigma

	return mean, err


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
	
# 	:returns: position of Gaussian--radial velocity in km/s, uncertainty of radial velocity in km/s
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
