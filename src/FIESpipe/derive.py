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

def contNormalize(wl,fl,bins=100,pdeg=10):
	'''Continuum normalize spectrum.

	Continuum normalization of observed spectrum.

	Following :py:class:`hipparchus.EchelleSpectrum.continuum_normalize`:
	https://github.com/bmorris3/hipparchus/blob/master/hipparchus/io.py#L180

	:param wl: Observed wavelength
	:type wl: array
	:param fl: Observed raw flux
	:type fl: array
	:param bins: Number of bins for max filtering. Default 100.
	:type bins: int, optional
	:param pdeg: Degree of polynomial fit to normalize. Default 10.
	:type pdeg: int, optional

	:return: observed wavelength, observed normalized flux
	:rtype: array, array
	'''

	## Normalize the spectrum
	## max filtering
	bs = scs.binned_statistic(wl-wl.mean(), fl, bins=bins,statistic='max')
	bincenters = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])
	## Fit a polynomial to the continuum and divide it out
	fit = np.polyval(np.polyfit(bincenters, bs.statistic, pdeg),wl-wl.mean())
	nfl = fl/fit

	return nfl

def excludeErrors(wave, flux, error, sigma=5):
	'''Exclude large errors from the spectrum.
	
	This is done by calculating the median of the error and excluding points that are above a certain threshold (sigma times the median).
	Also exclude negative errors, which can occur due to issues with the data reduction.
	
	:param wave: Wavelength array
	:type wave: numpy array

	:param flux: Flux array
	:type flux: numpy array

	:param error: Error array
	:type error: numpy array
	
	:param sigma: Threshold for excluding points (default is 5)
	:type sigma: float

	:return: Tuple of (wave, flux, error, mask) where mask is a boolean array indicating which points were excluded
	:rtype: tuple

	'''

	## Negative errors or flux
	mask = (error < 0) | (flux < 0)

	## Large errors
	med = np.nanmedian(abs(error[~mask]))
	mask |= abs(error) > sigma*med
	mask = ~mask

	return wave[mask], flux[mask], error[mask], ~mask

# def crm(wl,nfl,err=np.array([]),iters=1,q=[99.0,99.9,99.99], insert_nans=False):
# 	'''Cosmic ray mitigation.
	
# 	Excludes flux over qth percentile.

# 	.. note::
# 		There might be a more sophisticated way of doing this.
# 		Also, need to make sure it's working properly without flux errors.


# 	:param wl: Observed wavelength.
# 	:type wl: array 
# 	:param nfl: Observed normalized flux.
# 	:type: array
# 	:param err: Observed flux error. Default ``numpy.array([])``. Leave empty, if no error available.
# 	:type err: array, optional
# 	:param iters: Iterations of removing upper q[iter] percentile. Default 1.
# 	:type iters: int, optional
# 	:param q: Percentiles. Default ``[99.0,99.9,99.99]``.
# 	:type q: list, optional
# 	:param insert_nans: If True, insert NaNs for excluded points instead of removing them. Default False.
	
# 	:return: observed wavelength, observed normalized flux
# 	:rtype: array, array
# 	'''
# 	assert iters <= len(q), 'Error: More iterations than specified percentiles.'

# 	idxs = np.array([],dtype=int)
# 	idxs = np.append(idxs,np.where(np.isnan(nfl))[0])

# 	ret = True
# 	if not len(err):
# 		err = np.ones(len(wl))
# 		ret = False

	
# 	for ii in range(iters):
# 		cut = np.nanpercentile(nfl,q[ii]) # find upper percentile
# 		cosmic = nfl > cut # cosmic ray mitigation
# 		wl, nfl = wl[~cosmic], nfl[~cosmic]
# 		err = err[~cosmic]
# 		idxs = np.append(idxs,np.where(cosmic)[0])
# 	if ret:
# 		return wl, nfl, err, idxs
# 	return wl, nfl, idxs


# def crm(wl,fl,err=np.array([]),sigma=5,iters=1):
# 	'''Cosmic ray mitigation.

# 	Excludes flux values sigma above the mean.
	

# 	'''
	
# 	ret = True
# 	if not len(err):
# 		err = np.ones(len(wl))
# 		ret = False

# 	# idxs = np.array([],dtype=int)
# 	# idxs = np.append(idxs,np.where(np.isnan(fl))[0])
	
# 	for ii in range(iters):
# 		mu = np.nanmean(fl)
# 		sig = np.nanstd(fl)
# 		cosmic = fl > (mu + sigma*sig)
# 		wl[cosmic], fl[cosmic], err[cosmic] = np.nan, np.nan, np.nan
	
# 	idxs = np.where(np.isnan(fl))[0]
# 	if ret:
# 		return wl[~np.isnan(fl)], fl[~np.isnan(fl)], err[~np.isnan(fl)], idxs
# 	return wl[~np.isnan(fl)], fl[~np.isnan(fl)], idxs

def crm(wl,fl,err=np.array([]),sigma=5,iters=1):
	'''Cosmic ray mitigation.

	Excludes flux values sigma above the mean.

	:param wl: Wavelength array
	:type wl: numpy array

	:param fl: Flux array
	:type fl: numpy array

	:param err: Error array. Optional, defaults to an empty array. If not provided, the function will return only the wavelength and flux arrays without the error array.
	:type err: numpy array
	
	:param sigma: Threshold for excluding points (default is 5)
	:type sigma: float

	:param iters: Number of iterations for the cosmic ray mitigation (default is 1)
	:type iters: int

	:return: Tuple of (wl, fl, err, mask) where mask is a boolean array indicating which points were excluded. If the error array is not provided, the function will return (wl, fl, mask).
	:rtype: tuple
	'''
	
	ret = True
	if not len(err):
		err = np.ones(len(wl))
		ret = False

	fl_copy = fl.copy()
	for ii in range(iters):
		mu = np.nanmean(fl_copy)
		sig = np.nanstd(fl_copy)
		cosmic = np.where(fl_copy > (mu + sigma*sig))[0]
		fl_copy[cosmic] = np.nan
	
	## Find indices of masked points (NaNs) to return
	out = np.isnan(fl_copy)
	idxs = np.isfinite(fl_copy)
	if ret:
		return wl[idxs], fl[idxs], err[idxs], out
	return wl[idxs], fl[idxs], out


def calcSNR(wave, flux, ferr, edge=0):
	'''Calculate SNR.

	Calculate SNR in a given wavelength range defined by edges.

	:param wave: Wavelength array.
	:type wave: array
	
	:param flux: Flux array.
	:type flux: array

	:param ferr: Flux error array.
	:type ferr: array

	:param edge: Skip edge of detector - low S/N - in Angstrom. Default 0.0 Å.
	:type edge: float, optional

	:return: SNR.
	:rtype: float
	'''

	mask = (wave >= (wave[0]-edge)) & (wave <= (wave[-1]+edge))
	signal = np.median(flux[mask])
	noise = np.median(ferr[mask])
	if noise == 0:
		return np.inf
	else:
		return signal / noise

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
	
	flip_fl, flip_tfl = 1-nfl, 1-tfl

	r_fle = np.interp(lam,wl,fle)
	r_fl = np.interp(lam,wl,flip_fl)
	r_tl = np.interp(lam,twl,flip_tfl)
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


def multiEpochClipping(flux_array, sigma_thresh=5):
	'''Cross-epoch robust clipping.

	Clip outliers in each pixel column across epochs based on a robust
	median and MAD estimate, and replace clipped values with the per-pixel
	median.

	:param flux_array: Flux array with shape (n_epochs, n_pixels).
	:type flux_array: array
	:param sigma_thresh: Sigma threshold used for clipping. Default is 5.
	:type sigma_thresh: float, optional

	:return: Cleaned flux array and boolean mask where True indicates good values.
	:rtype: array, array
	'''

	flux_array = np.asarray(flux_array)

	median = np.median(flux_array, axis=0)
	mad = np.median(np.abs(flux_array - median), axis=0)
	sigma = 1.4826 * mad

	# Avoid division by zero
	sigma[sigma == 0] = np.nan

	residuals = flux_array - median
	mask = np.abs(residuals) < sigma_thresh * sigma

	cleaned = flux_array.copy()
	cleaned[~mask] = np.tile(median, (flux_array.shape[0], 1))[~mask]

	return cleaned, mask


# =============================================================================
# Cross-correlation
# =============================================================================

def getCCF(fl,tfl,fle,dv=1.0,rvr=401,ccf_mode='full'):
	'''Cross-correlation function.

	Perform the cross correlation 

	.. math:: 
		\mathrm{CCF}(v) = \Sigma_i f_i t_i (v) \, ,

	The CCF is normalized through

	.. math::
		a = \frac{\mathrm{CCF}(v)}{\sigma_f \sigma_t N} \, ,
	
	where :math:`\sigma_f` and :math:`\sigma_t` are the standard deviations of the flux and template, respectively, and :math:`N` is the number of bins.

	The variance is calculated as:

	.. math:: 
		\sigma_{\mathrm{CCF}}^2 (v) = \Sigma_i \sigma_i^2 [t_i (v)]^2 \, ,

	and the normalized error on the CCF is then
	.. math::
		\sigma_{\mathrm{CCF}} (v) = \sqrt{\Sigma_i \sigma_i^2 [t_i (v)]^2}/ a \, .

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
	ccferr = np.correlate(fle**2,tfl**2,mode=ccf_mode)
	## normalize ccf
	norm = np.std(fl) * np.std(tfl) * len(tfl)
	ccf = ccf/(norm)
	## normalize errors from ccf
	ccferr = np.sqrt(ccferr)/(norm)
	
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
	# ccf = ccf/np.mean(ccf) - 1 # shift 'continuum' to zero
	
	return rvs, ccf, ccferr

	# .. math:: 
	# 	\sigma^2 (v) = - \\left [ N  \\frac{C^{\prime \prime}(\hat{s})}{C(\hat{s})} \\frac{C^2(\hat{s})}{1 - C^2(\hat{s})} \\right ]^{-1} \, ,

	# where :math:`C(\hat{s})` is the cross-correlation function, and :math:`C^{\prime \prime}(\hat{s})` the second derivative. :math:`N` is the number of bins.
	

def getCCFBinMask(wl, nfl, fle, mask_wl, mask_w, rvs):
	'''Pixel-space CCF.

	Pixel-space CCF following Lafarga et al. (2020) Sects. 4.4-4.5.

	Each mask line is a Dirac-delta mapped to a 1-pixel-wide bin expressed
	in the *pixel coordinate* of the spectral order (Sect. 4.4).  For each
	trial RV the Doppler-shifted mask-line wavelength is mapped to a
	fractional pixel index via linear interpolation of the inverse
	(wavelength->pixel) mapping; the +-0.5-pixel bin is then overlapped with
	the 1-2 adjacent spectrum pixels and the normalised flux is linearly
	interpolated at the centre of each overlap region in pixel space
	(Sect. 4.4, last paragraph).

	The CCF is in the *flux* (dip) convention: baseline ~ 1 for a
	continuum-normalised spectrum, minimum at the stellar RV.

	:param wl: Wavelength.
	:type wl: array
	:param nfl: Normalized flux.
	:type nfl: array
	:param fle: Flux errors.
	:type fle: array
	:param mask_wl: Mask line wavelengths.
	:type mask_wl: array
	:param mask_w: Mask line weights.
	:type mask_w: array
	:param rvs: Velocity grid.
	:type rvs: array

	:return: velocity grid (passed through), CCF for this order (flux dip form), photon-noise uncertainty on the CCF, number of mask lines that contributed, sum of weights of the contributing lines
	:rtype: array, array, array, int, float
	'''

	c     = 299792.458  # km/s
	nspec = len(wl)

	# Integer pixel coordinates: pix[j] = j
	pix = np.arange(nspec, dtype=float)

	# Restrict to mask lines reachable in this order for any trial RV
	max_dv   = np.max(np.abs(rvs))
	in_range = (
		(mask_wl * (1.0 + max_dv / c) >= wl[0]) &
		(mask_wl * (1.0 - max_dv / c) <= wl[-1])
	)
	mwl = mask_wl[in_range]
	mw  = mask_w[in_range]

	if mwl.size == 0:
		return rvs, np.zeros_like(rvs), np.zeros_like(rvs), 0, 0.0

	nwl = len(mwl)
	nrv = len(rvs)

	# Doppler-shifted mask-line wavelengths: shape (nwl, nrv)
	wmsh = mwl[:, None] * (1.0 + rvs[None, :] / c)

	# Map shifted wavelengths to fractional pixel coordinates via linear
	# interpolation of the inverse (wavelength->pixel) mapping.
	pixmsh = np.interp(wmsh.ravel(), wl, pix).reshape(nwl, nrv)

	# Mask-bin edges in pixel space: +/- 0.5 pixel (Sect. 4.4)
	pm1 = pixmsh - 0.5   # left  edge, shape (nwl, nrv)
	pm2 = pixmsh + 0.5   # right edge, shape (nwl, nrv)

	# The (at most) two spectrum pixels whose +-0.5 bounds overlap the bin:
	#   j_L = floor(pixmsh),  j_R = ceil(pixmsh)
	# For non-integer pixmsh these are two adjacent pixels; for integer
	# pixmsh they coincide (j_L == j_R -> single-pixel overlap).
	j_L = np.clip(np.floor(pixmsh).astype(int), 0, nspec - 1)
	j_R = np.clip(np.ceil (pixmsh).astype(int), 0, nspec - 1)

	# --- Left-pixel contribution ---
	ov_lo_L = np.maximum(j_L.astype(float) - 0.5, pm1)
	ov_hi_L = np.minimum(j_L.astype(float) + 0.5, pm2)
	ov_L    = np.maximum(ov_hi_L - ov_lo_L, 0.0)
	pcen_L  = 0.5 * (ov_lo_L + ov_hi_L)

	# --- Right-pixel contribution (zero when j_R == j_L) ---
	same    = (j_L == j_R)
	ov_lo_R = np.maximum(j_R.astype(float) - 0.5, pm1)
	ov_hi_R = np.minimum(j_R.astype(float) + 0.5, pm2)
	ov_R    = np.where(same, 0.0, np.maximum(ov_hi_R - ov_lo_R, 0.0))
	pcen_R  = np.where(same, pcen_L, 0.5 * (ov_lo_R + ov_hi_R))

	# Flux and error interpolated at each overlap centre (pixel space)
	f_L = np.interp(pcen_L.ravel(), pix, nfl).reshape(nwl, nrv)
	e_L = np.interp(pcen_L.ravel(), pix, fle).reshape(nwl, nrv)
	f_R = np.interp(pcen_R.ravel(), pix, nfl).reshape(nwl, nrv)
	e_R = np.interp(pcen_R.ravel(), pix, fle).reshape(nwl, nrv)

	# Weighted contributions summed over mask lines -> CCF and its variance
	mw2d       = mw[:, None]                                     # (nwl, 1)
	contrib_f  = mw2d * (ov_L * f_L + ov_R * f_R)               # (nwl, nrv)
	contrib_e2 = mw2d**2 * (ov_L**2 * e_L**2 + ov_R**2 * e_R**2)

	ccf    = np.sum(contrib_f,          axis=0)                  # (nrv,)
	ccferr = np.sqrt(np.sum(contrib_e2, axis=0))

	return rvs, ccf, ccferr, nwl, float(np.sum(mw))

def getRVuncertainty(rvs, ccf, ccferr):
	'''Photon-noise limited RV uncertainty.

	Estimate the RV uncertainty from the CCF profile following Eqs. (9)-(10)
	in :cite:t:`Lafarga2020` (see also :cite:t:`Bouchy2001`).

	At each velocity point:

	.. math::
		\sigma_v(v) = \sigma_{\mathrm{CCF}}(v)\left|\frac{\mathrm{dCCF}}{\mathrm{d}v}\right|^{-1}

	and the final uncertainty is:

	.. math::
		\sigma_{\mathrm{RV}} = \left(\sum_v \frac{1}{\sigma_v(v)^2}\right)^{-1/2}

	:param rvs: Velocity grid in km/s.
	:type rvs: array
	:param ccf: Cross-correlation function values.
	:type ccf: array
	:param ccferr: Uncertainty on the CCF values.
	:type ccferr: array

	:return: Photon-noise limited RV uncertainty.
	:rtype: float

	.. note::
		For realistic uncertainties, the CCF sampling should be close to the
		pixel velocity scale (Sect. 3.5 in :cite:t:`Lafarga2020`).
	'''
	dccf_dv = np.gradient(ccf, rvs)  # d(CCF)/d(v) in units of 1/(km/s)

	with np.errstate(divide='ignore', invalid='ignore'):
		# Information weight at each point: W(v) = (dCCF/dv)^2 / sigma_CCF^2
		W = np.where(ccferr > 0, dccf_dv**2 / ccferr**2, 0.0)

	sigma_rv = 1.0 / np.sqrt(np.sum(W)) if np.sum(W) > 0 else np.inf
	return sigma_rv

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

def weightedCoaddCCF(order_ccfs, order_ccferrs, erv_orders, orders):
	'''Inverse-variance weighted co-addition of per-order CCFs.

	Co-adds CCFs from individual orders weighting each by the per-order
	photon-noise RV precision (:math:`w_i = 1/\\sigma_{\\rm RV,i}^2`).
	This suppresses the contribution of noisy or sparsely-covered orders,
	mirroring the inverse-variance weighting used in template-matching
	(e.g. SERVAL, :cite:t:`Zechmeister2018`).

	The propagated CCF uncertainty is:

	.. math::
		\\sigma_C(v) = \\frac{\\sqrt{\\sum_i w_i^2 \\, \\sigma_{C_i}(v)^2}}{\\sum_i w_i}

	:param order_ccfs: Per-order CCF values, shape ``(norders_total, nrv)``.
	:type order_ccfs: numpy.ndarray
	:param order_ccferrs: Per-order CCF uncertainties, shape ``(norders_total, nrv)``.
	:type order_ccferrs: numpy.ndarray
	:param erv_orders: Per-order photon-noise RV uncertainty (km/s), length ``norders_total``.
	:type erv_orders: numpy.ndarray
	:param orders: Indices of orders to include in the co-addition.
	:type orders: array-like

	:return: Co-added CCF, propagated CCF uncertainty.
	:rtype: tuple of numpy.ndarray
	'''
	nrv       = order_ccfs.shape[1]
	ccf_sum   = np.zeros(nrv)
	ccferr_sq = np.zeros(nrv)
	wsum      = 0.0

	for i in orders:
		erv_i = erv_orders[i]
		if not (np.isfinite(erv_i) and erv_i > 0):
			continue
		w          = 1.0 / erv_i**2
		ccf_sum   += w * order_ccfs[i]
		ccferr_sq += w**2 * order_ccferrs[i]**2
		wsum      += w

	if wsum == 0.0:
		return np.zeros(nrv), np.zeros(nrv)

	return ccf_sum / wsum, np.sqrt(ccferr_sq) / wsum

def coaddCCFs(order_ccfs, order_ccferrs, erv_array=None, orders=None, order_wsums=None, order_nlines=None, coadd_mode='snr_weighted'):
	''' Co-add CCFs from individual orders.

		Toggle between co-addition modes:
			'simple'       : plain sum of high-SNR order CCFs with Lafarga+2020 normalisation (fixed).
			'snr_weighted' : inverse-variance weighted co-addition using per-order photon-noise RV precision.

	:param order_ccfs: Per-order CCF values, shape ``(norders_total, nrv)``.
	:type order_ccfs: numpy.ndarray
	:param order_ccferrs: Per-order CCF uncertainties, shape ``(norders_total, nrv)``.
	:type order_ccferrs: numpy.ndarray
	:param erv_array: Optional. Per-order photon-noise RV uncertainty (km/s), length ``norders_total``.
	:type erv_array: numpy.ndarray, optional
	:param orders: Optional. Indices of orders to include in the co-addition.
	:type orders: array-like, optional
	:param order_wsums: Optional. Per-order sum of weights (for 'simple' mode). Default None.
	:type order_wsums: numpy.ndarray, optional
	:param order_nlines: Optional. Per-order number of contributing lines (for 'simple' mode). Default None.
	:type order_nlines: numpy.ndarray, optional
	:param coadd_mode: Co-addition mode. Options are 'simple' or 'snr_weighted'. Default is 'snr_weighted'.
	:type coadd_mode: str, optional

	:return: Co-added CCF, propagated CCF uncertainty.
	:rtype: tuple of numpy.ndarray
	'''
	assert coadd_mode in ['simple', 'snr_weighted'], f"Unknown coadd_mode '{coadd_mode}'. Use 'simple' or 'snr_weighted'."
	if coadd_mode == 'simple':
		## Assert that order_wsums and order_nlines are provided
		if order_wsums is None or order_nlines is None:
			raise ValueError("For 'simple' co-addition mode, order_wsums and order_nlines must be provided.")

		## Plain sum of high-SNR order CCFs.
		## Normalisation follows Lafarga+2020 Sect. 4.5, but restricted to
		## the selected orders so that avg_weight is self-consistent.
		total_weight = np.sum(order_wsums[orders])
		total_nlines = np.sum(order_nlines[orders])

		ccfs    = np.zeros_like(order_ccfs[0])
		ccferrs = np.zeros_like(order_ccferrs[0])
		for i in orders:
			ccfs    += order_ccfs[i]
			ccferrs += order_ccferrs[i]**2
		ccferrs = np.sqrt(ccferrs)
		if total_nlines > 0:
			avg_weight = total_weight / total_nlines
			ccfs    /= avg_weight
			ccferrs /= avg_weight

	elif coadd_mode == 'snr_weighted':
		## Assert that erv_array and norders is provided
		if erv_array is None or orders is None:
			raise ValueError("For 'snr_weighted' co-addition mode, erv_array and orders must be provided.")
		## Inverse-variance weighted co-addition using per-order photon-noise
		## RV precision.  The absolute CCF scale is set by the weighting and
		## is not comparable to the Lafarga normalisation, but the Gaussian
		## fit for RV, FWHM and contrast is unaffected by the overall scale.
		ccfs, ccferrs = weightedCoaddCCF(order_ccfs, order_ccferrs, erv_array, orders)

	else:
		raise ValueError(f"Unknown coadd_mode '{coadd_mode}'. Use 'simple' or 'snr_weighted'.")

	return ccfs, ccferrs

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
	## maxfev raised from the scipy default of 1000, since the default is
	## sometimes too low for noisy/shallow CCFs to converge
	gau_par, pcov = curve_fit(Gauss,xx,yy,p0=guess,sigma=yerr,maxfev=5000)

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

	try:
		gau_par, pcov = fitGauss(vel,ccf,yerr=ccferr,guess=guess,flipped=flipped)
	except RuntimeError:
		## Gaussian fit did not converge (e.g. noisy/shallow CCF).
		## Return NaNs with the same shapes/signature as a successful fit,
		## so callers can treat this the same way as any other bad epoch/order
		## (e.g. with np.isfinite checks) instead of crashing.
		nan_pars = np.full(4, np.nan)
		nan_pcov = np.full((4,4), np.nan)
		if return_pars:
			return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, nan_pars, nan_pcov
		return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
	
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

	Implemented identically to :py:func:`raccoon.ccf.computebisector_biserr`:
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

def computeBIS(rvs_epoch, ccfs, ccferrs):
	'''Bisector inverse slope from the co-added CCF.

	Wrapper around :py:func:`fp.getBIS` that handles the dip (absorption-line)
	convention of :py:func:`fp.getCCFBinMask` by flipping the CCF, and
	derives per-point velocity uncertainties from the CCF noise and its
	gradient for error propagation.

	:param rvs_epoch: Velocity grid in km/s.
	:type rvs_epoch: numpy.ndarray
	:param ccfs: Co-added CCF (dip convention; minimum at stellar RV).
	:type ccfs: numpy.ndarray
	:param ccferrs: Photon-noise uncertainty on the co-added CCF.
	:type ccferrs: numpy.ndarray

	:return: Bisector inverse slope (km/s), BIS uncertainty (km/s).
	:rtype: tuple of float
	'''
	dccf_dv = np.gradient(ccfs, rvs_epoch)
	with np.errstate(divide='ignore', invalid='ignore'):
		xerr_bis = np.where(np.abs(dccf_dv) > 0, ccferrs / np.abs(dccf_dv), np.inf)
	return getBIS(rvs_epoch, -ccfs, xerr=xerr_bis)


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
def weightedMean(vals,errs,error_estimation='Zechmeister2018',out='mad',sigma=5.0):
	'''Calculate weighted mean and error.

	As in Eqs. (14) and (15) in :cite:t:`Zechmeister2018`:

	.. math::
		v = \\frac{\\sum w_o v_o}{\\sum w_o} \, , 
	
	.. math::
		w = \\left ( \\frac{1}{\sum w_o} \cdot \\frac{1}{N_o-1} \cdot \sum (v_o-v)^2 \cdot w_o \\right )^{1/2}
	
	where :math:`w_o = 1/\\sigma_o^2` and :math:`v_o` is the RV for order :math:`o` and :math:`\sigma_o` is the error.

	.. note::
		- When using the error estimation from :cite:t:`Zechmeister2018`, the error is inflated by the chi-squared of the fit, as in Eq. (14) alternatively to the formal error in Eq. (15) in :cite:t:`Zechmeister2018`. The former is more conservative, but the latter might be more appropriate when the errors are well estimated. 


	:param vals: Values.
	:type vals: array
	:param errs: Errors.
	:type errs: array
	:param error_estimation: Method for error estimation. Options are 'Zechmeister2018' or 'formal'. Default is 'Zechmeister2018'.
	:type error_estimation: str
	:param out: Outlier rejection method. Options are 'mad' or 'gauss'. Default is 'mad'.
	:type out: str or False
	:param sigma: Sigma for outlier rejection in case of 'gauss'. Default is 5.0.
	:type sigma: float
	
	:return: Weighted mean, weighted error.
	:rtype: float, float
	
	'''
	assert error_estimation in ['Zechmeister2018','formal'], 'Error: error_estimation must be one of "Zechmeister2018" or "formal".'
	assert out in ['mad','gauss', False], 'Error: out must be one of "mad", "gauss", or False.'

	## reject outliers
	if out == 'mad':
		med = np.median(vals)
		mad = 1.4826*np.median(np.abs(vals - med))
		good = np.abs(vals - med) < 3*mad
	elif out == 'gauss':
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
	if error_estimation == 'Zechmeister2018':
		term1 = 1./np.sum(1./errs**2)
		term2 = 1./(len(vals)-1.0)
		term3 = np.sum((vals-mean)**2/errs**2)
		err = np.sqrt(term1*term2*term3)
	elif error_estimation == 'formal':
		err = np.sqrt(1.0 / np.sum(weights))


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
	bn = int(np.ceil(rvr/dv))
	if bn % 2 == 0: bn += 1 # Width (number of elements) of the broadening function. Must be odd.
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

	The rotational profile obtained by convolving the broadening function with a Gaussian following Eq. (2) in :cite:t:`Kaluzny2006`.

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

def rotBFfit(vel,bf,fitsize,
		res=67000,smooth=5.0,vsini=5.0,
		vary_gwidth=True,
		ldc=0.68,vary_ldc=False,
		print_report=True):
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
	:param vary_gwidth: Vary the Gaussian width? Default True.
	:type vary_gwidth: bool, optional
	:param ldc: Linear limb-darkening coefficient. Default 0.68.
	:type ldc: float, optional
	:param vary_ldc: Vary the linear limb-darkening coefficient? Default False.
	:type vary_ldc: bool, optional
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
	params.add('gwidth', value = gwidth, vary = vary_gwidth)
	params.add('cons', value = 0.0)
	params.add('vsini1', value = vsini)
	params.add('limbd1', value = ldc, vary = vary_ldc)  

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
		filenames=None,
		iter=1,
		window_length=51,
		k=3,
		):
	'''Make splines for template matching.

	
	Cubic B-splines created using :py:func:`scipy.interpolate.splrep` (`here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html>`__)
	and :py:func:`scipy.interpolate.Bspline` (`here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline>`__).
	
	:param normalized: Normalized spectra.
	:type normalized: dict
	:param filenames: Filenames for spectra. Default ``None``, in which case the filenames in ``normalized`` are used.
	:type filenames: list, optional
	:param iter: Number of iterations for outlier rejection through :py:func:`outSavGol` filter. Default 1.
	:type iter: int, optional
	:param window_length: Length of window for outlier rejection. Default 51.
	:type window_length: int, optional
	:param k: Polynomial degree for outlier rejection. Default 3.

	:return: The spline for each order.
	:rtype: dict

	'''
	norders = normalized['orders']
	if filenames is None: filenames = normalized['files']

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
		# wavg_rv, wavg_err, _, _  = weightedMean(rvs,ervs,out=1,sigma=5)
		wavg_rv, wavg_err = weightedMean(rvs,ervs,out='mad')
		bjd, bvc = getBarycorrs(file,wavg_rv)
		wrvs[file] = [bjd,wavg_rv,wavg_err,bvc]

	return wrvs

def coaddSpectra(
	filenames,
	normalized,
	orders=[]):
	'''Coadd spectra.
	
	Coadd spectra using the normalized spectra.

	.. note::
		The coadding here is done in a way that assumes that the spectra in the epoch
		have the exact same wavelength solution. 
	
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
			## Alternatively, it would be something along the line
			## idxs = np.where(np.abs(wls-wl)<deltaWL)[0]
			## where deltaWL is some threshold
			## Could be set to be a fraction of the median spacing 
			## between wavelength points in each of the spectra

			if len(idxs):
				## If the wavelength steps are not the same
				## then it should be 
				## fwls = np.append(fwls,np.mean(wls[idxs]))
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
		## The curvature is taken as the error.
		erv = np.sqrt(2/pars[0])
		if np.isfinite(rv) & np.isfinite(erv):
			rvs = np.append(rvs,rv)
			ervs = np.append(ervs,erv)
	
	wavg_rv, wavg_err = weightedMean(rvs,ervs,out='mad')
	# wavg_rv, wavg_err, _, _  = weightedMean(rvs,ervs,out=1,sigma=5)
	return wavg_rv, wavg_err

# =============================================================================
# ThAr
# =============================================================================

def tharSplines(
		prepped,
		norders=range(40,60)
		):
	'''Create splines for ThAr spectra.

	Create splines for ThAr spectra using the prepped ThAr spectra.

	:param prepped: Prepped ThAr spectra.
	:type prepped: dict
	:param norders: List of orders to create splines for. Default ``range(40,60)``.
	:type norders: list, optional
	
	:return: Splines for each order.
	:rtype: dict

	'''
	## Storage for splines
	## and normalized ThAr spectra
	splines = {}
	tharnames = prepped['files']

	for ii, order in enumerate(norders):
		## Collect the wavelength and flux arrays
		nwl = np.array([])
		nfl = np.array([])
		points = np.array([])
		for jj, file in enumerate(tharnames):
			# ## Extract the data from the FITS file
			# wave, flux, _, hdr = fp.extractFIES(file)
			# w = wave[order,:]
			# f = flux[order,:]

			# ## Normalize the spectrum
			# nw, nf = fp.normalize(w,f)

			## Extract prepped ThAr spectra
			nw, nf = prepped[file][order]

			nwl = np.append(nwl,nw)
			nfl = np.append(nfl,nf)
			points = np.append(points,len(nw))

		## How many points are in the wavelength
		## Number of knots
		Nknots = int(np.median(points))

		## Sort the arrays by wavelength
		ss = np.argsort(nwl)
		nwl, nfl = nwl[ss], nfl[ss]

		## Knots for the spline
		knots = np.linspace(nwl[np.argmin(nwl)],nwl[np.argmax(nwl)],Nknots)

		## Get the coefficients for the spline
		t, c, k = sci.splrep(nwl, nfl, k=3, t=knots[4:-4])
		## ...and create the spline
		spline = sci.BSpline(t, c, k, extrapolate=False)

		splines[order] = [nwl,spline]

	return splines

def thaRVs(
		prepped,
		splines,
		drvs=np.linspace(-0.5,0.5,40),
		pidx=3
		):
	'''
	Extract RVs from ThAr spectra.
	
	Extract RVs from ThAr spectra using template matching, where the template is created from a spline fit to the spectra.
	
	:param prepped: Prepped ThAr spectra.
	:type prepped: dict
	:param splines: Splines for each order.
	:type splines: dict
	:param drvs: RV grid. Default ``np.linspace(-0.5,0.5,40)`` (km/s).
	:type drvs: array, optional
	:param pidx: The indices around the peak of the :math:`\chi^2` polynomium. Default 3.
	:type pidx: int, optional
	
	:return: The RVs extracted from the spectra.
	:rtype: dict
	'''
	tharnames = prepped['files']
	norders = prepped['orders']

	## Dictionary to store the results
	wrvs = {}
	## Loop over files
	for ii, file in enumerate(tharnames):
		rvs = np.array([])
		ervs = np.array([])
		## Loop over orders
		for jj, order in enumerate(norders):
			## Collect the wavelength and spline
			twl, spline = splines[order]
			## Normalized ThAr
			nwl, nfl = prepped[file][order]
			chi2s = np.array([])
			for drv in drvs:
				## Shift the spectrum
				wl_shift = nwl/(1.0 + drv*1e3/const.c.value)
				mask = (wl_shift > min(twl)) & (wl_shift < max(twl))
				
				## Evaluate the spline at the shifted wavelength
				ys = spline(wl_shift[mask])
				## There are no errors on the ThAr spectra
				## delivered by FIEStool
				chi2 = np.sum((ys - nfl[mask])**2)
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
		# wavg_rv, wavg_err, _, _  = weightedMean(rvs,ervs,out=1,sigma=5)
		wavg_rv, wavg_err  = weightedMean(rvs,ervs,out='mad')
		bjd, _ = getBarycorrs(file,rvmeas=0.0)
		wrvs[file] = [bjd,wavg_rv,wavg_err]

	return wrvs

def ThArcorr(
		thimes,
		tharvs,
		times
		):
	'''Correct for the ThAr drift.

	Here the RVs from the ThAr exposures are used to correct the drift in the RVs of the science exposures.
	Here it is assumed that all science exposures are sandwiched between two ThAr exposures, schematically:

	.. graphviz::

		digraph acq {
			rankdir="LR";
			th1 -> sci1 -> th2;
			th1 [shape=box, label="ThAr"];
			th2 [shape=box, label="ThAr"];
			sci1 [label="Science"];
		}

	, which is also when it only really makes sense to apply this correction.

	:param thimes: BJDs for ThAr spectra.
	:type thimes: array
	:param tharvs: RVs for ThAr spectra.
	:type tharvs: array
	:param times: BJDs for science spectra.
	:type times: array

	:return: :math:`\Delta` RVs to be applied to RVs from science frames.
	:rtype: array

	'''

	dvs = np.array([])
	for ii, bjd in enumerate(times):
		## Find the two closest ThAr exposures
		close = np.argsort(np.abs(thimes - bjd))
		if bjd < thimes[close[0]]:
			after = close[0]
			before = close[1]
		else:
			after = close[1]
			before = close[0]
		
		## Fractional distance between the two closest ThAr exposures
		frac = (bjd - thimes[before]) / (thimes[after] - thimes[before])
		## The RV difference between the two closest ThAr exposures
		dv = (tharvs[after] - tharvs[before])*frac
		dvs = np.append(dvs,dv)

	return dvs

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
