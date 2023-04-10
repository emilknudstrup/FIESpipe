#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Different workfloes that stitch the various modules in :py:mod:`FIESpipe` together to obtain the parameters of interest.

"""

from .extract import *
from .derive import *

def standard(filename,
	ins='FIES',
	orders=[],
	exclude=[],
	rvsys=None,
	target=None,
	gauss=True,
	rvr=41,	#RV range
	chi2rvr=20, #RV range for chi2 minimization
	crmit=False, #cosmic ray mitigation
	out=True,sigma=3, #RV outlier rejection
	CaIIH=3968.47,
	CaIIK=3933.664,
	width=1.09,
	Rsection_low=3991.07,
	Rsection_high=4011.07,
	Vsection_low=3891.07,
	Vsection_high=3901.07,



	):
	'''Standard workflow for FIES data extraction.

	Should work regardless of how the data has been acquired.
	
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
		For now, the template is hardcoded. This should be fixed in the future.
		Oversampled CCFs are hardcoded to be 0.25 km/s.
		RV :math:`\chi^2` minimization is done by fitting a gaussian. Change to a second order polynomial fit as in :cite:t:`Zechmeister2018`?

	:param filename: Name of the file to be processed.
	:type filename: str
	:param ins: Instrument used to acquire the data. Currently only FIES is supported.
	:type ins: str
	:param orders: List of orders to be processed. If empty, all orders are processed.
	:type orders: list
	:param exclude: List of orders to be excluded from the processing.
	:type exclude: list
	:param rvsys: Systemic RV in km/s of the target. If not provided, it is assumed to be 0 km/s.
	:type rvsys: float
	:param target: Name of the target. Used to find systemic velocity.
	:type target: str
	:param gauss: If ``True``, the :math:`chi^2` RVs are fitted with a gaussian. If ``False``, a second order polynomial SHOULD be implemented.
	:type gauss: bool
	:param rvr: RV range in km/s for the CCFs.
	:type rvr: float
	:param chi2rvr: RV range in km/s for the :math:`\chi^2` minimization.
	:type chi2rvr: float
	:param crmit: Cosmic ray mitigation. If ``False``, no mitigation is performed. If ``1``, the median is used. If ``2``, the mean is used.
	:type crmit: bool
	:param out: Outlier rejection for :math:`\chi^2` RVs. If ``True``, the :math:`\chi^2` RVs are rejected if they are more than ``sigma`` away from the median.
	:type out: bool
	:param sigma: Number of sigma for the outlier rejection.
	:type sigma: float
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
	
	:return: Dictionary with the results of the processing.
	:rtype: dict
	
	'''
	## Storage
	data = {}


	if ins == 'FIES':
		R=67000
		s=2.1
		wave, flux, err, h = extractFIES(filename)
	
		## Get the barycentric correction AND the BJD in TDB
		bjd, bvc = getBarycorrs(filename,rvsys=rvsys)# Fix this, should not depend on instrument
		data['BJD_TDB'] = bjd
		data['BVC'] = bvc

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
	## Hardcoded for now -- fix!
	tpath = '/home/au339813/Desktop/PhD/scripts/templates/'
	temp = '4750_30_p02p00.ms.fits'
	twl, tfl = readKurucz(tpath+temp)
	
	## If no systemic velocity is provided, assume 0 km/s
	if rvsys is None:
		## do something with target name
		rvsys = 0.0
		## else do a preliminary RV measurement
		## to shift template
	## Shift the template to center the grid on 0 km/s
	twl *= (1.0 + (rvsys+bvc)*1e3/const.c.value)

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
		wl, nfl = normalize(rwl,rfl)
		nfle = rel_err*nfl
		## Mitigate cosmic rays
		if crmit:
			wl, nfl, nfle = crm(wl,nfl,nfle)
	
		## Chi2 minimization for RVs
		chi2s = chi2RV(drvs,wl,nfl,nfle,twl,tfl)
		## By fitting a gaussian ...for now
		if gauss:
			pars, cov = fitGauss(drvs,chi2s)
			perr = np.sqrt(np.diag(cov))
			data[key]['RV'] = pars[1]+rvsys
			data[key]['eRV'] = perr[1]
			print('Gaussian fit order {}: RV = {:.2f} +/- {:.2f}'.format(ii,pars[1]+rvsys,perr[1]))

		## Resample the observations and template
		lam, resamp_fl, resamp_tfl, resamp_fle = resample(wl,nfl,nfle,twl,tfl,dv=dv)
		rvs, ccf, errs = getCCF(resamp_fl,resamp_tfl,resamp_fle,rvr=arvr,dv=dv)
		
		## Shift RVs to the systemic velocity
		rvs += rvsys
		## Sum CCFs
		ccfs += ccf
		## CCF errors are added in quadrature
		ccferrs += errs**2
		## Store for each order
		arr = np.array([rvs,ccf,errs])
		data[key]['CCF'] = arr

		## Fit CCF
		## if the SNR is low, we can't get a good fit
		try:
			r, er2, fw, efw, c, ec = getRV(rvs,ccf,flipped=1)
		except RuntimeError:
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
		lam, resamp_fl, resamp_tfl, resamp_fle = resample(wl,nfl,nfle,twl,tfl,dv=0.25)
		rvs_ov, ccf_ov, errs_ov = getCCF(resamp_fl,resamp_tfl,resamp_fle,rvr=arvr_ov,dv=0.25)

		rvs_ov += rvsys
		ccfs_ov += ccf_ov
		ccferrs_ov += errs_ov**2

		arr = np.array([rvs_ov,ccf_ov,errs_ov])
		data[key]['CCF_ov'] = arr

		try:
			r, er2, fw, efw, c, ec = getRV(rvs_ov,ccf_ov,flipped=1)
		except RuntimeError:
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

	## Plot CCFs summed over all orders
	# r, er2, fw, efw, c, ec = getRV(rvs,ccfs,ccferr=ccferrs,flipped=1)
	# pars, cov = fitGauss(rvs,ccfs,flipped=1)
	# figccf = plt.figure()
	# axccf = figccf.add_subplot(111)
	# axccf.set_xlabel('Velocity (km/s)')
	# axccf.set_ylabel('CCF')
	# axccf.plot(rvs,ccfs,ls='-',color='k')
	# axccf.errorbar(rvs,ccfs,yerr=ccferrs,fmt='.')
	# axccf.plot(rvs,Gauss(rvs,*pars),ls='--',color='g')
	# axccf.plot([r-fw/2,r+fw/2],[pars[0]*0.5+pars[3],pars[0]*0.5+pars[3]])
	# axccf.plot([r,r],[pars[3],pars[3]+pars[0]])


	## Summed oversampled CCFs
	ccferrs_ov = np.sqrt(ccferrs_ov)
	arr_ov = np.array([rvs_ov,ccfs_ov,ccferrs_ov])
	data['CCFsum_ov'] = arr_ov

	# ## Plot chi2 solution for RV
	# figchi = plt.figure()
	# axchi = figchi.add_subplot(111)
	# axchi.set_xlabel('Velocity (km/s)')
	# axchi.set_ylabel('$\chi^2$')

	rv_all, erv_all = np.array([]), np.array([])
	for order in data['orders']:
		key = 'order_{}'.format(order)
		rv_all = np.append(rv_all,data[key]['RV'])
		erv_all = np.append(erv_all,data[key]['eRV'])
		# axchi.errorbar(order,data[key]['RV'],yerr=data[key]['eRV'])

	## Get weighted mean RVs 
	if out:
		wavg_rv, wavg_err, mu, std  = weightedMean(rv_all,erv_all,out=out,sigma=sigma)
		# axchi.errorbar(np.median(use_orders),wavg_rv,yerr=wavg_err)
		# yy = np.ones(len(xx))*mu
		# axchi.fill_between(xx, yy-std, yy+std, alpha=0.5,color='C7',label=r'${} \sigma \ rejection$'.format(sigma))
		# axchi.axhline(mu,color='C7',ls='--',label=r'$\mu$')
		# axchi.fill_between(xx, wavg_rv-wavg_err, wavg_rv+wavg_err, alpha=0.8,color='C0',label=r'$\sigma(\mathrm{RV}) \ (km/s) $')
		# axchi.axhline(wavg_rv,color='k',ls='--',label=r'$\rm RV \ (km/s$')

		data['order_all'] = {
			'RV': wavg_rv,
			'eRV': wavg_err,
			'mu_out': mu,
			'std_out': std,
		}
	else:
		wavg_rv, wavg_err = weightedMean(rv_all,erv_all,out=out)
		# axchi.errorbar(np.median(use_orders),wavg_rv,yerr=wavg_err)
		# axchi.fill_between(xx, wavg_rv-wavg_err, wavg_rv+wavg_err, alpha=0.8,color='C0',label=r'$\sigma(\mathrm{RV}) \ (km/s) $')
		# axchi.axhline(wavg_rv,color='k',ls='--',label=r'$\rm RV \ (km/s$')

		data['order_all'] = {
			'RV': wavg_rv,
			'eRV': wavg_err
		}

	# y = ccfs# - np.min(ccfs)
	# x = rvs
	# err, der, xerr = getError(rvs, y, ccferrs)
	# bis, biserr = getBIS(x, y, xerr)
	## BIS estimate from oversampled CCF
	err, der, xerr = getError(data['CCFsum_ov'][0,:], data['CCFsum_ov'][1,:], data['CCFsum_ov'][2,:])
	y = data['CCFsum_ov'][1,:]#arr_ov[1,:]#ccfs# - np.min(ccfs)
	x = data['CCFsum_ov'][0,:]#arr_ov[0,:]#rvs
	bis, biserr = getBIS(x, y, xerr)

	data['BIS'] = bis
	data['eBIS'] = biserr

	## Plot CaIIK
	# figHK = plt.figure()
	# axK = figHK.add_subplot(411)
	# axH = figHK.add_subplot(412)
	# axV = figHK.add_subplot(413)
	# axR = figHK.add_subplot(414)
	# axR.set_xlabel('Wavelength (Angstroms)')
	# axes = [axK,axH,axV,axR]
	# for ax in axes:
	# 	ax.set_ylabel('Flux')

	Korders = data['CaIIK']['orders']
	Ks = []
	for order in Korders:
		key = 'order_{}'.format(order)
		arr = data['CaIIK'][key]
		wl = arr[0]
		fl = arr[1]
		# axK.plot(wl,fl)
		# axK.axvline(CaIIK,color='k',ls='--')
		weights = triangle(wl,CaIIK,width*0.5,width*0.5)
		# axK.plot(wl,weights)
		core = weights > 0.0
		# axK.plot(wl[core],weights[core]*fl[core])
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
		# axH.plot(wl,fl)
		# axH.axvline(CaIIH,color='k',ls='--')
		# axH.plot(wl,weights)
		# axH.plot(wl[core],weights[core]*fl[core])

	Vorders = data['Vcontinuum']['orders']
	Vconts = []
	for order in Vorders:
		key = 'order_{}'.format(order)
		arr = data['Vcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		Vcont = np.median(fl[(wl > Vsection_low) & (wl < Vsection_high)])
		Vconts.append(Vcont)
		# axV.plot(wl,fl)
		# axV.fill_betweenx([min(fl),max(fl)],[Vsection_low,Vsection_low],[Vsection_high,Vsection_high],color='C1',alpha=0.2)
		# axV.axvline(Vsection_low,color='k',ls='--')
		# axV.axvline(Vsection_high,color='k',ls='--')

	Rorders = data['Rcontinuum']['orders']
	Rconts = []
	for order in Rorders:
		key = 'order_{}'.format(order)
		arr = data['Rcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		Rcont = np.median(fl[(wl > Rsection_low) & (wl < Rsection_high)])
		Rconts.append(Rcont)
		# axR.plot(wl,fl)
		# axR.fill_betweenx([min(fl),max(fl)],[Rsection_low,Rsection_low],[Rsection_high,Rsection_high],color='C3',alpha=0.2)
		# axR.axvline(Rsection_low,color='k',ls='--')
		# axR.axvline(Rsection_high,color='k',ls='--')

	S = (np.mean(Ks) + np.mean(Hs))/(np.mean(Vconts) + np.mean(Rconts))
	data['S_uncalibrated'] = S

	return data