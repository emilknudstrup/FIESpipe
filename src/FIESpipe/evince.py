#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routines to plot the various data products.

"""
import matplotlib.pyplot as plt
import numpy as np
from .derive import Gauss, triangle

def Spectrum(data):
	'''Spectrum.

	Plot the normalized (:py:func:`FIESpipe.derive.normalize` and perhaps :py:func:`FIESpipe.derive.crm` filtered) spectrum for each order.

	:param data: Dictionary containing the data.
	:type data: dict

	'''
	## Plot spectrum
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('Wavelength (Angstroms)')
	ax.set_ylabel('Flux')
	use_orders = data['orders']
	ii = 0
	for order in use_orders:
		key = 'order_{}'.format(order)
		wl, fl, efl = data[key]['spectrum'][0,:], data[key]['spectrum'][1,:], data[key]['spectrum'][2,:]
		ax.errorbar(wl,fl,yerr=efl)
		try:
			wlo, flo, eflo = data[key]['outliers'][0,:], data[key]['outliers'][1,:], data[key]['outliers'][2,:]
			if not ii: ax.plot(wlo,flo,'x',color='C3',label='Omitted outliers')
			else: ax.plot(wlo,flo,'x',color='C3')
			ii += 1
		except KeyError:
			pass
	if ii: ax.legend(loc='best')

def RVsols(data):
	'''RV :math:`\chi^2` solutions.

	Plot the RV solutions for each order.
	
	:param data: Dictionary containing the data.
	:type data: dict

	'''
	
	## Plot chi2 solution for RV
	figchi = plt.figure()
	axchi = figchi.add_subplot(111)
	axchi.set_xlabel('Velocity (km/s)')
	axchi.set_ylabel('$\chi^2$')
	wavg_rv = data['order_all']['RV']
	wavg_err = data['order_all']['eRV']
	use_orders = data['orders']
	for order in use_orders:
		key = 'order_{}'.format(order)
		axchi.errorbar(order,data[key]['RV'],yerr=data[key]['eRV'])
	axchi.errorbar(np.median(use_orders),wavg_rv,yerr=wavg_err)
	axchi.axhline(wavg_rv,color='k',ls='--',label=r'$\rm RV \ (km/s$')
	xx = np.linspace(np.min(use_orders),np.max(use_orders),100)
	axchi.fill_between(xx, wavg_rv-wavg_err, wavg_rv+wavg_err, alpha=0.8,color='C0',label=r'$\sigma(\mathrm{RV}) \ (km/s) $')

	try:
		mu = data['order_all']['mu_out']
		std = data['order_all']['std_out']
		yy = np.ones(len(xx))*mu
		axchi.fill_between(xx, yy-std, yy+std, alpha=0.5,color='C7',label=r'$\sigma \ rejection$')
		axchi.axhline(mu,color='C7',ls='--',label=r'$\mu$')
	except KeyError:
		pass

	axchi.legend(loc='best')


def plotCCFs(data):
	'''Plot the CCFs.

	Plot the CCFs for each order.
	
	:param data: Dictionary containing the data.
	:type data: dict

	'''
	## Plot CCFs
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('Velocity (km/s)')
	ax.set_ylabel('CCF')

	use_orders = data['orders']
	ii = 0
	for order in use_orders:
		key = 'order_{}'.format(order)
		arr = data[key]['CCF']
		rv = arr[0,:]
		ccf = arr[1,:]
		err = arr[2,:]
		ax.errorbar(rv,ccf,yerr=err)

def sumCCFit(data):
	'''Gaussian fit for summed CCFs.

	Plot the Gaussian fit for the summed CCFs, showing the FWHM and contrast.

	:param data: Dictionary containing the data.
	:type data: dict

	'''
	fw = data['FWHMsum']
	#efw = data['eFWHMsum']
	#c = data['Contrastsum']
	#ec = data['eContrastsum']
	r = data['RVsum']
	#er2 = data['eRVsum']
	pars = data['fitGausssum'][0]
	#cov = data['fitGausssum'][1]

	rvs = data['CCFsum'][0,:]
	ccfs = data['CCFsum'][1,:]
	ccferrs = data['CCFsum'][2,:]

	## Plot CCFs summed over all orders
	figccf = plt.figure()
	axccf = figccf.add_subplot(111)
	axccf.set_xlabel('Velocity (km/s)')
	axccf.set_ylabel('CCF')
	axccf.plot(rvs,ccfs,ls='-',color='C0')
	axccf.errorbar(rvs,ccfs,yerr=ccferrs,fmt='.',label='CCF',color='C0')
	axccf.plot(rvs,Gauss(rvs,*pars),ls='--',color='C2',label='Gaussian fit')
	axccf.plot([r-fw/2,r+fw/2],[pars[0]*0.5+pars[3],pars[0]*0.5+pars[3]],label='FWHM',color='C3')
	axccf.plot([r,r],[pars[3],pars[3]+pars[0]],label='Contrast',color='C1')
	
	axccf.legend(loc='best')

def Sindex(data,
	CaIIH=3968.47,
	CaIIK=3933.664,
	width=1.09,
	Rsection_low=3991.07,
	Rsection_high=4011.07,
	Vsection_low=3891.07,
	Vsection_high=3911.07,
	):
	'''S-index.
	
	Plot the CaII H and K cores along with the red and blue continua.

	:param data: Dictionary containing the data.
	:type data: dict
	:param CaIIH: Wavelength of the CaII H line. Default is 3968.47 Angstroms.
	:type CaIIH: float
	:param CaIIK: Wavelength of the CaII K line. Default is 3933.664 Angstroms.
	:type CaIIK: float
	:param width: Width of the CaII H and K cores. Default is 1.09 Angstroms.
	:type width: float
	:param Rsection_low: Lower limit of the red continuum. Default is 3991.07 Angstroms.
	:type Rsection_low: float
	:param Rsection_high: Upper limit of the red continuum. Default is 4011.07 Angstroms.
	:type Rsection_high: float
	:param Vsection_low: Lower limit of the blue continuum. Default is 3891.07 Angstroms.
	:type Vsection_low: float
	:param Vsection_high: Upper limit of the blue continuum. Default is 3911.07 Angstroms.
	:type Vsection_high: float
	

	'''

	figHK = plt.figure()
	axK = figHK.add_subplot(411)
	axH = figHK.add_subplot(412)
	axV = figHK.add_subplot(413)
	axR = figHK.add_subplot(414)
	axR.set_xlabel('Wavelength (Angstroms)')
	axes = [axK,axH,axV,axR]
	for ax in axes:
		ax.set_ylabel('Flux')

	Korders = data['CaIIK']['orders']
	for ii, order in enumerate(Korders):
		key = 'order_{}'.format(order)
		arr = data['CaIIK'][key]
		wl = arr[0]
		fl = arr[1]
		axK.plot(wl,fl)
		axK.axvline(CaIIK,color='k',ls='--')
		weights = triangle(wl,CaIIK,width*0.5,width*0.5)
		axK.plot(wl,weights,color='C7')
		core = weights > 0.0
		axK.plot(wl[core],weights[core]*fl[core],color='k',label='K core')
		if not ii:
			axK.legend(loc='best')

	Horders = data['CaIIH']['orders']
	for ii, order in enumerate(Horders):
		key = 'order_{}'.format(order)
		arr = data['CaIIH'][key]
		wl = arr[0]
		fl = arr[1]
		axH.plot(wl,fl)
		axH.axvline(CaIIH,color='k',ls='--')
		weights = triangle(wl,CaIIH,width*0.5,width*0.5)
		axH.plot(wl,weights,color='C7')
		core = weights > 0.0
		axH.plot(wl[core],weights[core]*fl[core],color='k',label='H core')
		if not ii:
			axH.legend(loc='best')

	Vorders = data['Vcontinuum']['orders']
	for ii, order in enumerate(Vorders):
		key = 'order_{}'.format(order)
		arr = data['Vcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		axV.plot(wl,fl)
		axV.fill_betweenx([min(fl),max(fl)],[Vsection_low,Vsection_low],[Vsection_high,Vsection_high],color='C1',alpha=0.2,label='V continuum')
		axV.axvline(Vsection_low,color='k',ls='--')
		axV.axvline(Vsection_high,color='k',ls='--')
		if not ii:
			axV.legend(loc='best')

	Rorders = data['Rcontinuum']['orders']
	for ii, order in enumerate(Rorders):
		key = 'order_{}'.format(order)
		arr = data['Rcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		axR.plot(wl,fl)
		axR.fill_betweenx([min(fl),max(fl)],[Rsection_low,Rsection_low],[Rsection_high,Rsection_high],color='C3',alpha=0.2,label='R continuum')
		axR.axvline(Rsection_low,color='k',ls='--')
		axR.axvline(Rsection_high,color='k',ls='--')
		if not ii:
			axR.legend(loc='best')

def plotBIS(data,
	bybotmin_percent=10.,
	bybotmax_percent=40.,
	bytopmin_percent=60.,
	bytopmax_percent=90.,
	):
	'''BIS and the CCF.

	Plot the slope of the bisector and the CCF.
	See :py:func:`FIESpipe.derive.getBIS` for more information.

	:param data: Dictionary containing the data.
	:type data: dict
	:param bybotmin_percent: Lower limit of the bottom region of the BIS. Default is 10%.
	:type bybotmin_percent: float
	:param bybotmax_percent: Upper limit of the bottom region of the BIS. Default is 40%.
	:type bybotmax_percent: float
	:param bytopmin_percent: Lower limit of the top region of the BIS. Default is 60%.
	:type bytopmin_percent: float
	:param bytopmax_percent: Upper limit of the top region of the BIS. Default is 90%.
	:type bytopmax_percent: float

	'''


	bx = data['BISx']
	by = data['BISy']
	x = data['CCFsum_ov'][0,:]
	y = data['CCFsum_ov'][1,:]
	xerr = data['CCFsum_ov_xerr']

	## Estimate top and bottom regions of the BIS
	by_min = np.min(by)
	by_max = np.max(by)
	dy = by_max - by_min

	bybotmin = by_min + dy*bybotmin_percent/100.
	bybotmax = by_min + dy*bybotmax_percent/100.
	bytopmin = by_min + dy*bytopmin_percent/100.
	bytopmax = by_min + dy*bytopmax_percent/100.
	
	## Plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('Velocity (km/s)')
	ax.set_ylabel('CCF')
	ax.plot(bx,by,color='k',label='BIS')
	ax.errorbar(x,y,xerr=xerr,fmt='o',color='C0',label='CCF')

	ax.axhline(bybotmin,color='k',ls='--')
	ax.axhline(bybotmax,color='k',ls='--')
	ax.fill_betweenx([bybotmin,bybotmax],[min(x),min(x)],[max(x),max(x)],color='C1',alpha=0.2,label='BIS bottom')
	ax.axhline(bytopmin,color='k',ls='--')
	ax.axhline(bytopmax,color='k',ls='--')
	ax.fill_betweenx([bytopmin,bytopmax],[min(x),min(x)],[max(x),max(x)],color='C3',alpha=0.2,label='BIS top')

	ax.legend(loc='best')

# =============================================================================
# "SERVAL" data products
# =============================================================================


def plotCoaddTemplate(data,
				    orders=[45,46,47,48,49],
					ignore=False):
	'''Plot the coadded template.

	Plot the coadded template for the specified orders as well as the residuals/outliers.
	
	:param data: Dictionary containing the data. From :py:func:`FIESpipe.workflows.servalish`.
	:type data: dict
	:param orders: List of orders to plot. Default is [45,46,47,48,49].
	:type orders: list
	:param ignore: Ignore the warning about the number of orders to plot. Default is False.
	:type ignore: bool

	'''
	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax2.set_xlabel('Wavelength (Angstrom)')
	ax1.set_ylabel('Flux')
	ax2.set_ylabel('Residuals')

	if not ignore:
		assert len(orders) < 10, 'Number of orders should be less than 10. Otherwise it will be hard to assess the results. Set ignore=True to ignore this warning.'

	## Loop over the specified orders
	for ii, order in enumerate(orders):
		key = 'order_{}'.format(order)
		files = data[key]['files'].keys()
		for file in files:
			warr = data[key]['files'][file]
			ax1.errorbar(warr[0],warr[1],yerr=warr[2],marker='o',alpha=0.25)


		arr = data[key]['spectrum']
		wl, fl, efl = arr[0], arr[1], arr[2]

		## Plot the spline
		knots, spline = servalData[key]['template']
		## Only give the label for the first order
		if ii:
			ax1.plot(knots,spline(knots),color='k',zorder=15,lw=2.0)
		else:
			ax1.plot(knots,spline(knots),color='k',zorder=15,label='spline',lw=2.0)


		## Plot the residuals
		## and outliers
		outer = servalData[key]['rejection']['iterations']
		for jj in range(outer):
			yhat, res, swl, mu, sigma = servalData[key]['rejection']['iteration_{}'.format(jj)]['filter']
			ax2.plot(swl,res,alpha=0.25,color='C7')
			owl, ofl, ofle = servalData[key]['rejection']['iteration_{}'.format(jj)]['outliers']
			if ii or jj:
				ax1.plot(owl,ofl,marker='x',color='C3',ls='none')
				ax1.plot(swl,yhat,color='k',lw=2.0,zorder=10)
				ax1.plot(swl,yhat,color='C7',lw=1.0,zorder=11)
				ax2.plot(swl,mu*np.ones_like(swl),color='C3')				
			else:
				ax1.plot(owl,ofl,marker='x',color='C3',label='outlier',ls='none')
				ax1.plot(swl,yhat,color='k',lw=2.0,zorder=10)
				ax1.plot(swl,yhat,color='C7',label='savgol filter',lw=1.0,zorder=11)
				ax2.plot(swl,mu*np.ones_like(swl),color='C3',label='mean')

			## Plot the sigma regions
			ax2.fill_between([min(swl),max(swl)],mu+sigma,mu-sigma,alpha=0.1,color='C{}'.format(jj+outer))


	ax1.legend()
	ax2.legend()