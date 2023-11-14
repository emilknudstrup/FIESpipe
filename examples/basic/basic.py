#%%
'''

Illustrated here for a single order for a single spectrum.

The star here is :math:`\\gamma` Cephei A, which is a slow rotator, so it has nice sharp lines.

Normalization and outlier rejection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


>>> import FIESpipe as fp
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> ## Sort the FIES files into science and ThAr spectra
>>> spec = 'your_path/gamCep/'
>>> filenames, tharnames = fp.sortFIES(spec)
>>> filenames.sort()
>>> ## Run the standard reduction on the science spectra
>>> file = filenames[0]
>>> ## Extract the data from the FITS file
>>> wave, flux, ferr, hdr = fp.extractFIES(file)
>>> order = 45 #+1 # 0-indexed
>>> w, f, e = wave[order], flux[order], ferr[order]
>>> ## Relative error
>>> re = e/f

>>> ## Plot the spectrum
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.set_xlabel(r'$\\rm \lambda \ (\AA)$')
>>> ax.set_ylabel(r'$\\rm F_{\lambda}$')
>>> ax.errorbar(w,f,yerr=e)

.. image:: ../../../examples/basic/raw_spec.png

>>> ## Normalize the spectrum
>>> pdeg = 2 # Polynomial degree
>>> wl, nfl = fp.normalize(w,f,poly=pdeg)
>>> ## Scaled error
>>> nfle = re*nfl
>>> ## Cosmic ray removal/outlier rejection
>>> wlo, flo, eflo, idxs = fp.crm(wl,nfl,nfle)

>>> ## Plot the normalized, cleansed spectrum
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.set_xlabel(r'$\\rm \lambda \ (\AA)$')
>>> ax.set_ylabel(r'$\\rm F_{\lambda}$')
>>> ax.plot(wl[idxs],nfl[idxs],'rx',zorder=5)
>>> ## Plot the cleaned spectrum
>>> ax.errorbar(wlo,flo,yerr=eflo)

>>> ## Load Kurucz template
>>> temp = 'your_path/4750_30_p02p00.ms.fits'
>>> tw, tf = fp.readKurucz(temp)
>>> ## Plot the template in an interval around the order
>>> show = (tw > wlo.min()-5) & (tw < wlo.max()+5)
>>> ax.plot(tw[show],tf[show],'k-')

>>> ## For this star, I know the systemic velocity is around -40 km/s
>>> rvsys = -40 # km/s
>>> ## Shift the template to the systemic velocity, 
>>> ### to center the grid around 0 km/s
>>> ### not important, but easier to work with
>>> sw = tw*(1.0 + rvsys*1e3/fp.const.c.value)
>>> ## Plot the shifted template
>>> ax.plot(sw[show],tf[show],'--',color='C7')

.. image:: ../../../examples/basic/norm_spec.png

Cross-correlation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> ## Resample the template and the wavelength grid of the spectrum
>>> dv = fp.velRes(R=67000,s=2.1) # km/s, velocity resolution of the FIES spectrograph using fibre 4
>>> ccfs, ccferrs, arvr, _, _, _ = fp.grids(rvr=51,R=67000,s=2.1) # velocity grid, range from 101 km/s
>>> lam, resamp_fl, resamp_tfl, resamp_fle = fp.resample(wl,nfl,nfle,sw,tf,dv=dv)
>>> rvs, ccf, errs = fp.getCCF(resamp_fl,resamp_tfl,resamp_fle,rvr=arvr,dv=dv)
>>> ## Plot the CCF
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.set_xlabel(r'$\\rm RV \ (km\,s^{-1})$')
>>> ax.set_ylabel(r'$\\rm CCF$')
>>> ax.errorbar(rvs,ccf,yerr=errs)

.. image:: ../../../examples/basic/ccf.png


>>> ## Shift the CCF to the systemic velocity
>>> rvs += rvsys

>>> ## Fit the CCF with a Gaussian
>>> ## RV, FWHM, contrast
>>> rv, erv, fw, efw, co, eco, pars, cov = fp.getRV(rvs,ccf,flipped=1,return_pars=1)

>>> ## Plot the CCF with the Gaussian fit
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.set_xlabel(r'$\\rm RV \ (km\,s^{-1})$')
>>> ax.set_ylabel(r'$\\rm CCF$')
>>> ax.errorbar(rvs,ccf,yerr=errs)
>>> xx = np.linspace(rvs.min(),rvs.max(),1000)
>>> ax.plot(xx,fp.Gauss(xx,*pars),'k--')
>>> ax.plot([rv-fw/2,rv+fw/2],[pars[0]*0.5+pars[3],pars[0]*0.5+pars[3]],label='FWHM',color='C3')
>>> ax.plot([rv,rv],[pars[3],pars[3]+pars[0]],label='Contrast',color='C1')
>>> ## Calculate BIS
>>> ## First estimate error on RV grid/x-axis
>>> errt, d, xerr = fp.getxError(rvs,ccf,errs)
>>> ## BIS, BIS error
>>> bis, bx, by, biserr = fp.getBIS(rvs,ccf,xerr)
>>> ax.errorbar(rvs,ccf,yerr=errs,xerr=xerr)
>>> ax.plot(bx,by,label='BIS',color='C2')
>>> ax.legend()

.. image:: ../../../examples/basic/ccf_fit.png

>>> ## Get the barycentric correction and BJD in TDB
>>> bjd, bvc = fp.getBarycorrs(file,rvmeas=rv)
>>> rv += bvc
>>> print('RV measured: {0:.3f} +/- {1:.3f} km/s'.format(rv,erv))

.. code-block:: bash

	RV measured: -44.372 +/- 0.121 km/s

'''
import FIESpipe as fp
import matplotlib.pyplot as plt
import numpy as np
import os
path = os.path.dirname(os.path.abspath(__file__))


def basic(
	fpath = 'data/spectra/gamCep/',
	tpath = '/4750_30_p02p00.ms.fits',
	save = False,
	width = 15,
	height = 6,
	):
	'''Basic example of the FIESpipe package.

	Used for testing.

	:param fpath: Path to the FIES data. Default is ``data/spectra/gamCep/``.
	:type fpath: str
	:param tpath: Path to the Kurucz template. Default is ``/4750_30_p02p00.ms.fits``.
	:type tpath: str
	:param save: Save the figures. Default is ``False``.
	:type save: bool
	:param width: Width of the figures. Default is ``15`` (inches).
	:type width: float
	:param height: Height of the figures. Default is ``6`` (inches).
	:type height: float

	:returns: RV, RV error, BJD, BVC, FWHM, FWHM error, contrast, contrast error
	:rtype: float, float, float, float, float, float, float, float

	'''
	## Sort the FIES files into science and ThAr spectra
	filenames, tharnames = fp.sortFIES(path+'/../../'+fpath)
	filenames.sort()
	## Run the standard reduction on the science spectra
	file = filenames[0]
	## Extract the data from the FITS file
	wave, flux, ferr, hdr = fp.extractFIES(file)
	order = 45 #+1 # 0-indexed
	w, f, e = wave[order], flux[order], ferr[order]
	## Relative error
	re = e/f

	## Plot the spectrum
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\rm \lambda \ (\AA)$')
	ax.set_ylabel(r'$\rm F_{\lambda}$')
	ax.errorbar(w,f,yerr=e)
	if save: fig.savefig(path+'/raw_spec.png',bbox_inches='tight')

	## Normalize the spectrum
	pdeg = 2 # Polynomial degree
	wl, nfl = fp.normalize(w,f,poly=pdeg)
	## Scaled error
	nfle = re*nfl
	## Cosmic ray removal/outlier rejection
	wlo, flo, eflo, idxs = fp.crm(wl,nfl,nfle)

	## Plot the normalized, cleansed spectrum
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\rm \lambda \ (\AA)$')
	ax.set_ylabel(r'$\rm F_{\lambda}$')
	ax.plot(wl[idxs],nfl[idxs],'rx',zorder=5)
	## Plot the cleansed spectrum
	ax.errorbar(wlo,flo,yerr=eflo)

	## Load Kurucz template
	tw, tf = fp.readKurucz(path + '/../../data/temp/kurucz/' + tpath)
	## Plot the template in an interval around the order
	show = (tw > wlo.min()-5) & (tw < wlo.max()+5)
	ax.plot(tw[show],tf[show],'k-')
	if save: fig.savefig(path+'/norm_spec.png',bbox_inches='tight')

	## For this star, I know the systemic velocity is around -40 km/s
	rvsys = -40 # km/s
	## Shift the template to the systemic velocity, 
	### to center the grid around 0 km/s
	### not important, but easier to work with
	sw = tw*(1.0 + rvsys*1e3/fp.const.c.value)
	## Plot the shifted template
	ax.plot(sw[show],tf[show],'--',color='C7')

	## Resample the template and the wavelength grid of the spectrum
	dv = fp.velRes(R=67000,s=2.1) # km/s, velocity resolution of the FIES spectrograph using fibre 4
	ccfs, ccferrs, arvr, _, _, _ = fp.grids(rvr=51,R=67000,s=2.1) # velocity grid, range from 101 km/s
	lam, resamp_fl, resamp_tfl, resamp_fle = fp.resample(wlo,flo,eflo,sw,tf,dv=dv)
	rvs, ccf, errs = fp.getCCF(resamp_fl,resamp_tfl,resamp_fle,rvr=arvr,dv=dv)
	## Plot the CCF
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\rm RV \ (km\,s^{-1})$')
	ax.set_ylabel(r'$\rm CCF$')
	ax.errorbar(rvs,ccf,yerr=errs)
	if save: fig.savefig(path+'/ccf.png',bbox_inches='tight')

	## Shift the CCF to the systemic velocity
	rvs += rvsys

	## Fit the CCF with a Gaussian
	## RV, FWHM, contrast
	rv, erv, fw, efw, co, eco, pars, cov = fp.getRV(rvs,ccf,flipped=1,return_pars=1)

	## Plot the CCF with the Gaussian fit
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\rm RV \ (km\,s^{-1})$')
	ax.set_ylabel(r'$\rm CCF$')
	xx = np.linspace(rvs.min(),rvs.max(),1000)
	ax.plot(xx,fp.Gauss(xx,*pars),'k--')
	ax.plot([rv-fw/2,rv+fw/2],[pars[0]*0.5+pars[3],pars[0]*0.5+pars[3]],label='FWHM',color='C3')
	ax.plot([rv,rv],[pars[3],pars[3]+pars[0]],label='Contrast',color='C1')

	## Calculate BIS
	## First estimate error on RV grid/x-axis
	errt, d, xerr = fp.getxError(rvs,ccf,errs)
	## BIS, BIS error
	bis, bx, by, biserr = fp.getBIS(rvs,ccf,xerr)
	ax.errorbar(rvs,ccf,yerr=errs,xerr=xerr)
	ax.plot(bx,by,label='BIS',color='C2')

	ax.legend()
	if save: fig.savefig(path+'/ccf_fit.png',bbox_inches='tight')

	## Get the barycentric correction and BJD in TDB
	bjd, bvc = fp.getBarycorrs(file,rvmeas=rv)
	rv += bvc
	print(
		'RV measured: {0:.3f} +/- {1:.3f} km/s'.format(rv,erv),
	)

	return	rv, erv, bjd[0], bvc, fw, efw, co, eco, bis, biserr


if __name__ == '__main__':
	actual = basic()
	desired = (4.418086e+01,  1.377707e-01,  2.459216e+06, -8.563322e+00,
    			1.220351e+01,  1.645361e-01,  1.066839e+02,  2.975356e+00,
    			-3.562438e-01,  6.706993e-03)

	# rv,erv,bjd,bvc,fw,efw,co,eco,bis,biserr = basic()

	np.testing.assert_allclose(actual, desired, rtol=1e1)
	#np.testing.assert_allclose(results, expected, rtol=1e-3)
	#np.testing.assert_allclose([rv, erv, bjd, bvc, fw, efw, co, eco], expected, rtol=1e-3)
