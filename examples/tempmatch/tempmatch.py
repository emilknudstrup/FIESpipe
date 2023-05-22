'''
>>> import FIESpipe as fp
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import glob

>>> ## Set the path to the data and template
>>> path = 'your_path/TOI-2158/'
>>> ## Group the spectra by epoch
>>> epochs = fp.groupByEpochs(path)
>>> ## We'll also create a list of the filenames
>>> ## for the initial preparation
>>> filenames = glob.glob(path+'*wave.fits')
>>> ## Load Kurucz template
>>> tpath = '/4750_30_p02p00.ms.fits'
>>> tw, tf = fp.readKurucz(path + 'kurucz/' + tpath)

>>> ## The orders used can influence the final RV
>>> ## and the error quite a bit
>>> ## We'll start out by using some of the 
>>> ## (typically) well-behaved orders (~5000-6000 AA)
>>> norders = range(40,60)

>>> ## Dictionary to store the prepped data
>>> ## normalized spectra/initial RVs
>>> prepped = fp.prepareSpectra(filenames,norders,tw,tf)

>>> ## Now we'll use all these prepped spectra
>>> ## to create the first iteration of splines
>>> splines = fp.makeSplines(prepped)

>>> ## Now we'll use the splines as a template
>>> ## to get the RVs
>>> ## The procedure is very similar to the one above

>>> ## Dictionary to store the results
>>> wrvs = fp.matchRVs(prepped,splines)

>>> ## We'll now use the newly derived RVs
>>> ## in the second iteration
>>> for file in wrvs:
>>> 	bjd, rv, erv, bvc = wrvs[file]
>>> 	prepped[file]['rv'] = rv
>>> 	prepped[file]['erv'] = erv
>>> ## and we'll make a new batch of splines

>>> ## New dictionary to store the splines
>>> second = fp.makeSplines(prepped)

>>> ## We could now use the splines to get the RVs for each individual spectrum again,
>>> ## but this time we'll coadd the spectra from different epochs first
>>> ## We'll store the weighted average RV and error
>>> ## for each epoch in arrays
>>> final_rvs = np.array([])
>>> final_ervs = np.array([])
>>> ## As well as the time and barycentric correction
>>> final_bjds = np.array([])
>>> final_bvcs = np.array([])
>>> for epoch in epochs:
>>> 	filenames = epochs[epoch]['names']
>>> 	## Fine grid centered on the measrued RV
>>> 	rv = np.array([])
>>> 	for file in filenames:
>>> 		rv = np.append(rv,prepped[file]['rv'])
>>> 	rvepoch = np.median(rv)
>>> 	drvs = np.arange(rvepoch - 10.0, rvepoch + 10.0, 0.1)
>>> 	## Coadd the spectra
>>> 	coadd = fp.coaddSpectra(filenames,prepped)
>>> 	## Get the RVs
>>> 	wavg_rv, wavg_err = fp.splineRVs(coadd,second,drvs)
>>> 	## Get the barycentric correction
>>> 	## and the BJD in TDB
>>> 	bjds = np.array([])
>>> 	bvcs = np.array([])
>>> 	for file in filenames:
>>> 		bjd, bvc = fp.getBarycorrs(file,wavg_rv)
>>> 		bjds = np.append(bjds,bjd)
>>> 		bvcs = np.append(bvcs,bvc)
>>> 	## ... this could be done in a more sophisticated way
>>> 	## by for example using the flux weighted midpoint of the exposure
>>> 	## but this is good enough for now
>>>	## Append to arrays
>>> 	final_bjds = np.append(final_bjds,np.mean(bjds))
>>> 	final_bvcs = np.append(final_bvcs,np.mean(bvcs))
>>> 	final_rvs = np.append(final_rvs,wavg_rv+np.mean(bvcs))
>>> 	final_ervs = np.append(final_ervs,wavg_err)

>>> for ii, bjd in enumerate(final_bjds):
>>> 	print('BJD: {:.5f}, RV: {:.3f} +/- {:.3f} m/s'.format(bjd,final_rvs[ii]*1e3,final_ervs[ii]*1e3))

.. code-block:: bash

	BJD: 2459333.70360, RV: -108121.328 +/- 10.511 m/s
	BJD: 2459339.68180, RV: -108163.728 +/- 5.878 m/s
	BJD: 2459351.64286, RV: -107978.377 +/- 7.666 m/s
	BJD: 2459355.66056, RV: -108132.583 +/- 7.999 m/s
	BJD: 2459364.63371, RV: -108129.309 +/- 5.484 m/s
	BJD: 2459365.60471, RV: -108107.542 +/- 4.827 m/s
	BJD: 2459367.51570, RV: -108023.135 +/- 8.147 m/s
	BJD: 2459367.64103, RV: -108033.972 +/- 5.167 m/s
	BJD: 2459368.63520, RV: -107984.135 +/- 6.939 m/s
	BJD: 2459369.68646, RV: -108008.666 +/- 8.646 m/s
	BJD: 2459371.62623, RV: -108043.780 +/- 3.844 m/s
	BJD: 2459372.49619, RV: -108087.778 +/- 8.811 m/s
	BJD: 2459376.59300, RV: -108037.943 +/- 9.436 m/s
	BJD: 2459380.64168, RV: -108088.978 +/- 8.693 m/s

>>> ## Plot the RVs
>>> fig = plt.figure(figsize=(width,height))
>>> ax = fig.add_subplot(111)
>>> ax.errorbar(final_bjds,final_rvs,yerr=final_ervs,fmt='o',color='k')	
>>> ax.set_xlabel(r'$\\mathrm{BJD}_\\mathrm{TDB}$')
>>> ax.set_ylabel(r'RV (km/s)')

.. image:: ../../../examples/tempmatch/rvs_func.png

'''
#%%

import FIESpipe as fp
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
path = os.path.dirname(os.path.abspath(__file__))

def tempmatch(
	fpath = 'data/spectra/TOI-2158/',
	tpath = '/4750_30_p02p00.ms.fits',
	save = False,
	width = 15,
	height = 6,	        
	):
	'''Example of the FIESpipe package for template matching.

	Used for testing.
	

	:param fpath: Path to the FIES data. Default is ``data/spectra/TOI-2158/``.
	:type fpath: str
	:param tpath: Path to the Kurucz template. Default is ``/4750_30_p02p00.ms.fits``.
	:type tpath: str
	:param save: Save the figures. Default is ``False``.
	:type save: bool
	:param width: Width of the figures. Default is ``15`` (inches).
	:type width: float
	:param height: Height of the figures. Default is ``6`` (inches).
	:type height: float

	:returns: BJD, RVs, RV errors, BVC
	:rtype: array, array, array, array
	
	'''
	## Group the spectra by epoch
	epochs = fp.groupByEpochs(path+'/../../'+fpath)
	## We'll also create a list of the filenames
	## for the initial preparation
	filenames = glob.glob(path+'/../../'+fpath+'*wave.fits')
	## Load Kurucz template
	tw, tf = fp.readKurucz(path + '/../../data/temp/kurucz/' + tpath)
	
	## The orders used can influence the final RV
	## and the error quite a bit
	## We'll start out by using some of the 
	## (typically) well-behaved orders (~5000-6000 AA)
	norders = range(40,60)

	## Dictionary to store the prepped data
	## normalized spectra/initial RVs
	prepped = fp.prepareSpectra(filenames,norders,tw,tf)

	## Now we'll use all these prepped spectra
	## to create the first iteration of splines	
	splines = fp.makeSplines(prepped)

	## Now we'll use the splines as a template
	## to get the RVs
	## The procedure is very similar to the one above

	## Dictionary to store the results
	wrvs = fp.matchRVs(prepped,splines)

	## We'll now use the newly derived RVs
	## in the second iteration
	for file in wrvs:
		bjd, rv, erv, bvc = wrvs[file]
		prepped[file]['rv'] = rv
		prepped[file]['erv'] = erv
	## and we'll make a new batch of splines

	## New dictionary to store the splines
	second = fp.makeSplines(prepped)

	## We could now use the splines to get the RVs for each individual spectrum again,
	## but this time we'll coadd the spectra from different epochs first

	## We'll store the weighted average RV and error
	## for each epoch in arrays
	final_rvs = np.array([])
	final_ervs = np.array([])
	## As well as the time and barycentric correction
	final_bjds = np.array([])
	final_bvcs = np.array([])

	for epoch in epochs:
		filenames = epochs[epoch]['names']
		## Fine grid centered on the measrued RV
		rv = np.array([])
		for file in filenames:
			rv = np.append(rv,prepped[file]['rv'])
		rvepoch = np.median(rv)
		drvs = np.arange(rvepoch - 10.0, rvepoch + 10.0, 0.1)
		
		## Coadd the spectra
		coadd = fp.coaddSpectra(filenames,prepped)
		## Get the RVs
		wavg_rv, wavg_err = fp.splineRVs(coadd,second,drvs)
		## Get the barycentric correction
		## and the BJD in TDB
		bjds = np.array([])
		bvcs = np.array([])
		for file in filenames:
			bjd, bvc = fp.getBarycorrs(file,wavg_rv)
			bjds = np.append(bjds,bjd)
			bvcs = np.append(bvcs,bvc)
		## ... this could be done in a more sophisticated way
		## by for example using the flux weighted midpoint of the exposure
		## but this is good enough for now

		final_bjds = np.append(final_bjds,np.mean(bjds))
		final_bvcs = np.append(final_bvcs,np.mean(bvcs))
		final_rvs = np.append(final_rvs,wavg_rv+np.mean(bvcs))
		final_ervs = np.append(final_ervs,wavg_err)

	## Plot the RVs
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\mathrm{BJD}_\mathrm{TDB}$')
	ax.set_ylabel(r'RV (km/s)')
	ax.errorbar(final_bjds,final_rvs,yerr=final_ervs,fmt='o',color='k')
	if save: fig.savefig('./rvs_func.png',bbox_inches='tight')

	return final_bjds, final_rvs, final_ervs, final_bvcs

if __name__ == '__main__':
	bjds, rvs, ervs, bvcs = tempmatch(save=1)
	print(bjds,rvs,ervs,bvcs)

	for ii, bjd in enumerate(bjds):
		print('BJD: {:.5f}, RV: {:.3f} +/- {:.3f} m/s'.format(bjd,rvs[ii]*1e3,ervs[ii]*1e3))

	arr = np.array([bjds,rvs,ervs]).T
