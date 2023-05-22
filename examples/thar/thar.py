'''
.. codeauthor:: Emil Knudstrup

.. code-block:: python

	import FIESpipe as fp
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy.interpolate as sci

	## Set the path to the data and template
	path = 'your_path/to/spectra/'

	## Again, we'll start by grouping the spectra into 
	## science and ThAr files
	filenames, tharnames = fp.sortFIES(path)
	tharnames.sort()

	## Again we'll start use some of the central orders
	## (~5000-6000 AA)
	norders = range(40,60)
	## Prepare the ThAr spectra
	prepped = fp.prepareThAr(tharnames,norders=norders)
	## Create the splines
	splines = fp.tharSplines(prepped,norders=norders)
	## Get the RVs
	wrvs = fp.thaRVs(prepped,splines)

	## This is how the RVs look
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\mathrm{BJD}_\mathrm{TDB}$')
	ax.set_ylabel(r'RV (m/s)')
	ax.axhline(0.0,linestyle='--',color='C7')
	## Collect RVs and timestamps
	## to correct for the drift below
	thimes = np.array([])
	tharvs = np.array([])

	for ii, file in enumerate(tharnames):
		bjd, rv, erv = wrvs[file]
		ax.errorbar(bjd,rv*1e3,yerr=erv*1e3,fmt='o',mfc='C0',mec='k',ecolor='C0')
		thimes = np.append(thimes,bjd)
		tharvs = np.append(tharvs,rv)

		
.. image:: ../../../examples/thar/drift_func.png

.. code-block:: python

	## In the following we'll use the RVs from the ThAr exposures
	## to correct the drift in the RVs of the science exposures
	## Here we'll assume that all science exposures are sandwiched between
	## two ThAr exposures, which is also when it only really makes sense
	dvs = np.array([])
	bjds = np.array([])
	for ii, file in enumerate(filenames):
		bjd, _ = fp.getBarycorrs(file,rvmeas=0.0)
		bjds = np.append(bjds,bjd)

	dvs = fp.ThArcorr(thimes,tharvs,bjds)

	## These are then the correction to the science RVs

'''

#%%
import FIESpipe as fp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import os
path = os.path.dirname(os.path.abspath(__file__))

def thar(fpath = 'data/spectra/KELT-3/',
		save = False,
		width = 15,
		height = 6,
		):
	'''Example showing how to correct for the ThAr Drift.
	
	Used for testing.

	:param fpath: Path to the FIES data. Default is ``data/spectra/KELT-3/``.
	:type fpath: str
	:param save: Save the figures. Default is ``False``.
	:type save: bool
	:param width: Width of the figures. Default is ``15`` (inches).
	:type width: float
	:param height: Height of the figures. Default is ``6`` (inches).
	:type height: float

	:return: Correction to RVs.
	:rtype: array

	'''


	## Again, we'll start by grouping the spectra into 
	## science and ThAr files
	filenames, tharnames = fp.sortFIES(path+'/../../'+fpath)
	tharnames.sort()

	norders = range(40,60)

	## Prepare the ThAr spectra
	prepped = fp.prepareThAr(tharnames,norders=norders)
	## Create the splines
	splines = fp.tharSplines(prepped,norders=norders)
	## Get the RVs
	wrvs = fp.thaRVs(prepped,splines)

	## This is how the RVs look
	fig = plt.figure(figsize=(width,height))
	ax = fig.add_subplot(111)
	ax.set_xlabel(r'$\mathrm{BJD}_\mathrm{TDB}$')
	ax.set_ylabel(r'RV (m/s)')
	ax.axhline(0.0,linestyle='--',color='C7')
	## Collect RVs and timestamps
	## to correct for the drift below
	thimes = np.array([])
	tharvs = np.array([])

	for ii, file in enumerate(tharnames):
		bjd, rv, erv = wrvs[file]
		ax.errorbar(bjd,rv*1e3,yerr=erv*1e3,fmt='o',mfc='C0',mec='k',ecolor='C0')
		thimes = np.append(thimes,bjd)
		tharvs = np.append(tharvs,rv)
	if save: fig.savefig('./drift_func.png',bbox_inches='tight')


	## In the following we'll use the RVs from the ThAr exposures
	## to correct the drift in the RVs of the science exposures
	## Here we'll assume that all science exposures are sandwiched between
	## two ThAr exposures, which is also when it only really makes sense
	dvs = np.array([])
	bjds = np.array([])
	for ii, file in enumerate(filenames):
		bjd, _ = fp.getBarycorrs(file,rvmeas=0.0)
		bjds = np.append(bjds,bjd)

	dvs = fp.ThArcorr(thimes,tharvs,bjds)

	## These are then the correction to the science RVs
	return dvs

if __name__ == '__main__':
	dvs = thar()

#%%