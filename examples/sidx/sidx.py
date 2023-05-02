'''

The calculation of the S-index is based on the method described in Section 3.1.2 of :cite:t:`BoroSaikia2018`.
The S-index is defined as the ratio of the flux in the Ca II H and K lines to the flux in two continuum regions:

.. math::
	S = \\alpha \\frac{F_\mathrm{CaIIH} + F_\mathrm{CaIIK}}{F_\mathrm{R} + F_\mathrm{V}} \, ,

where :math:`\\alpha` is a proportionality constant.


.. note::

	The S-index calculated here is not calibrated to the Mount Wilson S-index.
	
.. code-block:: python

	import FIESpipe as fp
	import matplotlib.pyplot as plt
	import numpy as np

	## Sort the FIES files into science and ThAr spectra
	fpath = '/home/au339813/Desktop/PhD/projects/gamCep_master/FIES/spectra/'
	filenames, tharnames = fp.sortFIES(fpath)
	filenames.sort()
	## Grab one spectrum
	file = filenames[0]

	## Extract the data from the FITS file
	wave, flux, ferr, hdr = fp.extractFIES(file)

	## Number of orders
	orders = range(1,len(wave)+1)

	## Radial velocity of the star
	## used to shift the spectra to the rest frame
	rvsys = -44.372 # km/s, see basic.py
	## Barycentric velocity correction
	_, bvc = fp.getBarycorrs(file,rvmeas=rvsys)
	## Radial velocity correction
	rvc = rvsys - bvc

	## Position of Ca II H and K lines
	CaIIH = 3968.47 # AA
	CaIIK = 3933.664 # AA
	## Continuum regions
	## R: 3991.07 - 4011.07 AA
	Rsection_low = 3991.07
	Rsection_high = 4011.07
	## V: 3891.07 - 3911.07 AA
	Vsection_low = 3891.07
	Vsection_high = 3911.07

	## Width of the Ca II H and K lines
	linewidth = 1.09
	## Dictionary to store the data
	data = {
		'CaIIH' : {'orders':[]},
		'CaIIK' : {'orders':[]},
		'Rcontinuum' : {'orders':[]},
		'Vcontinuum' : {'orders':[]},
	}

	## Only include the Ca II H and K lines
	## if they are not towards the edge of the order
	exc = 10 # AA

	## Loop over the orders
	## to identify the lines and continuum regions
	for ii in orders:
		## Extract the data, 0-indexed
		w, f, e = wave[ii-1], flux[ii-1], ferr[ii-1]

		## Shift raw wavelength to rest frame
		swl = w/(1.0 + rvc*1e3/fp.const.c.value)

		## Ca II H and K lines
		## Grab if the order contains the lines
		if len(np.where( (CaIIH > (min(swl)+exc)) & (CaIIH < (max(swl)-exc)) )[0]):
			data['CaIIH']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['CaIIH']['order_{}'.format(ii)] = arr
		if len(np.where( (CaIIK > (min(swl)+exc)) & (CaIIK < (max(swl)-exc)) )[0]):
			data['CaIIK']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['CaIIK']['order_{}'.format(ii)] = arr

		## Continuum regions
		## Grab if the order spans the full region
		if any(swl < Rsection_low) and any(swl > Rsection_high):
			data['Rcontinuum']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['Rcontinuum']['order_{}'.format(ii)] = arr
		if any(swl < Vsection_low) and any(swl > Vsection_high):
			data['Vcontinuum']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['Vcontinuum']['order_{}'.format(ii)] = arr

.. code-block:: python

	## Get S-index 
	## and plot the quantities that go into the calculation
	fig = plt.figure(figsize=(width,height))
	gs = fig.add_gridspec(4,3)
	axK = fig.add_subplot(gs[0,:2])
	axKz = fig.add_subplot(gs[0,2])
	axH = fig.add_subplot(gs[1,:2])
	axHz = fig.add_subplot(gs[1,2])
	axV = fig.add_subplot(gs[2,:2])
	axVz = fig.add_subplot(gs[2,2])
	axR = fig.add_subplot(gs[3,:2])
	axRz = fig.add_subplot(gs[3,2])

	## CaII K line
	Korders = data['CaIIK']['orders']
	Ks = []
	for ii, order in enumerate(Korders):
		key = 'order_{}'.format(order)
		arr = data['CaIIK'][key]
		wl = arr[0]
		fl = arr[1]
		## The weighting is given as a triangular function
		## centered on the line with a width of 1.09 AA
		weights = fp.triangle(wl,CaIIK,linewidth*0.5,linewidth*0.5)
		core = weights > 0.0
		K = np.median(fl[core]*weights[core])
		Ks.append(K)
		## Plot the data
		axK.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axKz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axK.axvline(CaIIK,color='C7',ls='--',lw=1.0)
		axKz.axvline(CaIIK,color='C7',ls='--',lw=1.0)
		axK.plot(wl,weights,color='k')
		axKz.plot(wl,weights,color='k')
		axK.plot(wl[core],weights[core]*fl[core],color='C3')
		axKz.plot(wl[core],weights[core]*fl[core],color='C3')
	axKz.set_xlim(CaIIK-linewidth*3,CaIIK+linewidth*3)

	## CaII H line
	Horders = data['CaIIH']['orders']
	Hs = []
	for ii, order in enumerate(Horders):
		key = 'order_{}'.format(order)
		arr = data['CaIIH'][key]
		wl = arr[0]
		fl = arr[1]
		weights = fp.triangle(wl,CaIIH,linewidth*0.5,linewidth*0.5)
		core = weights > 0.0
		H = np.median(fl[core]*weights[core])
		Hs.append(H)
		## Plot the data
		axH.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axHz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axH.axvline(CaIIH,color='C7',ls='--',lw=1.0)
		axHz.axvline(CaIIH,color='C7',ls='--',lw=1.0)
		axH.plot(wl,weights,color='k')
		axHz.plot(wl,weights,color='k')
		axH.plot(wl[core],weights[core]*fl[core],color='C3')
		axHz.plot(wl[core],weights[core]*fl[core],color='C3')
	axHz.set_xlim(CaIIH-linewidth*3,CaIIH+linewidth*3)

	## V continuum
	Vorders = data['Vcontinuum']['orders']
	Vconts = []
	for ii, order in enumerate(Vorders):
		key = 'order_{}'.format(order)
		arr = data['Vcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		## The continuum is the median of the flux in the region
		Vcont = np.median(fl[(wl > Vsection_low) & (wl < Vsection_high)])
		Vconts.append(Vcont)
		## Plot the data
		axV.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axVz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axV.fill_betweenx([min(fl),max(fl)],[Vsection_low,Vsection_low],[Vsection_high,Vsection_high],color='C1',alpha=0.2)
		axVz.fill_betweenx([min(fl),max(fl)],[Vsection_low,Vsection_low],[Vsection_high,Vsection_high],color='C1',alpha=0.2)
		axV.axvline(Vsection_low,color='C7',ls='--',lw=0.75)
		axVz.axvline(Vsection_low,color='C7',ls='--',lw=0.75)
		axV.axvline(Vsection_high,color='C7',ls='--',lw=0.75)
		axVz.axvline(Vsection_high,color='C7',ls='--',lw=0.75)
	axVz.set_xlim(Vsection_low,Vsection_high)

	## R continuum
	Rorders = data['Rcontinuum']['orders']
	Rconts = []
	for ii, order in enumerate(Rorders):
		key = 'order_{}'.format(order)
		arr = data['Rcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		Rcont = np.median(fl[(wl > Rsection_low) & (wl < Rsection_high)])
		Rconts.append(Rcont)
		## Plot the data
		axR.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axRz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axR.fill_betweenx([min(fl),max(fl)],[Rsection_low,Rsection_low],[Rsection_high,Rsection_high],color='C3',alpha=0.2)
		axRz.fill_betweenx([min(fl),max(fl)],[Rsection_low,Rsection_low],[Rsection_high,Rsection_high],color='C3',alpha=0.2)
		axR.axvline(Rsection_low,color='C7',ls='--',lw=0.75)
		axRz.axvline(Rsection_low,color='C7',ls='--',lw=0.75)
		axR.axvline(Rsection_high,color='C7',ls='--',lw=0.75)
		axRz.axvline(Rsection_high,color='C7',ls='--',lw=0.75)
	axRz.set_xlim(Rsection_low,Rsection_high)

	## And here we'll just polish the plots up a bit
	axR.set_xlabel(r'$\lambda \ (\AA)$')
	axRz.set_xlabel(r'$\lambda \ (\AA)$')
	axes = [axK,axH,axV,axR]
	for ax in axes:
		ax.set_ylabel(r'$\\rm F_\lambda$')

	axz = [axKz,axHz,axVz,axRz]
	for ax in axz:
		ax.set_yticklabels([])

	plt.subplots_adjust(hspace=0.25,wspace=0.0)

.. image:: ../../../examples/sidx/sindex.png

	

.. code-block:: python

	## Now we'll calculate the S index
	## This is the uncalibrated version
	S = (np.mean(Ks) + np.mean(Hs))/(np.mean(Vconts) + np.mean(Rconts))
	print('Uncalicrated S-index = {:.3f}'.format(S))


'''
#%%
import FIESpipe as fp
import matplotlib.pyplot as plt
import numpy as np
import os
path = os.path.dirname(os.path.abspath(__file__))

def sidx(
	fpath='data/spectra/gamCep/',
	save=False,
	width=15,
	height=6,	
	):
	'''Calculate the S-index for a FIES spectrum.

	Used for testing.

	:param fpath: Path to the FIES data.
	:type fpath: str
	:param save: Save the plot?
	:type save: bool
	:param width: Width of the plot in inches.
	:type width: float
	:param height: Height of the plot in inches.
	:type height: float

	:return: Uncaclibrated S-index.
	:rtype: float
	'''


	## Sort the FIES files into science and ThAr spectra
	filenames, tharnames = fp.sortFIES(path+'/../../'+fpath)
	filenames.sort()
	## Grab one spectrum
	file = filenames[0]

	## Extract the data from the FITS file
	wave, flux, ferr, hdr = fp.extractFIES(file)

	## Number of orders
	orders = range(1,len(wave)+1)

	## Radial velocity of the star
	## used to shift the spectra to the rest frame
	rvsys = -44.372 # km/s, see basic.py
	## Barycentric velocity correction
	_, bvc = fp.getBarycorrs(file,rvmeas=rvsys)
	## Radial velocity correction
	rvc = rvsys - bvc

	## Position of Ca II H and K lines
	CaIIH = 3968.47 # AA
	CaIIK = 3933.664 # AA
	## Continuum regions
	## R: 3991.07 - 4011.07 AA
	Rsection_low = 3991.07
	Rsection_high = 4011.07
	## V: 3891.07 - 3911.07 AA
	Vsection_low = 3891.07
	Vsection_high = 3911.07

	## Width of the Ca II H and K lines
	linewidth = 1.09
	## Dictionary to store the data
	data = {
		'CaIIH' : {'orders':[]},
		'CaIIK' : {'orders':[]},
		'Rcontinuum' : {'orders':[]},
		'Vcontinuum' : {'orders':[]},
	}

	## Only include the Ca II H and K lines
	## if they are not towards the edge of the order
	exc = 10 # AA

	## Loop over the orders
	## to identify the lines and continuum regions
	for ii in orders:
		## Extract the data, 0-indexed
		w, f, e = wave[ii-1], flux[ii-1], ferr[ii-1]

		## Shift raw wavelength to rest frame
		swl = w/(1.0 + rvc*1e3/fp.const.c.value)

		## Ca II H and K lines
		## Grab if the order contains the lines
		if len(np.where( (CaIIH > (min(swl)+exc)) & (CaIIH < (max(swl)-exc)) )[0]):
			data['CaIIH']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['CaIIH']['order_{}'.format(ii)] = arr
		if len(np.where( (CaIIK > (min(swl)+exc)) & (CaIIK < (max(swl)-exc)) )[0]):
			data['CaIIK']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['CaIIK']['order_{}'.format(ii)] = arr

		## Continuum regions
		## Grab if the order spans the full region
		if any(swl < Rsection_low) and any(swl > Rsection_high):
			data['Rcontinuum']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['Rcontinuum']['order_{}'.format(ii)] = arr
		if any(swl < Vsection_low) and any(swl > Vsection_high):
			data['Vcontinuum']['orders'].append(ii)
			arr = np.array([swl,f,e])
			data['Vcontinuum']['order_{}'.format(ii)] = arr

	## Get S-index 
	## and plot the quantities that go into the calculation
	fig = plt.figure(figsize=(width,height))
	gs = fig.add_gridspec(4,3)
	axK = fig.add_subplot(gs[0,:2])
	axKz = fig.add_subplot(gs[0,2])
	axH = fig.add_subplot(gs[1,:2])
	axHz = fig.add_subplot(gs[1,2])
	axV = fig.add_subplot(gs[2,:2])
	axVz = fig.add_subplot(gs[2,2])
	axR = fig.add_subplot(gs[3,:2])
	axRz = fig.add_subplot(gs[3,2])

	## CaII K line
	Korders = data['CaIIK']['orders']
	Ks = []
	for ii, order in enumerate(Korders):
		key = 'order_{}'.format(order)
		arr = data['CaIIK'][key]
		wl = arr[0]
		fl = arr[1]
		## The weighting is given as a triangular function
		## centered on the line with a width of 1.09 AA
		weights = fp.triangle(wl,CaIIK,linewidth*0.5,linewidth*0.5)
		core = weights > 0.0
		K = np.median(fl[core]*weights[core])
		Ks.append(K)
		## Plot the data
		axK.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axKz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axK.axvline(CaIIK,color='C7',ls='--',lw=1.0)
		axKz.axvline(CaIIK,color='C7',ls='--',lw=1.0)
		axK.plot(wl,weights,color='k')
		axKz.plot(wl,weights,color='k')
		axK.plot(wl[core],weights[core]*fl[core],color='C3')
		axKz.plot(wl[core],weights[core]*fl[core],color='C3')
	axKz.set_xlim(CaIIK-linewidth*3,CaIIK+linewidth*3)

	## CaII H line
	Horders = data['CaIIH']['orders']
	Hs = []
	for ii, order in enumerate(Horders):
		key = 'order_{}'.format(order)
		arr = data['CaIIH'][key]
		wl = arr[0]
		fl = arr[1]
		weights = fp.triangle(wl,CaIIH,linewidth*0.5,linewidth*0.5)
		core = weights > 0.0
		H = np.median(fl[core]*weights[core])
		Hs.append(H)
		## Plot the data
		axH.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axHz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axH.axvline(CaIIH,color='C7',ls='--',lw=1.0)
		axHz.axvline(CaIIH,color='C7',ls='--',lw=1.0)
		axH.plot(wl,weights,color='k')
		axHz.plot(wl,weights,color='k')
		axH.plot(wl[core],weights[core]*fl[core],color='C3')
		axHz.plot(wl[core],weights[core]*fl[core],color='C3')
	axHz.set_xlim(CaIIH-linewidth*3,CaIIH+linewidth*3)

	## V continuum
	Vorders = data['Vcontinuum']['orders']
	Vconts = []
	for ii, order in enumerate(Vorders):
		key = 'order_{}'.format(order)
		arr = data['Vcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		## The continuum is the median of the flux in the region
		Vcont = np.median(fl[(wl > Vsection_low) & (wl < Vsection_high)])
		Vconts.append(Vcont)
		## Plot the data
		axV.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axVz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axV.fill_betweenx([min(fl),max(fl)],[Vsection_low,Vsection_low],[Vsection_high,Vsection_high],color='C1',alpha=0.2)
		axVz.fill_betweenx([min(fl),max(fl)],[Vsection_low,Vsection_low],[Vsection_high,Vsection_high],color='C1',alpha=0.2)
		axV.axvline(Vsection_low,color='C7',ls='--',lw=0.75)
		axVz.axvline(Vsection_low,color='C7',ls='--',lw=0.75)
		axV.axvline(Vsection_high,color='C7',ls='--',lw=0.75)
		axVz.axvline(Vsection_high,color='C7',ls='--',lw=0.75)
	axVz.set_xlim(Vsection_low,Vsection_high)

	## R continuum
	Rorders = data['Rcontinuum']['orders']
	Rconts = []
	for ii, order in enumerate(Rorders):
		key = 'order_{}'.format(order)
		arr = data['Rcontinuum'][key]
		wl = arr[0]
		fl = arr[1]
		Rcont = np.median(fl[(wl > Rsection_low) & (wl < Rsection_high)])
		Rconts.append(Rcont)
		## Plot the data
		axR.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axRz.plot(wl,fl,color='C{}'.format(ii),lw=1.0)
		axR.fill_betweenx([min(fl),max(fl)],[Rsection_low,Rsection_low],[Rsection_high,Rsection_high],color='C3',alpha=0.2)
		axRz.fill_betweenx([min(fl),max(fl)],[Rsection_low,Rsection_low],[Rsection_high,Rsection_high],color='C3',alpha=0.2)
		axR.axvline(Rsection_low,color='C7',ls='--',lw=0.75)
		axRz.axvline(Rsection_low,color='C7',ls='--',lw=0.75)
		axR.axvline(Rsection_high,color='C7',ls='--',lw=0.75)
		axRz.axvline(Rsection_high,color='C7',ls='--',lw=0.75)
	axRz.set_xlim(Rsection_low,Rsection_high)

	## And here we'll just polish the plots up a bit
	axR.set_xlabel(r'$\lambda \ (\AA)$')
	axRz.set_xlabel(r'$\lambda \ (\AA)$')
	axes = [axK,axH,axV,axR]
	for ax in axes:
		ax.set_ylabel(r'$\rm F_\lambda$')

	axz = [axKz,axHz,axVz,axRz]
	for ax in axz:
		ax.set_yticklabels([])

	plt.subplots_adjust(hspace=0.25,wspace=0.0)
	if save: plt.savefig('./sindex.png',bbox_inches='tight')
	## Now we'll calculate the S index
	## This is the uncalibrated version
	S = (np.mean(Ks) + np.mean(Hs))/(np.mean(Vconts) + np.mean(Rconts))
	print('Uncalicrated S-index = {:.3f}'.format(S))

	return S

if __name__ == '__main__':
	s = sidx()
