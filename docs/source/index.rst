.. FIESpipe documentation master file, created by
   sphinx-quickstart on Wed Apr  5 19:33:25 2023.


Welcome to FIESpipe's documentation!
====================================

FIESpipe is a software package used to extract various data products from spectra acquired with the FIES spectrograph :cite:p:`Telting2014` at the Nordic Optical Telescope :cite:p:`Djupvik2010`. The package is intended to be used on the reduced "order-by-order" FIES spectra extracted from the `FIEStool <http://www.not.iac.es/instruments/fies/fiestool/>`_ data reduction software.

There are a number of examples showing how to extract various quantities.


Radial velocities (RVs) can be extracted through either:

* the cross-correlation function (:ref:`basic`),
* the broadening function (:ref:`broad`),
* :math:`\chi^2` minimization (:ref:`chisq`),
* or template matching (:ref:`tempmatch`)

Furthermore, in :ref:`thar` an example on how to trace the (RV) drift of the spectrograph using the ThAr calibration spectra is given.

Activity indicators, namely the Full Width Half Maximum (FWHM) and Bisector Inverse Slope (BIS), are calculated in :ref:`basic`, and the S-index is calculated in :ref:`sidx`.

The examples show how to extract the quantities of interest through individual function calls, whereas the modules in :ref:`workflow` are collections of these function calls. Plots showing these products can be created using the functions in :ref:`evince`.

In `templates <examples/temps.ipynb>`_ it is shown how to download and read/extract wavelength,flux from some template spectra.

.. image:: gallery/ccf.gif

.. toctree::
   :maxdepth: 2
   :caption: Workflows
   
   workflows

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/basic
   examples/broad
   examples/chisq
   examples/tempmatch
   examples/sidx
   examples/thar
   examples/temps

   
.. toctree::
   :maxdepth: 2
   :caption: API
   
   API/extract
   API/derive
   API/evince

.. toctree::
   :maxdepth: 2
   :caption: Installation
   
   installation

.. toctree::
   :maxdepth: 1
   :caption: References

   references

.. toctree::
   :maxdepth: 1
   :caption: Acknowledgements

   ackn 

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
