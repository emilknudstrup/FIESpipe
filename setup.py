from setuptools import find_packages, setup

dependencies=''
with open("requirements.txt","r") as f:
	dependencies = f.read().splitlines()



setup(
	name="FIESpipe",
	version='0.0.1',
	description='extract parameters from FIES spectra',
	url='https://github.com/emilknudstrup/FIESpipe',
	packages=find_packages(where="src"),
	package_dir={"": "src"},
	include_package_data=True,
	classifiers = ["Programming Language :: Python :: 3"],
	install_requires = dependencies
	
)
