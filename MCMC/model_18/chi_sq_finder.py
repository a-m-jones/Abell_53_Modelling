#!/usr/bin/python

## import necessary modules

import numpy as np
import matplotlib.pyplot as plt
import os
import pyCloudy as pc
import glob
from datetime import datetime
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from scipy import interpolate
import pandas as pd
from joblib import Parallel, delayed
import time
from subprocess import call, check_output
from scipy.integrate import trapz
from scipy.optimize import fsolve
from scipy.stats import chi2
import string
import random

def chi_sq_finder(cs_T, cs_L, r_c, n_1, n_2, FF):
	# these are the parameters that will be fit in the MCMC procedure - 
	# can change at will

	## Define emission line dictionary that allows Cloudy and pyCloudy to read lines ##

	emis_lines = {'OII_3727'   : ['BLND_372700A', 'Blnd 3727.00A'],
				  'NeIII_3868' : ['NE_3_386876A', 'Ne 3 3868.76A'],
				  'NeIII_3969' : ['NE_3_396747A', 'Ne 3 3967.47A'],
	              'Hd_4102'    : ['H__1_410173A', 'H  1 4101.73A'],
				  'Hc_4340'    : ['H__1_434046A', 'H  1 4340.46A'],
	              'OIII_4363'  : ['BLND_436300A', 'Blnd 4363.00A'],
				  'HeII_4686'  : ['HE_2_468564A', 'He 2 4685.64A'],
				  'Hb_4861'    : ['H__1_486133A', 'H  1 4861.33A'],
				  'OIII_4959'  : ['O__3_495891A', 'O  3 4958.91A'],
				  'OIII_5007'  : ['O__3_500684A', 'O  3 5006.84A'],
				  'NI_5199'    : ['N__1_519790A', 'N  1 5197.90A'],
				  'NII_5754'   : ['N__2_575461A', 'N  2 5754.61A'],
				  'HeI_5876'   : ['HE_1_587564A', 'He 1 5875.64A'],
				  'OI_6300'    : ['O__1_630030A', 'O  1 6300.30A'],
				  'SIII_6312'  : ['S__3_631206A', 'S  3 6312.06A'],
				  'OI_6364'    : ['O__1_636378A', 'O  1 6363.78A'],
				  'NII_6548'   : ['N__2_654805A', 'N  2 6548.05A'],
				  'Ha_6563'    : ['H__1_656281A', 'H  1 6562.81A'],
				  'NII_6583'   : ['N__2_658345A', 'N  2 6583.45A'],
				  'HeI_6678'   : ['HE_1_667815A', 'He 1 6678.15A'],
				  'SII_6716'   : ['S__2_671644A', 'S  2 6716.44A'],
				  'SII_6731'   : ['S__2_673082A', 'S  2 6730.82A'],
				  'ArV_7006'   : ['AR_5_700583A', 'Ar 5 7005.83A'],
				  'HeI_7065'   : ['HE_1_706522A', 'He 1 7065.22A'],
				  'ArIII_7136' : ['AR_3_713579A', 'Ar 3 7135.79A'],
				  'HeI_7281'   : ['HE_1_728135A', 'He 1 7281.35A'],
				  'OII_7320'   : ['BLND_732300A', 'Blnd 7323.00A'],
				  'OII_7330'   : ['BLND_733200A', 'Blnd 7332.00A'],
				  'ArIII_7751' : ['AR_3_775111A', 'Ar 3 7751.11A']}

	## Define some lists ##

	# abundances are fixed here
	abundances = ['helium -0.885172','nitrogen -3.753978', 'oxygen -3.095145', 'neon -3.357303','sulphur -5.327289',
					'argon -5.469643','carbon -3.188', 'silicon -6.699','chlorine -6.796','magnesium -6.886','iron -7.097']
	# line labels for the fluxes that are to be fit to
	lines_total = ['OII_3727', 'NeIII_3868', 'NeIII_3969', 'Hc_4340', 'HeII_4686',
				   'OIII_4959', 'OIII_5007', 'NI_5199', 'NII_5754', 
	 			   'HeI_5876', 'OI_6300', 'SIII_6312', 'OI_6364', 'NII_6548',
	 			   'NII_6583', 'HeI_6678', 'SII_6716', 'SII_6731', 'HeI_7065', 
	 			   'ArIII_7136', 'HeI_7281', 'OII_7320', 'OII_7330', 'ArIII_7751']
	# names of SB profiles to be fit to
	lines_profiles = ['Ha_6563', 'OIII_5007', 'NII_6583', 'SII_6731', 'HeI_5876', 'HeII_4686']
	# some other parameters to go into the Cloudy input file
	other = ['init file="ism.ini"',
			 'COSMIC RAY BACKGROUND',
			 'print lines sort wavelength',
			 'print last iteration',
			 'sphere',
			 'iterate',
			 'stop temperature off']
	save_commands = ['save last radius ".rad"',
					 'save last continuum ".cont"',
					 'save last overview ".ovr"',
					 'save last physical conditions ".phy"']

	## Define some fixed parameters ##

	distance = 2.029 # in kpc
	r_out    = 0.167042 # outer radius in pc, always fixed for MCMC procedure
	obs_ha_flux = 7.6724e-11 # the observed total H-alpha flux from photometry
	obs_ha_unc  = 1.2766e-11 # associated uncertainty

	## Create the cloudy model ## 

	# Creates a random file name for the input model
	stringLength = 20
	letters = string.ascii_letters
	model_name = ''.join(random.choice(letters) for i in range(stringLength))
	input_model = model_name+'.in'

	# this writes the cloudy input file - edit this depending on what is needed to fit
	with open(input_model, 'w') as f:
		f.write("#"*5 + "CS parameters" + "#"*5 + "\n")
		# can change next line to "blackbody {} \n".format(cs_T)
		# if only black body is needed
		f.write("table star Rauch log {} 7.0 \n".format(cs_T))
		f.write("luminosity total solar {} \n".format(cs_L))
		f.write("#"*5 + "Geometry" + "#"*5 + "\n")
		# starting radius - fixed here at 0.07 pc
		f.write("radius 17.33449 \n")
		f.write("distance {} parsecs linear \n".format(1000.*distance))
		# dlaw command allows various densities to be defined at particular radii
		# this can be changed as desired
		f.write("dlaw table radius \n")
		f.write("continue 14.5 {} \n".format(n_1))
		f.write("continue {} {} \n".format(r_c, n_1))
		f.write("continue {} {} \n".format(np.log10((10**r_c)+3.086e15), n_2))
		f.write("continue 18.0 {} \n".format(n_2))
		f.write("end of dlaw \n")
		f.write("filling factor {} \n".format(FF))
		f.write("#"*5 + "Dust and shocks" + "#"*5 + "\n")
		f.write("grains ISM \n")
		f.write("#"*5 + "Abundances" + "#"*5 + "\n")       
		for abun in abundances:
			f.write("element abundance {} \n".format(abun))
		f.write("#"*5 + "Stopping condition" + "#"*5 + "\n")
		f.write("stop radius {}\n".format(np.log10(r_out*3.086e18)))
		f.write("#"*5 + "Other" + "#"*5 + "\n")    
		for j in other:
			f.write(j + '\n')    
		for j in save_commands:
			f.write(j + '\n')
		f.write("#"*5 + "Lines to save" + "#"*5 + "\n")
		f.write('save last lines emissivity ".emis" \n')
		for j in emis_lines.values():
			f.write(j[1] + '\n')
		f.write("end of lines \n")

	## run the model ##

	# change location of cloudy.exe file
	exe = '/home/ajones/Cloudy/c17.01/source/cloudy.exe'
	call([exe, input_model])

	## load the pycloudy model ##
	# this section makes the 3D model and the SB profiles

	cube_size = 340
	centre = 1
	n_dim  = 1

	model = pc.load_models('{}'.format(model_name), read_grains=True, distance=distance)[0]
	m3d   = pc.C3D(list_of_models=model, dims=cube_size, center=centre, n_dim=n_dim)

	x_arcsec        = np.arange(0.05, 17.0, 0.1)
	x_arcsec_interp = np.arange(0.0, 17.1, 0.25)

	def image_maker(line, prof):
		line_id    = emis_lines[line][0]
		## create the 2d image from the emissivity profile
		image      = (m3d.get_emis(line_id) * m3d.ff).sum(0) * m3d.cub_coord.cell_size / (4.0*np.pi*(distance*1000.0*3.086e18)**2)
		## convolve to match observations
		kernel     = Gaussian2DKernel(6.5/2.355)  # 6.5 pix = 0.65 arcsec
		conv_image = convolve(image, kernel)
		## extract a 1 x 17 arcsec width box
		cutout     = conv_image[165:175, 170:340]
		## sum across to make the profile
		y          = np.sum(cutout, axis=0)  # the sb profile - now need to calibrate
		## flux calibrate
		slit_flux  = np.sum(y)        # flux contained within half the slit
		if prof:
			area  = trapz(y=y, x=x_arcsec, axis=-1)
			cal_y = y * slit_flux / area
			## interpolate onto same radii as observations
			interp_y = np.interp(x_arcsec_interp, x_arcsec, cal_y)
			return interp_y
		else:
			return slit_flux

	## find total h-alpha chi sq
	total_ha_flux  = model.get_emis_vol(ref='H__1_656281A', at_earth=True)
	total_ha_chisq = (obs_ha_flux - total_ha_flux)**2 / obs_ha_unc**2

	## find surface brightness profiles chi sq
	obs_profiles = pd.read_csv('../data/observed_sb_profiles.txt', delimiter=' ')
	profile_chisqs = []
	for line in lines_profiles:
		obs_profile_val = obs_profiles[line+'_a'].values
		obs_profile_unc = obs_profiles[line+'_a_unc'].values
		chisq = np.sum((obs_profile_val - image_maker(line, prof=True))**2 / obs_profile_unc**2)
		profile_chisqs.append(chisq)
	profile_chisq = np.sum(profile_chisqs)

	## find line ratios chi sq
	obs_line_ratios = np.genfromtxt('../data/observed_line_ratios.txt', delimiter=' ', skip_header=1)
	obs_line_ratios_val = obs_line_ratios[:,1]
	obs_line_ratios_unc = obs_line_ratios[:,2]
	ha_slit_flux = image_maker('Ha_6563', prof=False)
	slit_line_ratios = [image_maker(line, prof=False)/ha_slit_flux for line in lines_total]
	line_ratios_chisq = np.sum((obs_line_ratios_val - slit_line_ratios)**2 / obs_line_ratios_unc**2)

	## delete files ##

	endings = ['.cont', '.emis', '.in', '.out', '.ovr', '.phy', '.rad']

	for end in endings:
		os.system('rm ' + model_name + end)

	## calculate the total chi squared ##

	total_chi_squared = total_ha_chisq + line_ratios_chisq + profile_chisq

	return total_chi_squared


if __name__=='__main__':
	# if main parameters are changed in above function, they need to be changed here too!
	import sys

	cs_T=float(sys.argv[1])
	cs_L=float(sys.argv[2])
	r_c=float(sys.argv[3])
	n_1=float(sys.argv[4])
	n_2=float(sys.argv[5])
	FF=float(sys.argv[6])

	print chi_sq_finder(cs_T, cs_L, r_c, n_1, n_2, FF)
