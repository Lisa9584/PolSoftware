# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 19 16:17:41 2020

# @author: Lisa
# """
import os, sys
import numpy as np
import PolMath
import PolLibrary
import PolAnalysis as pa

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def createNFTdetector():
    #Detector definition
    detector_size = 8 #cm    
    pcb_size = 1 #cm
    sensitive_layer_size = 2 #cm
    total_pcb_number = 4
    total_sensitive_number = 4
    space_between_voxels = 0 #cm
    top_surface_coordinate = 0 #cm
    voxel_size = 0.025 #cm
    
    #Defines the detector object and its characteristics
    detector = PolLibrary.squareDetector(detector_size, pcb_size, sensitive_layer_size, total_pcb_number, total_sensitive_number, space_between_voxels, top_surface_coordinate, voxel_size)
    detector.setEnergyResolution(open('NFTEnergyResolutions.geo', 'r'))
    
    return detector

def createWFMdetector():
    detector_size = 8 #cm    
    pcb_size = 2 #cm
    sensitive_layer_size = 4 #cm
    total_pcb_number = 4
    total_sensitive_number = 4
    space_between_voxels = 0 #cm
    top_surface_coordinate = 8.075 #cm
    voxel_size = 0.5
    
    detector = PolLibrary.squareDetector(detector_size, pcb_size, sensitive_layer_size, total_pcb_number, total_sensitive_number, space_between_voxels, top_surface_coordinate, voxel_size)
    detector.setEnergyResolution(open('WFMEnergyResolutions.geo', 'r'))
    
    return detector

######### MAIN PROGRAM ##########################

#Set-up variable
verbose = True
save_files = True
save_plots = True
reduced = True #If True, only the selected Compton events are stored in the memory, if
#False, all the events in the .tra files are stored and the selection must be made when
#creating the scattering map. If the .tra file is very big, setting reduces = True 
#can significatively reduce the memory consumption of the program

#Define the position of the input files and reads their name. The name of the files
#are read from the unpolarized data input folder. The file of polarized and 
#unpolarized data must have the same name except for the prefix 'Pol' at the 
#beginning of the polarized data file

##### FOR DEBUG ######
#detector_type = 'WFM'
#pol_in_file_name = 'TEST/Pol/PolCrab.inc1.id1.tra.gz'
#unpol_in_file_name = 'TEST/Unpol/UnPolCrab.inc1.id1.tra.gz'
#save_path = 'Save_dump'
#energy_range = [0, np.inf]
######################

detector_type = sys.argv[1]
pol_in_file_name = sys.argv[2]
unpol_in_file_name = sys.argv[3]
save_path = sys.argv[4]
energy_range = [sys.argv[5], sys.argv[6]]


if verbose:
    print('POL FILE: ', pol_in_file_name)
    print('UNPOL FILE: ', unpol_in_file_name)
               

#Creates the folder in which the plots and the other results will be saved 
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(save_path+'/Pol'):
    os.mkdir(save_path+'/Pol')
if not os.path.exists(save_path+'/Unpol'):
    os.mkdir(save_path+'/Unpol')


if detector_type == 'NFT':
    detector = createNFTdetector()
elif detector_type == 'WFM':
    detector = createWFMdetector()

#Process polarized and unpolarized data
unpol_events = pa.read_events(unpol_in_file_name, detector, reduced)

unpol_histogram, y_err_unpol, hist_classes, _ = pa.process_events(unpol_events, unpol_in_file_name, energy_range, detector, 'Unpol', reduced, detector_type, verbose, save_plots, save_path=save_path)
del(unpol_events)

pol_events = pa.read_events(pol_in_file_name, detector, reduced)
pol_histogram, y_err_pol, hist_classes, selected_ev = pa.process_events(pol_events, pol_in_file_name, energy_range, detector, 'Pol', reduced, detector_type, verbose, save_plots, save_path=save_path)
del(pol_events)

if len(unpol_histogram) != 0 and len(pol_histogram) != 0:
    #Fits the unpolarized data
    max_data_unpol = max(unpol_histogram)
    yval_unpol = unpol_histogram 
    angle_fit = [((hist[0]+hist[1])/2)*np.pi/180 for hist in hist_classes]
 
    best_vals, fit_error_one_sigma, x_fit, y_fit, chi_squared = PolMath.PolFitOLS(PolMath.fitSplineUnpol, angle_fit, yval_unpol, y_err_unpol)  
    pa.plotModulationCurve(angle_fit, yval_unpol, y_err_unpol, x_fit, y_fit, chi_squared, save_path + '/Unpol/Q_Plot.png')
    
    #OUTPUT FILE FORMAT
    #   Energy - Voxel Size - Normalization - Norm Error - Q - Q Error - Polarization Angle - Pol Angle Error
    if save_files:
        with open(save_path+'/Unpol_res.txt', 'w') as out_file:
            out_file.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(energy_range, detector.voxel_size, best_vals[0], fit_error_one_sigma[0], best_vals[1], fit_error_one_sigma[1], best_vals[2]*180/np.pi, fit_error_one_sigma[2]*180/np.pi))

    else:
        print("UNPOL {0} {1} {2} {3} {4} {5} {6} {7}\n".format(energy_range, detector.voxel_size, best_vals[0], fit_error_one_sigma[0], best_vals[1], fit_error_one_sigma[1], best_vals[2]*180/np.pi, fit_error_one_sigma[2]*180/np.pi))
   
    #Fits the non corrected polarized data
    max_data_pol = max(pol_histogram)
    yval_pol = pol_histogram 
    
    best_vals, fit_error_one_sigma, x_fit, y_fit, chi_squared = PolMath.PolFitOLS(PolMath.fitSplinePol, angle_fit, yval_pol, y_err_pol)  
    pa.plotModulationCurve(angle_fit, yval_pol, y_err_pol, x_fit, y_fit, chi_squared, save_path + '/Pol/Q_Plot.png')
    #OUTPUT FILE FORMAT
    #   Energy - Voxel Size - Normalization - Norm Error - Q - Q Error - Polarization Angle - Pol Angle Error
    if save_files:
        with open(save_path+'/Pol_Q_no_correction_res.txt', 'w') as out_file:
            out_file.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(energy_range, detector.voxel_size, best_vals[0], fit_error_one_sigma[0], best_vals[1], fit_error_one_sigma[1], best_vals[2]*180/np.pi, fit_error_one_sigma[2]*180/np.pi))
    else:
        print("POL {0} {1} {2} {3} {4} {5} {6} {7}\n".format(energy_range, detector.voxel_size, best_vals[0], fit_error_one_sigma[0], best_vals[1], fit_error_one_sigma[1], best_vals[2]*180/np.pi, fit_error_one_sigma[2]*180/np.pi))

    #Fits the corrected polarized data
    max_unpol_error = np.sqrt(max_data_unpol)

    corrected_val = [(pol_histogram[i]/unpol_histogram[i]*max_data_unpol ) for i in range(len(pol_histogram))] 
    tmp1 = [((pol_histogram[i] / unpol_histogram[i]) *max_unpol_error)**2 for i in range(len(pol_histogram)) ]
    tmp2 = [((max_data_unpol / unpol_histogram[i]) *y_err_pol[i])**2 for i in range(len(pol_histogram)) ]
    tmp3 = [((max_data_unpol*(pol_histogram[i] / unpol_histogram[i]**2)) *y_err_unpol[i])**2 for i in range(len(pol_histogram)) ]

    corrected_val_errors = [np.sqrt(tmp1[i] + tmp2[i] + tmp3[i]) for i in range(len(pol_histogram))]  
    
    best_vals, fit_error_one_sigma, x_fit, y_fit, chi_squared = PolMath.PolFitOLS(PolMath.fitSplinePol, angle_fit, corrected_val,corrected_val_errors)       
    
    pa.plotModulationCurve(angle_fit, corrected_val, corrected_val_errors, x_fit, y_fit, chi_squared, save_path + '/Corrected_Q_Plot.png')
    
    #OUTPUT FILE FORMAT
    #Energy - Voxel Size - Normalization - Norm Error - Q - Q Error - Polarization Angle - Pol Angle Error - Number of Selected Events 
    if save_files:
        with  open(save_path+'/Corrected_Q_res.txt', 'w') as out_file:
            out_file.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(energy_range, detector.voxel_size, best_vals[0], fit_error_one_sigma[0], best_vals[1], fit_error_one_sigma[1], best_vals[2]*180/np.pi, fit_error_one_sigma[2]*180/np.pi))
    else:
        print("CORRECTED {0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(energy_range, detector.voxel_size, best_vals[0], fit_error_one_sigma[0], best_vals[1], fit_error_one_sigma[1], best_vals[2]*180/np.pi, fit_error_one_sigma[2]*180/np.pi, selected_ev))

