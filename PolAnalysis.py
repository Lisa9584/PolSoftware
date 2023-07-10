# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:57:25 2023

@author: Lisa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import gzip
import PolMath 
import PolLibrary 

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

#Read the events from the input file. The input file can be a .tra file. 
#The input of the function is the path of the file to be read,
#the output is a dictionary containing all the events. The keys of the dictionary
#are the event ID in the .tra file, while each entry of the dictionary represents a
#full chain of events as recostructed by REVAN. The "reduced" bool input value tells
#the program if all the Compton events in the .tra files are stored in the memory or if
#only the events which will be used for data analysis must be stored

def read_events(in_file_name, detector, reduced, energy_range = [0, np.inf]):
    input_file_format = in_file_name.split('.')[-1]
    
    if input_file_format == 'gz':
        input_file = gzip.open(in_file_name, 'rt')
    else:    
        input_file = open(in_file_name, 'r')
        
    events = {}
    event_id = 0
    order = 0
    

    #Reads all the lines in a .tra input file and populate the events dictionary.
    
    #Skips the first 7 lines of the .tra file.
    for i in range(7): input_file.readline()
    del(i)
    
    comptonEv = False
    event = None
    for line in input_file:
        if "SE" in line:
            event_id += 1
            comptonEv = False
            if event != None:
                if reduced:
                    keep_event = selectEvent(event, detector, energy_range)
                else: 
                    keep_event = True
            
                if keep_event:
                    events[event_id - 1] = event
                    event = None
                    
        elif 'ET CO' in line:
            comptonEv = True
            event = PolLibrary.comptonEventChain(event_id)
            order = 0
        elif 'CH' in line and comptonEv:
              order += 1
              in_str = line.rstrip('\n').split(" ")
              coord = [float(in_str[2]), float(in_str[3]), float(in_str[4])]
              ener = float(in_str[5])
              event.addEvent(order, PolLibrary.singleComptonEvent(coord, ener))
                  
        
    input_file.close()
    return events

#This function processes the events and create scattering map, energy spectrum plots, 
#modulation plots and others. 
#It takes the following input:
#   -events: a Compton chain events dictionary
#   -in_file_name: the name of the input file. It is used to name the plots
#   -energy: the energy of the generated photons. If the beam is polycromatic, then
#            you still have to define an energy, but it will have no meaning
#   -detector: a Detector class object which represents the detector
#   -pol_type: string value which can be only "Pol" or "Unpol". Used to save the plots
#   -eband: energy band for energy selection of the events. 
#The function gives the following outputs:
#   -histogram: the rebinned modulation histogram
#   -hist_errors: the errors associated to each bin
#   -hist_classes: the minimum and maximum value of the angular sectors
#   -selected_ev: the number of selected events

def process_events(events, in_file_name, energy_range, detector, pol_type, reduced, dect_type = 'NFT', verbose = False, save_file = False, save_path = None):
    
    #Evaluate the scattering matrix and the number of selected events
    if verbose:
        print('Creating the scattering map...')
    scattering_matrix, selected_ev = None, 0
    
    if dect_type == 'NFT':
        scattering_matrix, selected_ev = createScatteringMap_square(events, detector, energy_range, pol_type, reduced, save_file, save_path)
    elif dect_type == 'WFM':
        scattering_matrix, selected_ev = createScatteringMap_hex(events, detector, energy_range, pol_type, reduced, save_file, save_path)
    else:
        raise Exception('Detector name not valid')
    
    histogram, hist_classes, hist_errors = [], [], []
 
    #Evaluate and save the modulation plots based on the scattering map of the selected events
    #if not scattering_matrix.isEmpty():
    if verbose:
        print('Creating the modulation plot...')
    histogram, hist_errors, hist_classes, angle_bin_size = createModPlot(scattering_matrix, dect_type)

    pos = np.arange(len(hist_classes))
    width = 1.0     
  
    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot()
    ax.set_xticks(pos)

    tick = ["{0}°-{1}°".format(hist_class[0], hist_class[1]) for hist_class in hist_classes] 
    ax.set_title('Modulation Plot')
    ax.set_xticklabels(tick)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    #ax.bar(pos, [hist/ (angle_bin_size * PolMath.RAD_TO_DEG) for hist in histogram], width, color = 'grey', edgecolor = 'black')
    ax.bar(pos, histogram / (angle_bin_size * PolMath.RAD_TO_DEG), width, color = 'grey', edgecolor = 'black')
    
    if save_file:
        fig.savefig(save_path + '/' + pol_type + '/Mod_Plot.png')
        with open(save_path + '/' + pol_type + '/Modulation_Plot.txt', 'w') as save_file:
            for i in range(len(histogram)):
                save_file.write('{} {} {}\n'.format((hist_classes[i][0] +  hist_classes[i][1])/2, histogram[i], hist_errors[i]))
            
    plt.close()

    return histogram, hist_errors, hist_classes, selected_ev

############ SCATTERING MAP AND RELATED #########################
#This function creates the scattering map, the energy spectrum of the interactions, the
#angular plots and counts the number of selected events. 
#It takes the following input:
#   -events: a Compton chain events dictionary
#   -energy: the energy of the generated photons. If the beam is polycromatic, then
#            you still have to define an energy, but it will have no meaning
#   -detector: a Detector class object which represents the detector
#   -pol_type: string value which can be only "Pol" or "Unpol". Used to save the plots
#   -eband: energy band for energy selection of the events. 
#The function gives the following outputs:
#   -scattering_matrix: the scattering map of only the selected events
#   -selected_ev: the number of selected events
def createScatteringMap_square(events, detector, energy_range, pol_type, reduced, save_file = False, save_path = None):
    
    def updateScatteringMap(event):
        x = int((event[2].x - event[1].x)/detector.voxel_size)
        y = int((event[2].y - event[1].y)/detector.voxel_size)
        scattering_matrix[x, y] += 1
        polar_angle.append(event.geometricalAngleFirstInt)
        azimuth_angle.append(np.arctan2(y, x))
        en_1_selected.append(event[1].energy)
        en_2_selected.append(event[2].energy)
        
        
    #Define the number of pixels of the detector
    detector_pixels = int(detector.detector_xy_size/detector.voxel_size)
    

    #Define the number of sigma for the single layer selection and the number of pixels
    #for the distance selection
    
    #Defines the scattering matrix and the geometrical filter matrix
    scattering_matrix = PolLibrary.squareScatteringMatrix(detector_pixels)
    
    #Define an array containing all the polar angle, the azimuth angle, the energy of the first 
    #and the second interaction before and after the selection and the number of selected events
    polar_angle, azimuth_angle, selected_ev = [], [], 0 
    en_1_total, en_2_total, en_1_selected, en_2_selected = [], [], [], []
    if reduced:
        #In this case, the event selection was already made when the file was read, so
        #the program only takes the data and populate the scattering map
        selected_ev = len(events)
        for key in events:
            en_1_total.append(events[key][1].energy)
            en_2_total.append(events[key][2].energy)
            updateScatteringMap(events[key])
    else:
        #In this case all the data in the .tra file are collected, so the program checks
        #if the event is good and must be kept before storing it in the scattering matrix
         for key in events:
            good_event = selectEvent(events[key], detector, energy_range)
            en_1_total.append(events[key][1].energy)
            en_2_total.append(events[key][2].energy)
            if good_event:
                updateScatteringMap(events[key])
                selected_ev += 1

    #Saves the plots, if the scattering map is not empty
    if not scattering_matrix.isEmpty() and save_file:
        #######SPOSTARE OGNUNO IN UNA FUNZIONE APPOSITA
        fig = plt.figure(tight_layout = True)
        ax = fig.add_subplot()

        ax.hist(en_1_selected, bins = 50, color = "grey", edgecolor = 'black')
        ax.set_title("Energy Spectrum of the First Interaction (only selected events)")
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Counts")
        
        fig.savefig(save_path + '/' + pol_type + '/En_First_Int.png')
        plt.close()
        
        fig = plt.figure(tight_layout = True)
        ax = fig.add_subplot()
        ax.hist(en_2_selected, bins = 50, color = "grey", edgecolor = 'black')
        ax.set_title("Energy Spectrum of the Second Interaction (only selected events)")
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Counts")
        fig.savefig(save_path + '/' + pol_type + '/En_Second_Int.png')
        plt.close()
    
        fig, ax1 = plt.subplots(tight_layout = True)
        size_scatt_matrix = scattering_matrix.matrix.shape
        pos = ax1.imshow(scattering_matrix.matrix[int(size_scatt_matrix[0]/4):int(size_scatt_matrix[0]*3/4), int(size_scatt_matrix[1]/4):int(size_scatt_matrix[1]*3/4)], cmap='magma',  norm=colors.LogNorm(), interpolation='None')
        #pos = ax1.imshow(scattering_matrix.matrix, cmap='magma',  norm=colors.LogNorm(), interpolation='None')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        #[279:360,279:360]
        fig.colorbar(pos, ax=ax1, label = "Counts", pad = 0.004)
        fig.savefig(save_path + '/' + pol_type + '/Scattering_map.png')
        histPlot(polar_angle, 24, save_file = save_file, file_name = save_path + '/' + pol_type + '/Polar_Angle.png')
        histPlot(azimuth_angle, 24, save_file = save_file, file_name = save_path + '/' + pol_type + '/Azimuth_Angle.png')
        np.savetxt(save_path + '/' + pol_type + '/Scattering_map.txt', scattering_matrix.matrix)
        plt.close()

    return scattering_matrix, selected_ev

def createScatteringMap_hex(events, detector,  energy_range, pol_type, reduced, save_file = False, save_path = None):
    
    def updateScatteringMap(event):
        x = round(event[2].x - event[1].x, 6)
        y = round(event[2].y - event[1].y, 6)
        if (x, y) in scattering_matrix.keys():
            scattering_matrix[(x, y)] += 1
        else:
            scattering_matrix[(x, y)] = 1
        polar_angle.append(event.geometricalAngleFirstInt)
        azimuth_angle.append(np.arctan2(y, x))
        en_1_selected.append(event[1].energy)
        en_2_selected.append(event[2].energy)
    
    #print('Creating Scattering Map...')
    #Define the scattering matrix
    scattering_matrix = {}
    
    #Define an array containing all the polar angle, the azimuth angle, the energy of the first 
    #and the second interaction before and after the selection and the number of selected events
    polar_angle, azimuth_angle, selected_ev = [], [], 0 
    en_1_total, en_2_total, en_1_selected, en_2_selected = [], [], [], []
    
    if reduced:
        #In this case, the event selection was already made when the file was read, so
        #the program only takes the data and populate the scattering map
        selected_ev = len(events)
        for key in events:
            en_1_total.append(events[key][1].energy)
            en_2_total.append(events[key][2].energy)
            updateScatteringMap(events[key])
    else:
        #In this case all the data in the .tra file are collected, so the program checks
        #if the event is good and must be kept before storing it in the scattering matrix
        for key in events:
            good_event = selectEvent(events[key], detector, energy_range)
            en_1_total.append(events[key][1].energy)
            en_2_total.append(events[key][2].energy)
            if good_event:
                updateScatteringMap(events[key])
                selected_ev += 1
      
    #Saves the plots, if the scattering map is not empty
    #print(scattering_matrix)
    if not len(scattering_matrix) == 0 and save_file:
        fig = plt.figure(tight_layout = True)
        ax = fig.add_subplot()

        #ax.hist(en_1_total, bins = 50, color = "grey", label = "Total")
        ax.hist(en_1_selected, bins = 50, color = "grey", edgecolor = 'black')
        ax.set_title("Energy Spectrum of the First Interaction (only selected events)")
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Counts")
        
        fig.savefig(save_path + '/' + pol_type + '/En_First_Int.png')
        plt.close()
        
        
        fig = plt.figure(tight_layout = True)
        ax = fig.add_subplot()
        #ax.hist(en_2_total, bins = 50, color = "grey", label = "Total")
        ax.hist(en_2_selected, bins = 50, color = "grey", edgecolor = 'black')
        ax.set_title("Energy Spectrum of the Second Interaction (only selected events)")
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("Counts")
        fig.savefig(save_path + '/' + pol_type + '/En_Second_Int.png')
        plt.close()
        
       
        PlotScatteringMap_hex(scattering_matrix, save_path + '/' + pol_type + '/Scattering_map.png')
        histPlot(polar_angle, 24, save_path + '/' + pol_type + '/Polar_Angle.png', 'Polar Angle')
        histPlot(azimuth_angle, 24, save_path + '/' + pol_type + '/Azimuth_Angle.png', 'Azimuthal Angle')

    return scattering_matrix, selected_ev

#This method is used to plot the hexagonal scattering matrix. The input are the scattering 
#matrix data and a path indicating where the image must be saved
def PlotScatteringMap_hex(scattering_matrix, savepath):
    #Define the used colormap
    from matplotlib.patches import RegularPolygon
    import matplotlib 
    
    cmap = matplotlib.cm.get_cmap('magma')
    x_center, y_center, col = [], [], []
    max_count = max(scattering_matrix.values())
    #print(max(scattering_matrix.values()), min(scattering_matrix.values()))
    #Create the plot

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    
    #Create two arrays containing the x and y position of the cells of the scattering map
    #and a third array containing the color associated to the number of events in each cell
    for key in scattering_matrix:
        x_center.append(key[0])
        y_center.append(key[1])
        col.append(cmap(np.log10(scattering_matrix[key])))
        
    #Create an hexagon for each point of the scattering map
    for i in range(len(x_center)):
        pol = RegularPolygon((x_center[i], y_center[i]), numVertices = 6, radius = 0.25, orientation = np.pi/2, color = col[i])
        ax.add_patch(pol)
        pos = ax.scatter(x_center[i], y_center[i], color = col[i], s = 1, marker = '+')
    
    ax.set_facecolor('white')
    ax.set_title('Scattering Map')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin = min(scattering_matrix.values()), vmax = max(scattering_matrix.values())), cmap=cmap), ax=ax, label = 'Counts')
    
    fig.savefig(savepath)
    plt.close()
#This method tells if a Compton event must be kept or not. The input are a single event
#and the "detector" object, which rappresents the characteristics of the detector. The 
#method returns a boolean value indicating if the event is good (True) or not (False) and the
#x, y and z coordinates of the event
def selectEvent(event, detector, energy_range):
    good = False
    voxel_size = detector.voxel_size
    n_sigma, num_of_pix = 5, 2
    selection_1 = event.multiplicity >= 2
    selection_2 = event.totalEnergy >= energy_range[0] and event.totalEnergy <= energy_range[1]
    
    if selection_1 and selection_2: 
        #Evaluate the distance from the origin of the event
        distance = np.sqrt((event[2].x - event[1].x)**2 + (event[2].y - event[1].y)**2 + (event[2].z - event[1].z)**2)
        selection_3 = distance > num_of_pix * voxel_size
        
        #Apply distance and eventually further geometrical selections
        if selection_3:  
            #Evaluate the Compton angle and the expected energy of the first and second 
            #interaction. 
            theta = event.geometricalAngleFirstInt
            total_en = event[1].energy + event[2].energy
                
            e2_expected = total_en/(1 + total_en/511*(1-np.cos(theta*np.pi/180)))
            #If the expected energy of the first interaction is below minimum threeshold of the 
            #detector (5 keV), set the expected minimum interaction equal to 5 keV.
            e1_expected = max(total_en - e2_expected, 5.0)
            
            #Energy resolution of the instrument at the energy of the first and second event.
            res_1 =  detector.energy_res[round(e1_expected,0)]
            res_2 = detector.energy_res[round(e2_expected,0)]
            selection_4 = abs(event[1].energy -  e1_expected) < n_sigma * res_1 and abs( event[2].energy -  e2_expected) <  n_sigma * res_2
            if  selection_4:
                #Mark the event as a good event
                good = True
    
    return good



############ MODULATION PLOT AND RELATED #########################
#This function creates the modulation plot starting from a given scattering map. 
#It takes the following input:
#   -scattering_matrix: the scattering matrix of the events
#   -energy: the energy of the generated photons. 
#The function gives the following outputs:
#   -histogram: the rebinned modulation histogram
#   -hist_errors: the errors associated to each bin
#   -hist_classes: the minimum and maximum value of the angular sectors
#   -angle_bin_size: the angular size of the sectors
def createModPlot(scattering_matrix, dect_type):
    
    #Define the number of angular bins and evaluate the angular size
    bin_number = 24
    angle_bin_size = 2 * np.pi / bin_number
    #Define the number of subpixels to be used in the evaluation of the pixel fraction
    #inside a bin
    subpix_num = 10    
    histogram, hist_errors = [], []
    
    #Define the angular classes for the modulation histogram
    angles = np.linspace(0,360,bin_number+1)
    hist_classes = np.stack([angles[:len(angles)-1], angles[1:]],1)
    #Evaluate the counts inside each angular bin
    for i in range(bin_number):
        if dect_type == 'NFT':
            bin_res = evaluate_bin_square(angle_bin_size, subpix_num, scattering_matrix, i)
        elif dect_type == 'WFM':
            bin_res = evaluate_bin_hex(angle_bin_size, subpix_num, scattering_matrix, i)
        else:
            bin_res = (0,0)
            raise Exception('Detector name not valid')
        histogram.append(bin_res[0])
        hist_errors.append(bin_res[1])
        
    histogram = np.array(histogram) 
    hist_errors = np.array(hist_errors)
    
    return histogram, hist_errors, hist_classes, angle_bin_size


#This function evaluates the number of counts inside an angular bin taking in account
#the fraction of each pixel which falls inside a given bin.
#It takes the following input:
#   -angle_bin_size: the angular size of bins    
#   -subpix_num: number of subpixels to be used in the evaluation of the pixel fraction
#   -scattering_matrix: the scattering matrix of the events
#   -i: the ordinal number of the bin to be evaluated. 
def evaluate_bin_square(angle_bin_size, subpix_num, scattering_matrix, i):
    #Variables containing the number of counts in the angular bin and the error
    bin_val = 0
    bin_err = 0
    #Evaluate the minimum and maximum angle of the angular beam
    theta_min = int(i *angle_bin_size * PolMath.RAD_TO_DEG)
    theta_max = int((i+1) * angle_bin_size * PolMath.RAD_TO_DEG)
    #Check all the pixels of the scattering matrix
    for x in scattering_matrix.x_range:
        for y in scattering_matrix.y_range:
            #Only perform the calculation for non empty pixels
            if  scattering_matrix[x,y] != 0:
                ####### WITHOUT PIXEL FRACTION #######
                angle = PolMath.convertAngle((np.arctan2(y, x)))*PolMath.RAD_TO_DEG
                if (theta_min <= angle < theta_max):
                    bin_val += scattering_matrix[x,y]
                #    bin_err += np.sqrt(scattering_matrix[x,y])
                
                ####### WITH PIXEL FRACTION #######
                #Evaluate the fraction of the [x,y] pixel which falls inside a given
                #angular sector and add to the total number of counts in the bin. May
                #significatively slow down the evaluation if the numbers of pixels or 
                #subpixels are very high
                #frac = computeFraction(x, y, theta_min, theta_max, subpix_num)
                #bin_val += scattering_matrix[x,y] * frac
                #bin_err += np.sqrt(scattering_matrix[x,y]) * frac
    bin_err += np.sqrt(bin_val)
    return (bin_val, bin_err)

def evaluate_bin_hex(angle_bin_size, subpix_num, scattering_matrix, i):
    #Variables containing the number of counts in the angular bin and the error
    bin_val = 0
    bin_err = 0
    #Evaluate the minimum and maximum angle of the angular beam
    theta_min = int(i *angle_bin_size * PolMath.RAD_TO_DEG) #- 5#angle_bin_size* 180/np.pi/4
    theta_max = int((i+1) * angle_bin_size * PolMath.RAD_TO_DEG)
    
    #Check all the pixels of the scattering matrix    
    for key in scattering_matrix:
        x = key[0]
        y = key[1]
        angle = PolMath.convertAngle((np.arctan2(y, x))) * PolMath.RAD_TO_DEG #- 5#angle_bin_size* 180/np.pi/4
        if (theta_min <= angle < theta_max):
           bin_val += scattering_matrix[key]
           bin_err += np.sqrt(scattering_matrix[key])
        
    return (bin_val, bin_err)

#This function evaluates the fraction of a pixel which falls inside a given angular bin.
#It takes the following input:
#   -x_pix, y_pix: x and y coordinate of the pixel    
#   -theta_min, theta_max: number of subpixels to be used in the evaluation of the pixel fraction
#   -subpix_number: number of subpixels to be used in the evaluation of the pixel fraction
#The function divides the pixel into a finer, discrete grid and counts how many of those
#subpixels fall inside the angular bin. This results divided by the total number of subpixels
#gives the pixel fraction
def computeFraction(x_pix, y_pix, theta_min, theta_max, subpix_number):
  pix_count = 0
  for x in range(-int(subpix_number/2) + 1, int(subpix_number/2) + 1):
    for y in range(-int(subpix_number/2) + 1, int(subpix_number/2) + 1):
      sub_x = x_pix + x * 1/subpix_number
      sub_y = y_pix + y * 1/subpix_number
      angle = PolMath.convertAngle((np.arctan2(sub_y, sub_x)))*PolMath.DEG_TO_RAD
      if theta_min <= angle < theta_max:
        pix_count += 1

  return pix_count / (subpix_number**2)

######### PLOTS #####################################

#Function to plot and save an histogram given some input data, the number of bins (nbin)
#and the name of the file 
def histPlot(data, nbin, save_file = False, file_name = None):
    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot()
    n, bins, patches = ax.hist(data, bins = nbin)
    ax.set_xlabel('Azitmuthal Angle [deg]')
    ax.set_ylabel('Counts/degree')
    if save_file:
        fig.savefig(file_name)
    plt.close()

#Function which plots and saves the energy spectrum and its gaussian fit. The inputs are
#the energy data, the x variable data used to fit, the best fit paramenter, the 
#number of bins (nbin) and the name of the file to be saved
def energyResPlot(energy_data, x_data, best_fit, nbin = 'auto', save_file = False, file_name = None):
    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot()
    ax.hist(energy_data, bins = nbin)
    yfit = PolMath.gaussian(x_data, *best_fit)
    ax.plot(x_data, yfit)
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel('Counts')
    if save_file:
        fig.savefig(file_name)
    plt.close()
    
def plotModulationCurve(xdata, ydata, yerrors, x_fit, y_fit, save_file = False, file_name = None):
    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot()
    ax.scatter(xdata, ydata, color = 'b')
    ax.errorbar(xdata, ydata, yerr = yerrors, linestyle = "None", color = 'b')
    ax.plot(x_fit, y_fit, color = 'black')
    ax.set_xlabel('Azitmuthal Angle [deg]')
    ax.set_ylabel('Counts')
    if save_file:
        fig.savefig(file_name)
    plt.close()