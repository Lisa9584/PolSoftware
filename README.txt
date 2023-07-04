PolSoftware - V1
 
1 - Files content
-PolSoftware.py: main script to run the program
-PolAnalysis.py: library containing all the data analysis function
-PolLibrary.py: library containing the definitions of data structure and functions used to represent 		the Compton events and the detectors
-PolMath.py: library containing auxiliary math functions and similar
-NFTEnergyResolutions.geo: file containing the enrgy resolution per energy information for NFT
-WFMEnergyResolutions.geo: file containing the enrgy resolution per energy information for WFM

2 - Run the program (basic)
The program can be run by terminal. To run the code:
python PolSoftware.py detector pol_file_name unpol_file_name save_path min_energy max_energy
-detector: Must be either "NFT" or "WFM". Indicates which detector must be considered
-pol_file_name: .tra or .tra.gz file containing the polarized events information
-unpol_file_name: .tra or .tra.gz file containing the unpolarized events information
-save_path: path at which the program must save output plot or files
-min_energy: minimum total energy of the event to be considered for the event selection
-max_energy: maximum total energy of the event to be considered for the event selection

4 set-up variable are present. To change them, you need to open the script and manually change their value:
1-verbose (default = True): if True, print more text output while running the code
2-save_files (default = True): if True, saves the output of the Q and selected event evaluation on a file. If false, the results are printed on screen
3-save_plots (default = True): if True, saves a series of plot associated with the data analisis (scattering map, modulation plot etc.)
4-reduced (default = True): if True, stores in the memory ONLY the event which passed the event selection. It is useful to reduce memory consumption when the input file is very big


3 - Change advanced configurations

----To change the number of bin for the modulation plot: open the PolAnalysis.py file with an editor.    
Go to the "createModPlot" function and change the value of bin_number to the desired value (default = 24)

----To change event selection criteria: open the PolAnalysis.py file with an editor and go to the "selectEvent" function. All the event selection is handled in this function, so if you want to change anything about the event selection, you can change the stuff here

4 - Contact me in case anything is needed: Lisa Ferro - frrlsi@unife.it
