import numpy as np
import PolMath
import warnings

__all__ = ['singleComptonEvent', 'comptonEventChain', 'squareScatteringMatrix', 'squareFilterMatrix', 'squareDetector']

#Class describing a single compton events. The attributes of the class are:
#   -coordinates: x, y, and z coordinate of the compotn interaction
#   -energy: the energy of the interaction    

class singleComptonEvent:
  coordinates = np.array([])
  energy = 0

  #Constructor for the class
  def __init__(self, coordinates, Energy):
      self.coordinates = np.array(coordinates)
      self.energy = Energy

  #Return the x coordinate of the interaction
  @property
  def x(self):
    return self.coordinates[0]

  #Return the y coordinate of the interaction
  @property
  def y(self):
    return self.coordinates[1]

  #Return the z coordinate of the interaction
  @property
  def z(self):
    return self.coordinates[2]

  #Assign a new coordinate vector to the event
  def change_coordinates(self, new_coord):
    self.coordinates = np.copy(new_coord)
  
  def __str__(self):
    return "Compton Event - x: {0}, y: {1}, z: {2} ; Energy: {3}".format(self.coordinates[0], self.coordinates[1], self.coordinates[2], self.energy)

  def __repr__(self):
    return "{0};{1};{2};{3}".format(self.coordinates[0], self.coordinates[1], self.coordinates[2], self.energy)

#Class describing a chain of compont events. The attributes of the class are:
#   -events: A dictionary containing all the events in the chain. The keys of the 
#            dictionary are the ordinal number of the interaction, the values are
#            the single compton events, represented by singleComptonEvent objects
#   -event_id: a numerical identifier for the event.
class comptonEventChain:
    events = {} 
    event_id = 0
    
    #Constructor of the class
    def __init__(self, event_id):
        self.event_id = event_id
        self.events = {}

    #Returns the total energy deposited by the event chain
    @property
    def totalEnergy(self):
      return sum([self.events[key].energy for key in self.events])
      
    #Returns the geometrical angle associated to the first interaction
    @property
    def geometricalAngleFirstInt(self):
        x = self.events[2].x - self.events[1].x
        y = self.events[2].y - self.events[1].y
        z = self.events[2].z - self.events[1].z
        #theta = np.arctan(np.sqrt(x**2 + y**2)/z)*180/np.pi
        #if theta < 0: theta += 180 
        theta = np.arctan2(round(np.sqrt(x**2 + y**2), 10),round(-z,10))*PolMath.RAD_TO_DEG
        
        return theta
    
    #Returns the total number of Compton events in the cahin
    @property 
    def multiplicity(self):
        return len(self.events.keys())

    #Function which adds a new event to the chain. The inputs are:
    #   -order: the ordinal number of the event
    #   -event: a singleComptonEvent object representing one single Compton interaction
    def addEvent(self, order, event):
        self.events[order] = event  
    
    #Left to be eventually updated and used in the future
    #def vectorTranslationFirstInt(self):
    #    translation_vector = self.events[1].coordinates
    #    new_event_1 = singleComptonEvent([self.events[1].coordinates[i] - translation_vector[i] for i in (0,1,2)], self.events[1].energy )
    #    new_event_2 = singleComptonEvent([self.events[2].coordinates[i] - translation_vector[i] for i in (0,1,2)], self.events[2].energy )
    #    new_double_event = comptonEventChain(self.event_id)
    #    new_double_event.addEvent(1, new_event_1 )
    #    new_double_event.addEvent(2, new_event_2 )
    #    return new_double_event
    
    def __str__(self):
        return str(self.events)
    
    def __repr__(self):
        return str(self.events)
    
    def __getitem__(self, order):
      return self.events[order]
  
    def __setitem__(self, order, val):
      self.events[order] = val

#Internal class representing the structure of a xy pixelized scattering map for 
#a square voxel detector.
#The attributes of the class are:
#   -matrix: a 2n-1 x 2n-1 matrix, where n is the number of pixels in the detector.
#            Each entry of the matrix represents a single square pixel and the origin
#            of the reference frame is placed in the central pixel of the map.
#   -detector_pixels: the number n of pixels of the detector
#   -x_min, x_max: the minimum and the maximum value of the x coordinate of the map
#   -y_min, y_max: the minimum and the maximum value of the y coordinate of the map
#   -x_range, y_range: the  range of values for the x and y coordinates
class _squareDetectorMatrix:
  matrix = []
  detector_pixels = 0
  x_min, x_max, y_min, y_max = 0, 0, 0, 0
  x_range, y_range = '', ''
  
  #Constructor of the class
  def __init__(self, detector_pixels, np_type):
      self.matrix = np.zeros((2*detector_pixels-1, 2*detector_pixels-1), dtype = np_type)
      self.detector_pixels = detector_pixels
      self.x_min = - detector_pixels 
      self.y_min = - detector_pixels 
      self.x_max = detector_pixels
      self.y_max = detector_pixels
      self.x_range = range(self.x_min, self.x_max )
      self.y_range = range(self.y_min, self.y_max )

  #Gets one element from the matrix using the scattering map coordinate system
  def __getitem__(self, coordinates):
      x, y = coordinates
      return self.matrix[ y + (self.detector_pixels-1) , x + (self.detector_pixels-1)]
  
  #Sets one element from the matrix using the scattering map coordinate system
  def __setitem__(self, coordinates, val):
      x, y = coordinates
      self.matrix[ y + (self.detector_pixels-1) , x + (self.detector_pixels-1)] = val
      
  def __str__(self):
        return str(self.matrix)
    
  def __repr__(self):
        return self.matrix

#Class representing a 2D scattering map for a square voxel detector.
class squareScatteringMatrix(_squareDetectorMatrix):
  
  def __init__(self, detector_pixels):
    super().__init__(detector_pixels, "int32")
    
  #Checks if the scattering map is empty
  def isEmpty(self):
    return sum(map(sum, self.matrix)) == 0

#Class representing a 2D geometrical filter which can be applied on a square voxel
#scattering map. The filter is 2D matrix of boolean value which can be used to flag
#which pixels have to be included in the data analysis (True) and which not (False)
class squareFilterMatrix(_squareDetectorMatrix):

    def __init__(self, detector_pixels):
        super().__init__(detector_pixels, "bool")

    #A square shaped filter, centered on the (0,0) pixel of the scattering map and
    #with a given edge length. All the pixels INSIDE the square are flagged False
    def squareFilter(self, edge):
        for x in self.x_range:
            for y in self.y_range:
                self[x,y] = not (abs(x) < edge and abs(y) < edge)

    #A square shaped filter, centered on the (0,0) pixel of the scattering map and
    #with a given edge length. All the pixels OUTSIDE the square are flagged False
    def inverseSquareFilter(self, edge):
        for x in self.x_range:
            for y in self.y_range:
                self[x,y] = abs(x) < edge and abs(y) < edge
    
    #A rectangle shaped filter, centered on the (0,0) pixel of the scattering map and
    #with a given x and y edge length. All the pixels OUTSIDE the square are flagged False
    def rectangleFilter(self, x_edge, y_edge):
        for x in self.x_range:
            for y in self.y_range:
                self[x,y] = not (abs(x) < x_edge and abs(y) < y_edge)

    #A circular shaped filter, centered on the (0,0) pixel of the scattering map and
    #with a chosen radius. All the pixels INSIDE the square are flagged False
    def circularFilter(self, radius):
        for x in self.x_range:
            for y in self.y_range:
                self[x,y] = np.sqrt(x**2 + y**2) >= radius
    
    #A circular shaped filter, centered on the (0,0) pixel of the scattering map and
    #with a chosen radius. All the pixels OUTSIDE the square are flagged False
    def inverseCircularFilter(self, radius):
        for x in self.x_range:
            for y in self.y_range:
                self[x,y] = np.sqrt(x**2 + y**2) < radius
    
    #An angular sector shaped filter, centered on the (0,0) pixel of the scattering map.
    #All the pixels OUTSIDE the minimum and the maximum angle of the sector are flagged as
    #False
    def sectorFilter(self, min_angle, max_angle):
        for x in self.x_range:
            for y in self.y_range:
                self[x,y] = min_angle <= PolMath.convertAngle(np.arctan2(y,x))*180/np.pi  < max_angle 

    #Flags as False a list of single points to be excluded
    def singlePointsFilter(self, point_list):
        self.matrix = np.full((2*self.detector_pixels, 2*self.detector_pixels), True)
        for point in point_list:
            self[point] = False
                         
    #Removes all the filters
    def noFilter(self):
        self.matrix = np.full((2*self.detector_pixels, 2*self.detector_pixels), True)
        
    #Makes the entry by entry boolean multiplication of a list of masks and return
    #the final filter matrix
    @staticmethod
    def multiplyMask(mask_list):
        detector_pixels = mask_list[0].detector_pixels
        new_mask = squareFilterMatrix(detector_pixels)
        new_mask.noFilter()
        for x in new_mask.x_range:
            for y in new_mask.y_range:
                for mask in mask_list:
                    new_mask[x,y] = new_mask[x,y] and mask[x,y]
        return new_mask
    
    #Makes the entry by entry boolean sum of a list of masks and return
    #the final filter matrix
    @staticmethod
    def sumMask(mask_list):
        detector_pixels = mask_list[0].detector_pixels
        new_mask = squareFilterMatrix(detector_pixels)
        for x in new_mask.x_range:
            for y in new_mask.y_range:
                for mask in mask_list:
                    new_mask[x,y] = new_mask[x,y] or mask[x,y]
        return new_mask
    
#Class defining a detector based on square shaped voxels. The class is strongly
#based on the current design of NFT. To be expanded and updated
class squareDetector:
    pcb_size = 0
    sensitive_layer_size = 0
    total_pcb_number = 0
    total_sensitive_number = 0
    space_between_voxels = 0
    top_surface_coordinate = 0 
    voxel_size = 0
    voxel_layers = {}
    sensitive_layers_start_centroid = []
    voxels_per_layer = {}
    detector_xy_size = 0
    energy_res = {}
    

    def __init__(self, detector_xy_size = 0, pcb_size = 0, sensitive_layer_size = 0, total_pcb_number = 0, total_sensitive_number = 0, space_between_voxels = 0, top_surface_coordinate = 0, voxel_size = 0 ):
        self.detector_xy_size = detector_xy_size
        self.pcb_size = pcb_size
        self.sensitive_layer_size = sensitive_layer_size
        self.total_pcb_number = total_pcb_number
        self.total_sensitive_number = total_sensitive_number
        self.space_between_voxels = space_between_voxels
        self.top_surface_coordinate = top_surface_coordinate
        self.voxel_size = voxel_size
        
        first_centroid = round(top_surface_coordinate - voxel_size/2 ,4)
        
        self.sensitive_layers_start_centroid = [round(first_centroid - (i-1) * (self.sensitive_layer_size + self.pcb_size),4) for i in range(1, self.total_sensitive_number + 1) ]
        #print(self.sensitive_layers_start_centroid)
        total_voxels_per_layer = int(self.sensitive_layer_size/voxel_size)
        for i in range(len(self.sensitive_layers_start_centroid)):
            for j in range(total_voxels_per_layer):
                self.voxel_layers[i *total_voxels_per_layer + j ] = round(self.sensitive_layers_start_centroid[i] - j * voxel_size, 4)
        
        for i in range(self.total_sensitive_number):
            self.voxels_per_layer[i+1] = []
            for j in range(total_voxels_per_layer):
                self.voxels_per_layer[i+1].append(round(self.sensitive_layers_start_centroid[i] - j * self.voxel_size,4))
        #print(self.voxel_layers)
    
    def findSensLayer(self, z):
        res = 0
        
        for key in self.voxels_per_layer:
            #print(z in self.voxels_per_layer[key])
            if z in self.voxels_per_layer[key]:
                res = key
        
        return res
    
    def setEnergyResolution(self, infile):
        
        for line in infile:
            if "CZTDetector_1" in line:
                tmp = line.split(' ')
                self.energy_res[float(tmp[2])] = float(tmp[4])
                
                
################## HEX DETECTORS WIP #####################################

# class hexScatteringMatrix:
    
#     hex_map = {}
#     row_num, col_num = 0, 0
#     x_min, x_max, y_min, y_max = 0, 0, 0, 0
#     x_range, y_range = '', ''
    
#     def __init__(self, col_num, row_num):
#         self.row_num = row_num
#         self.col_num = col_num
        
#     def addEvent(self, coord):
#         #coord = (x, y)
#         #print(coord)
#         #print(list(self.hex_map.keys()))
#         if not coord in self.hex_map.keys():
#             self.hex_map[coord] = 1
#         else:
#             self.hex_map[coord] += 1
            
#     def isEmpty(self):
#         return len(hex_map) == 0
            
        
    
    


class _hexDetectorMatrix:
  matrix = []
  row_num, col_num = 0, 0
  x_min, x_max, y_min, y_max = 0, 0, 0, 0
  x_range, y_range = '', ''
  
  #Constructor of the class
  def __init__(self, col_num, row_num, np_type):
      self.matrix = np.zeros((2*col_num, 2*row_num-1), dtype = np_type)
      self.row_num = row_num
      self.col_num = col_num
      self.x_min = - row_num 
      self.y_min = - col_num 
      self.x_max = row_num
      self.y_max = col_num
      self.x_range = range(self.x_min+1, self.x_max)
      self.y_range = range(self.y_min+1, self.y_max)
      #print(self.x_range)
      #print(self.y_range)
      

  #Gets one element from the matrix using the scattering map coordinate system
  def __getitem__(self, coordinates):
      x, y = coordinates
      return self.matrix[ y + (self.col_num-1) , x + (self.row_num-1)]
  
  #Sets one element from the matrix using the scattering map coordinate system
  def __setitem__(self, coordinates, val):
      x, y = coordinates
      #print(x,y)
      if self[x,y] != -1:
          self.matrix[y + (self.col_num-1) , x + (self.row_num-1)] = val
      else:
          warnings.warn('Warning: Attempted access to an invalid entry')
        
  def __str__(self):
        return str(self.matrix)
    
  def __repr__(self):
        return self.matrix

#Class representing a 2D scattering map for a square voxel detector.
class hexScatteringMatrix(_hexDetectorMatrix):
  
  def __init__(self, col_num, row_num):
    super().__init__(col_num, row_num, "int32")
    
    for i in self.x_range:
          if i % 2 == 0:
              self[i, self.y_max] = -1
    
  #Checks if the scattering map is empty
  def isEmpty(self):
    return sum(map(sum, self.matrix)) == 0

#Class representing a 2D geometrical filter which can be applied on a square voxel
#scattering map. The filter is 2D matrix of boolean value which can be used to flag
#which pixels have to be included in the data analysis (True) and which not (False)
class hexFilterMatrix(_hexDetectorMatrix):

    def __init__(self, col_num, row_num):
        super().__init__(col_num, row_num, "bool")

    # #A square shaped filter, centered on the (0,0) pixel of the scattering map and
    # #with a given edge length. All the pixels INSIDE the square are flagged False
    # def squareFilter(self, edge):
    #     for x in self.x_range:
    #         for y in self.y_range:
    #             self[x,y] = not (abs(x) < edge and abs(y) < edge)

    # #A square shaped filter, centered on the (0,0) pixel of the scattering map and
    # #with a given edge length. All the pixels OUTSIDE the square are flagged False
    # def inverseSquareFilter(self, edge):
    #     for x in self.x_range:
    #         for y in self.y_range:
    #             self[x,y] = abs(x) < edge and abs(y) < edge
    
    # #A rectangle shaped filter, centered on the (0,0) pixel of the scattering map and
    # #with a given x and y edge length. All the pixels OUTSIDE the square are flagged False
    # def rectangleFilter(self, x_edge, y_edge):
    #     for x in self.x_range:
    #         for y in self.y_range:
    #             self[x,y] = not (abs(x) < x_edge and abs(y) < y_edge)

    # #A circular shaped filter, centered on the (0,0) pixel of the scattering map and
    # #with a chosen radius. All the pixels INSIDE the square are flagged False
    # def circularFilter(self, radius):
    #     for x in self.x_range:
    #         for y in self.y_range:
    #             self[x,y] = np.sqrt(x**2 + y**2) >= radius
    
    # #A circular shaped filter, centered on the (0,0) pixel of the scattering map and
    # #with a chosen radius. All the pixels OUTSIDE the square are flagged False
    # def inverseCircularFilter(self, radius):
    #     for x in self.x_range:
    #         for y in self.y_range:
    #             self[x,y] = np.sqrt(x**2 + y**2) < radius
    
    # #An angular sector shaped filter, centered on the (0,0) pixel of the scattering map.
    # #All the pixels OUTSIDE the minimum and the maximum angle of the sector are flagged as
    # #False
    # def sectorFilter(self, min_angle, max_angle):
    #     for x in self.x_range:
    #         for y in self.y_range:
    #             self[x,y] = min_angle <= convertAngle(np.arctan2(y,x))*180/np.pi  < max_angle 

    #Flags as False a list of single points to be excluded
    def singlePointsFilter(self, point_list):
        self.matrix = np.full((2*self.col_num, 2*self.row_num-1), True)
        for point in point_list:
            self[point] = False
                         
    #Removes all the filters
    def noFilter(self):
        self.matrix = np.full((2*self.detector_pixels, 2*self.detector_pixels), True)
        
    #Makes the entry by entry boolean multiplication of a list of masks and return
    #the final filter matrix
    @staticmethod
    def multiplyMask(mask_list):
        detector_pixels = mask_list[0].detector_pixels
        new_mask = squareFilterMatrix(detector_pixels)
        new_mask.noFilter()
        for x in new_mask.x_range:
            for y in new_mask.y_range:
                for mask in mask_list:
                    new_mask[x,y] = new_mask[x,y] and mask[x,y]
        return new_mask
    
    #Makes the entry by entry boolean sum of a list of masks and return
    #the final filter matrix
    @staticmethod
    def sumMask(mask_list):
        detector_pixels = mask_list[0].detector_pixels
        new_mask = squareFilterMatrix(detector_pixels)
        for x in new_mask.x_range:
            for y in new_mask.y_range:
                for mask in mask_list:
                    new_mask[x,y] = new_mask[x,y] or mask[x,y]
        return new_mask
                     
        