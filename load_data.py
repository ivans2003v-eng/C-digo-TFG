import re
import numpy as np
from scipy.io import loadmat

class Register:
    """
    This class represents a register that stores information about seizures detected in a patient
    
    Attributes:
     -  name (str): The name of the file associated with the register.
     -  fs (int): The sampling frequency of the data.
     -  nseizures (int): The number of seizures recorded in that file.
     -  seizures (list): A list storing the start and end times of seizures.
     -  channels (list): A list of channels used in the register.
     -  ictaltime (int): The total duration of seizures recorded.
    """
    def __init__(self, name, fs, nseizures):
        self.name = name
        self.fs = fs
        self.nseizures = nseizures
        self.seizures = []
        self.channels = []
        self.ictaltime = 0
            
    def addSeizure (self, start, end):       # añade una convulsión
        """ 
        Adds a seizure event to the register 
        """
        self.ictaltime += end - start
        seizure = [start, end]
        self.seizures.append(seizure)
        
        
class EDFReader:
    """
    This class reads and process an EDF (European Data Format) file to extract information.
    It processes data related to the file's sampling rate, channels and seizures.

    Attributes:
     - annotation (str): the path to the file containing the information.
     - registers (dict): a dictionary storing Register objects keyed by their file names.
     - channels_dict (dict): a dictionary storing the count of occurrences of each channel.
     - nmontages (int): the number of different montages of channels in the file.
     - fs (int): the sampling frequency of the data.
     - common_channels (list): a list of channels that appear in all the montages.
     - channel_index (dict): a dictionary that maps channel indices to channel names.
    """
    def __init__(self, annotation, database_variables):
        self.annotation = annotation
        self.database_variables = database_variables
        self.registers = {}
        self.channels_dict = {}
        self.nmontages = 1
        self.fs = None
        self.common_channels = []
        self.channel_index = {}
        
        self.__process_file()
        self.__process_channels()
    
    def __process_file(self): 
        """
        Processes the annotation file line by line to extract relevant information such as:
        - Data Sampling Rate (fs)
        - Channel names and their occurrences
        - Number of montages
        - File names and seizures within each file

        This method obtains the registers, channels_dict, fs, and nmontages attributes.
        """
        if self.database_variables['nseizures'] == None: 
            files_with_seizures = {}
            with open(self.annotation) as f:
                for line in f:
                    if (self.database_variables['file_name'] in line):

                        name = line.split()[-1].strip()

                        # Si el archivo ya está en el diccionario, aumentar su contador
                        if name in files_with_seizures:
                            files_with_seizures[name] += 1
                        else:
                            # Si es la primera vez que se lee, inicializar con 1 crisis
                            files_with_seizures[name] = 1
                        nseizures = files_with_seizures[name]
                        
        with open(self.annotation) as f:
            for line in f:
                line_str = line
                if self.database_variables['fs'] in line:
                    line = line.split()              # se divide en palabras
                    self.fs = int(line[3])           # se extrae la freq de muestreo (4º posicion de la línea)
                
                if (self.database_variables['channels'] in line):
                    line = line.split()
                    channel = line[2]                # se extrae el nombre del canal (3º palabra)
                    if channel in self.channels_dict:                                   
                        self.channels_dict.update({channel: self.channels_dict[channel]+1})   # si el canal ya existe incrementa su contador
                    else:
                        self.channels_dict[channel] = 1

                if self.database_variables.get('nmontages') and self.database_variables['nmontages'] in line:
                    self.nmontages += 1
                
                if (self.database_variables['file_name'] in line):
                    name = line.split()[2]           # se extrae el nombre del archivo (3º palabra)
                    if self.database_variables['nseizures'] == None:
                        register = Register(name, self.fs, nseizures)
                        count_seizures = 0
                
                if self.database_variables.get('nseizures') and self.database_variables['nseizures'] in line:
                    nseizures = int(line.split()[5])              # MODIFICAR ESTE CINCO PARA Q VALGA PARA CUALQUIER FORMATO
                    register = Register(name, self.fs, nseizures) # crea un objeto Register
                    count_seizures = 0
                
                if (self.database_variables['file_start'] in line) and self.database_variables['tiempo_acumulado']:
                    file_start = line.split()[3]   
                    file_start = self.__process_time(file_start)    

                if (self.database_variables['file_end'] in line) and self.database_variables['tiempo_acumulado']:
                    file_end = line.split()[3]   
                    file_end = self.__process_time(file_end)   

                if (re.match(self.database_variables['start_seizure'], line_str)):
                    if (line_str.split()[3] == self.database_variables['start_seizure'].split()[-1]):
                        start = line_str.split()[4]
                        start = self.__process_time(start)
                        if self.database_variables['tiempo_acumulado']:
                            start = start - file_start
                    else: 
                        start = line_str.split()[3]
                        start = self.__process_time(start)
                        if self.database_variables['tiempo_acumulado']:
                            start = start - file_start
            
                if re.match(self.database_variables['end_seizure'], line_str):
                    if (line_str.split()[3] == self.database_variables['end_seizure'].split()[-1]):
                        end = line_str.split()[4]
                        end = self.__process_time(end)
                        if self.database_variables['tiempo_acumulado']:
                            end = end - file_start
                    else: 
                        end = line_str.split()[3]
                        end = self.__process_time(end) 
                        if self.database_variables['tiempo_acumulado']:
                            if end > file_end: 
                                end = file_end
                            end = end - file_start
                    count_seizures += 1
                    register.addSeizure(start, end)

                    if count_seizures == nseizures:
                        self.registers[name] = register
                        count_seizures = 0
            
    def __process_channels(self):
        """
        Processes the channels to identify those common across all montages and creates a channel index.
        Obtains the common_channels and channel_index attributes.
        """
        self.common_channels = [ch for ch, count in self.channels_dict.items() if count == self.nmontages]              # busca los canales presentes en todos los montajes
        self.channel_index = dict(zip( list(np.arange(len(self.common_channels))), self.common_channels ))              # channel_index: asigna indices a los canales comunes
    
    def __process_time(self, time):

        if self.database_variables['time_format'] == 's': 
            try:
                time = int(time)
                return time
            except ValueError:
                print(f"Error: Formato incorrecto en el tiempo '{time}'. Se esperaba 's'.")
                return None
            
        if self.database_variables['time_format'] == 'hh.mm.ss': 
            try:
                hours, minutes, seconds = map(int, time.split('.'))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return total_seconds
            except ValueError:
                print(f"Error: Formato incorrecto en el tiempo '{time}'. Se esperaba 'hh.mm.ss'.")
                return None
        
        if self.database_variables['time_format'] == 'hh:mm:ss': 
            try:
                hours, minutes, seconds = map(int, time.split(':'))
                total_seconds = hours * 3600 + minutes * 60 + seconds
                return total_seconds
            except ValueError:
                print(f"Error: Formato incorrecto en el tiempo '{time}'. Se esperaba 'hh:mm:ss'.")
                return None
        
    def get_registers(self):
        """
        Returns the dictionary of registers containing the seizure data.
        """
        return self.registers
    
    def get_channel_index(self):
        """
        Returns the dictionary that maps channel indices to channel names.
        """
        return self.channel_index



class MATReader:
    """
    This class is responsible for reading and processing a MATLAB (.mat) annotation file to extract information.
    """
    def __init__(self, info_filename, database_variables, patient):
        self.info_filename = info_filename
        self.data_info = loadmat(info_filename)
        self.database_variables = database_variables

        self.fs = None

        self.registers = {}
        self.channel_index = {}
        
        self.__process_registers(patient)
        
    def __process_registers(self, patient):
        self.fs = self.data_info[self.database_variables['fs']][0][0]

        files_with_seizures = []
        start_list = []
        end_list = []

        for i in range(len(self.data_info[self.database_variables['start_seizure']])):
            start = int(self.data_info[self.database_variables['start_seizure']][i][0])
            end = int(self.data_info[self.database_variables['end_seizure']][i][0])
            start_hour = int(start / 3600)
            end_hour = int(end / 3600)
            
            if start_hour != end_hour: 
                name1 = patient + "_" + str(start_hour) + "h.mat"
                name2 = patient + "_" + str(start_hour+1) + "h.mat"

                files_with_seizures.append(name1)
                files_with_seizures.append(name2)

                end1 = int(((start_hour+1) * 3600) - 1)
                start2 = int(((start_hour+1) * 3600))

                start_list.append(start % 3600)
                start_list.append(start2 % 3600)
                end_list.append(end1 % 3600)
                end_list.append(end % 3600)

            else: 
                name = patient + "_" + str(start_hour) + "h.mat"
                files_with_seizures.append(name)
                start_list.append(start % 3600)
                end_list.append(end % 3600)

        # SE PUEDE PONER CONTROL DE ERRORES: SI EL TAMAÑO DE START_LIST, END Y FILES ES IGUAL: 
        for i in range(len(files_with_seizures)):
            
            nseizures = files_with_seizures.count(files_with_seizures[i])
            register = Register(files_with_seizures[i], self.fs, nseizures)
            register.addSeizure(start_list[i], end_list[i]) 

            self.registers[files_with_seizures[i]] = register
            
    def read_data(self, data_filename):
        data = loadmat(data_filename)

        if not self.database_variables['channels']:
            nchannels = data[self.database_variables['data']].shape[0]       #ESTO INVENTA LOS NOMBRES DE LOS CANALES
            channels = [f"channel{i}" for i in range(1, nchannels+1)]

        else: 
            channels = self.data_info[self.database_variables['channels']]
            channels = [item[0] for item in channels]
        
        self.channel_index = dict(zip( list(np.arange(len(channels))), channels))

        data = data[self.database_variables['data']]
        numMuestras = data.shape[1]
        time = numMuestras / self.fs
        
        return data, self.fs, time

    def get_registers(self):
        """
        Returns the dictionary of registers containing the seizure data.
        """
        return self.registers
    
    def get_channel_index(self):
        """
        Returns the dictionary that maps channel indices to channel names.
        """
        return self.channel_index

            
