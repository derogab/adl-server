import threading
from constants import Constants
from powers import Power

class Teacher(threading.Thread):

    def __init__(self):
        # Init thread
        threading.Thread.__init__(self)
        # Init data
        self.archive = None
        self.position = None
        self.activity = None    
        self.collection = []
        self.power = Power()

    def collect(self, archive, index, activity, sensor, position, values):

        self.archive = archive
        self.position = position
        self.activity = activity

        x = values['x']
        y = values['y']
        z = values['z']
        t = values['t']

        self.collection.append([sensor, index, x, y, z, t])

    def save(self):
        print('[Info] Save...')

        # select the dataset
        file_acc = Constants.datasets_path + Constants.sensor_type_accelerometer + ".csv"
        file_gyro = Constants.datasets_path + Constants.sensor_type_gyroscope + ".csv"

        # save the collection to storage
        with open(file_acc, "a+") as f:
            for row in self.collection:
                if row[0] == Constants.sensor_type_accelerometer: #check sensor
                    # archive, index, x, y, z, t, position, activity
                    f.write(str(self.archive)+ "," \
                        + str(row[1]) + "," \
                        + str(row[2]) + "," \
                        + str(row[3]) + "," \
                        + str(row[4]) + "," \
                        + str(row[5]) + "," \
                        + str(self.position) + "," \
                        + str(self.activity) + ';\n')

        with open(file_gyro, "a+") as f:
            for row in self.collection:
                if row[0] == Constants.sensor_type_gyroscope: #check sensor
                    # archive, index, x, y, z, t, position, activity
                    f.write(str(self.archive)+ "," \
                        + str(row[1]) + "," \
                        + str(row[2]) + "," \
                        + str(row[3]) + "," \
                        + str(row[4]) + "," \
                        + str(row[5]) + "," \
                        + str(self.position) + "," \
                        + str(self.activity) + ';\n')

    def run(self):
        print('[Info] Teaching...')
        self.power.teach()