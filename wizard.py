from constants import Constants
from powers import Power
import math

class Wizard():

    def __init__(self):
        # Init data
        self.archive = None
        self.position = None
        self.collection = []
        self.power = Power()

    def collect(self, archive, index, sensor, position, values):

        self.archive = archive
        self.position = position

        x = values['x']
        y = values['y']
        z = values['z']
        t = values['t']

        self.collection.append([sensor, index, x, y, z, t])

    def __order_by_index(self, row):
        return row[1] # index

    def predict(self):
        print('[Info] Do a magic...')

        # snapshot collection 
        collection = self.collection
        # snapshot position
        position = self.position

        # Sort the data
        collection.sort(key=self.__order_by_index)

        # Print number of data used in prediction
        print('[Info] Magic w/ ', len([row for row in collection if row[0] == Constants.sensor_type_accelerometer]), 'accelerometer data')
        print('[Info] Magic w/ ', len([row for row in collection if row[0] == Constants.sensor_type_gyroscope]), 'gyroscope data')

        # Get the accelerometer data
        acc = {
            'x-axis':       [row[2] for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'y-axis':       [row[3] for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'z-axis':       [row[4] for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'timestamp':    [row[5] for row in collection if row[0] == Constants.sensor_type_accelerometer]
        }

        # Get the gyroscope data
        gyro = {
            'x-axis':       [row[2] for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'y-axis':       [row[3] for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'z-axis':       [row[4] for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'timestamp':    [row[5] for row in collection if row[0] == Constants.sensor_type_gyroscope]
        }

        # Compress data
        features = {
            Constants.sensor_type_accelerometer: acc,
            Constants.sensor_type_gyroscope: gyro
        }

        # Predict
        return self.power.predict(features, position)
