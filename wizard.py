from constants import Constants
from powers import Power

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

    def predict(self):
        print('[Info] Do a magic...')

        # snapshot collection 
        collection = self.collection
        # snapshot position
        position = self.position

        # Get the accelerometer data
        acc = {
            'x-axis':           [row[2]     for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'y-axis':           [row[3]     for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'z-axis':           [row[4]     for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'timestamp':        [row[5]     for row in collection if row[0] == Constants.sensor_type_accelerometer],
            'phone-position':   [position   for row in collection if row[0] == Constants.sensor_type_accelerometer],
        }

        # Get the gyroscope data
        gyro = {
            'x-axis':           [row[2]     for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'y-axis':           [row[3]     for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'z-axis':           [row[4]     for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'timestamp':        [row[5]     for row in collection if row[0] == Constants.sensor_type_gyroscope],
            'phone-position':   [position   for row in collection if row[0] == Constants.sensor_type_gyroscope],
        }

        # Compress data
        data = {
            Constants.sensor_type_accelerometer: acc,
            Constants.sensor_type_gyroscope: gyro
        }

        # Predict
        return self.power.predict(data)
