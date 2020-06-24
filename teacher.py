from constants import Constants
from powers import Power
import time
import math

class Teacher:

    def __init__(self):
        # Init data
        self.archive = None
        self.position = None
        self.activity = None    
        self.collection = []
        self.power = Power()
        # Memo data
        self.hasBeenSaved = False
        self.hasBeenTrained = False

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

        print('[Info] Saving collected data...')
        
        if not self.hasBeenSaved:
            
            # Set save status
            self.hasBeenSaved = True

            # select the dataset
            file_acc = Constants.datasets_path + Constants.sensor_type_accelerometer + ".csv"
            file_gyro = Constants.datasets_path + Constants.sensor_type_gyroscope + ".csv"

            # snapshots
            snapshot_archive    = self.archive
            snapshot_position   = self.position
            snapshot_activity   = self.activity
            snapshot_collection = self.collection

            # timestamp to relative & distance
            timestamp_min_acc = math.inf
            timestamp_min_gyro = math.inf
            previous_row_acc = None
            previous_row_gyro = None
            for row in snapshot_collection:
                if row[5] < timestamp_min_acc and row[0] == Constants.sensor_type_accelerometer:
                    timestamp_min_acc = row[5]
                if row[5] < timestamp_min_gyro and row[0] == Constants.sensor_type_gyroscope:
                    timestamp_min_gyro = row[5]
            for row in snapshot_collection:
                if row[0] == Constants.sensor_type_accelerometer:
                    row[5] = row[5] - timestamp_min_acc # absolute to relative
                    if previous_row_acc:
                        row[5] = row[5] - previous_row_acc[5] # distance
                    previous_row_acc = row
                if row[0] == Constants.sensor_type_gyroscope:
                    row[5] = row[5] - timestamp_min_gyro # absolute to relative
                    if previous_row_gyro:
                        row[5] = row[5] - previous_row_gyro[5] # distance
                    previous_row_gyro = row

            # save the collection to storage
            with open(file_acc, "a+") as f:
                for row in snapshot_collection:
                    if row[0] == Constants.sensor_type_accelerometer: #check sensor
                        # archive, index, x, y, z, t, position, activity
                        f.write(str(snapshot_archive)+ "," \
                            + str(row[1]) + "," \
                            + str(row[2]) + "," \
                            + str(row[3]) + "," \
                            + str(row[4]) + "," \
                            + str(row[5]) + "," \
                            + str(snapshot_position) + "," \
                            + str(snapshot_activity) + ';\n')

            with open(file_gyro, "a+") as f:
                for row in snapshot_collection:
                    if row[0] == Constants.sensor_type_gyroscope: #check sensor
                        # archive, index, x, y, z, t, position, activity
                        f.write(str(snapshot_archive)+ "," \
                            + str(row[1]) + "," \
                            + str(row[2]) + "," \
                            + str(row[3]) + "," \
                            + str(row[4]) + "," \
                            + str(row[5]) + "," \
                            + str(snapshot_position) + "," \
                            + str(snapshot_activity) + ';\n')


    def teach(self):

        print('[Info] Train all the dataset...')

        # wait save
        i = 1
        while (not self.hasBeenSaved) and i < 10:
            # back-off waiting
            time.sleep(i)
            # and increment next time wait
            i += 1
        
        # and then train
        if not self.hasBeenTrained:

            # Set save status
            self.hasBeenTrained = True

            # Launch train
            self.power.teach()
