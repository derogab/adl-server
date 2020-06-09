from constants import Constants

class Teacher():

    archive = None
    position = None
    activity = None    
    collection = []

    def __init__(self):
        print("Teacher created.")
        # Init data
        self.archive = None
        self.position = None
        self.activity = None    
        self.collection = []
        pass

    def collect(self, archive, index, activity, sensor, position, values):

        self.archive = archive
        self.position = position
        self.activity = activity

        x = values['x']
        y = values['y']
        z = values['z']

        self.collection.append([sensor, index, x, y, z])

        pass

    def save(self):

        # select the dataset
        file_acc = "./dataset/" + Constants.sensor_type_accelerometer + ".csv"
        file_gyro = "./dataset/" + Constants.sensor_type_accelerometer + ".csv"

        # save the collection to storage
        with open(file_acc, "a+") as f:
            for row in self.collection:
                if row[0] == Constants.sensor_type_accelerometer: #check sensor
                    # archive, position, index, x, y, z, activity
                    f.write(str(self.archive)+ "," \
                        + str(self.position) + "," \
                        + str(row[1]) + "," \
                        + str(row[2]) + "," \
                        + str(row[3]) + "," \
                        + str(row[4]) + "," \
                        + str(self.activity) + ';\n')

        with open(file_gyro, "a+") as f:
            for row in self.collection:
                if row[0] == Constants.sensor_type_gyroscope: #check sensor
                    # archive, position, index, x, y, z, activity
                    f.write(str(self.archive)+ "," \
                        + str(self.position) + "," \
                        + str(row[1]) + "," \
                        + str(row[2]) + "," \
                        + str(row[3]) + "," \
                        + str(row[4]) + "," \
                        + str(self.activity) + ';\n')

        pass

    def teach(self):
        
        # fit here

        pass
    
    pass