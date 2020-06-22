from constants import Constants
import h5py
import urllib.request, json 
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics, preprocessing
import keras
from keras.utils import np_utils
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Reshape

class Power:

    ### Init ###
    def __init__(self):
        pass
    
    ### Useful functions ###
    
    # activity num
    def __num_activity(self):
        
        total_activity_num = 0
        with urllib.request.urlopen("https://api.adl.derogab.com/activities") as url:
            data = json.loads(url.read().decode())
            # get activities
            activities = data['activities']
            # get num of all activities
            total_activity_num = len(data['activities'])

        return total_activity_num
    
    # Function to convert a value to float
    def __convert_to_float(self, x):
        try:
            return np.float(x)
        except:
            return np.nan

    # Function to read a csv file
    def __read_data(self, file_path):

        column_names = ['user-id',
                        'index',
                        'x-axis',
                        'y-axis',
                        'z-axis',
                        'timestamp',
                        'phone-position',
                        'activity']
        df = pd.read_csv(file_path,
                        header=None,
                        names=column_names)
        # Last column has a ";" character which must be removed ...
        df['activity'].replace(regex=True,
        inplace=True,
        to_replace=r';',
        value=r'')
        # ... and then (if last data is a float) this column must be transformed to float explicitly
        # if it must be a float decomment following row
        df['activity'] = df['activity'].apply(self.__convert_to_float)
        # This is very important otherwise the model will not fit and loss
        # will show up as NAN
        df.dropna(axis=0, how='any', inplace=True)

        return df

    # Function to get some informations about file
    def __get_basic_dataframe_info(self, dataframe):

        columns = dataframe.shape[1]
        rows = dataframe.shape[0]

        return rows, columns

    # Function to print some informations about file
    def __show_basic_dataframe_info(self, dataframe):

        # Get rows and columns
        rows, columns = self.__get_basic_dataframe_info(dataframe)

        # Shape and how many rows and columns
        print('Number of columns in the dataframe: %i' % columns)
        print('Number of rows in the dataframe: %i\n' % rows)

    # num features
    def __num_features(self):
        features = ['x', 'y', 'z', 't', 'p']
        return len(features)

    # Function to create segments
    def __create_segments(self, df, time_steps, step):
        segments, labels = self.__create_segments_and_labels(df, time_steps, step, None)
        return segments

    # Function to create segments and labels
    def __create_segments_and_labels(self, df, time_steps, step, label_name):

        # features = x, y, z, t, p
        # x, y, z acceleration, timestamp and phone position as features
        N_FEATURES = 5
        # Number of steps to advance in each iteration (for me, it should always
        # be equal to the time_steps in order to have no overlap between segments)
        # step = time_steps
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):

            xs = df['x-axis'].values[i: i + time_steps]
            ys = df['y-axis'].values[i: i + time_steps]
            zs = df['z-axis'].values[i: i + time_steps]
            ts = df['timestamp'].values[i: i + time_steps]
            ps = df['phone-position'].values[i: i + time_steps]
            
            # Create segments
            segments.append([xs, ys, zs, ts, ps])

            # Create labels
            if label_name:
                # Retrieve the most often used label in this segment
                label = stats.mode(df[label_name][i: i + time_steps])[0][0]
                # and then
                labels.append(label)   

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)

        return reshaped_segments, labels


    ### Main method ###
    def __teach_using(self, dataset_file, model_file):

        # Set some standard parameters upfront
        pd.options.display.float_format = '{:.1f}'.format
        # The number of steps within one time segment
        TIME_PERIODS = Constants.ml_time_periods
        # The steps to take from one segment to the next; if this value is equal to
        # TIME_PERIODS, then there is no overlap between the segments
        STEP_DISTANCE = Constants.ml_step_distance

        # Hyper-parameters
        BATCH_SIZE = Constants.ml_batch_size # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        EPOCHS = Constants.ml_epoch

        # Load data set containing all the data from csv
        df = self.__read_data(dataset_file)

        # Transform non numeric column in numeric
        df['user-id-encoded'] = preprocessing.LabelEncoder().fit_transform(df['user-id'].values.ravel())

        # Convert in float
        df['phone-position'] = [self.__convert_to_float(x) for x in df['phone-position'].to_numpy()]
        df['activity'] = [self.__convert_to_float(x) for x in df['activity'].to_numpy()]

        # Get archives num
        archives_num = df['user-id-encoded'].nunique()
        
        # Calculate the point where to divide
        division_point = archives_num / 3 # 1/3 to test, 2/3 to train 

        # Differentiate between test set and training set
        df_train = df[df['user-id-encoded'] > division_point]
        df_test = df[df['user-id-encoded'] <= division_point]

        # Normalize features for training data set (values between 0 and 1)
        # Surpress warning for next 3 operation
        pd.options.mode.chained_assignment = None  # default='warn'
        df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
        df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
        df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()
        df_train['timestamp'] = df_train['timestamp'] / df_train['timestamp'].max()
        df_train['phone-position'] = df_train['phone-position'] / 6

        # Round numbers
        df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4, 'timestamp': 4, 'phone-position': 4})

        # Create segments and labels
        x_train, y_train = self.__create_segments_and_labels(df_train, TIME_PERIODS, STEP_DISTANCE, 'activity')

        # Set input & output dimensions
        num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
        num_classes = self.__num_activity()

        # compress two-dimensional data in one-dimensional data
        input_shape = (num_time_periods*num_sensors)
        x_train = x_train.reshape(x_train.shape[0], input_shape)

        # convert data to float: keras want float data
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
        y_train_hot = np_utils.to_categorical(y_train, num_classes)

        # machine learning
        model_m = Sequential()
        # Ugly but functional workaround
        # To fix a Keras bug with last version of tensorflow
        # https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
        keras.backend.tensorflow_backend._SYMBOLIC_SCOPE.value = True
        # Remark: since coreml cannot accept vector shapes of complex shape like
        # [TIME_PERIODS, __num_features()] this workaround is used in order to reshape the vector internally
        # prior feeding it into the network
        model_m.add(Reshape((TIME_PERIODS, self.__num_features()), input_shape=(input_shape,)))
        model_m.add(Dense(100, activation='relu'))
        model_m.add(Dense(100, activation='relu'))
        model_m.add(Dense(100, activation='relu'))
        model_m.add(Flatten())
        model_m.add(Dense(num_classes, activation='softmax'))

        # Callback list
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath=Constants.tmp_path+'best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
        ]

        # Model compile
        model_m.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

        # Train
        try:
            
            # Fit the model
            # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
            history = model_m.fit(x_train,
                                y_train_hot,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                callbacks=callbacks_list,
                                validation_split=0.2,
                                verbose=1)

            # evaluate the model
            scores = model_m.evaluate(x_train, y_train_hot, verbose=0)
            print("[METRICS] %s: %.2f%%" % (model_m.metrics_names[1], scores[1]*100))

            # save the network to disk
            print("[INFO] serializing network to '{}'...".format(model_file))
            model_m.save(model_file, overwrite=True)

        except:
            print('[Warning] Still little data to learn...')


    def teach(self):
        self.__teach_using(Constants.datasets_path + Constants.sensor_type_accelerometer + '.csv', Constants.models_path + Constants.sensor_type_accelerometer + '.h5')
        #self.__teach_using(Constants.datasets_path + Constants.sensor_type_gyroscope + '.csv', Constants.models_path + Constants.sensor_type_gyroscope + '.h5')

    # Program to find most frequent  
    # element in a list 
    def __most_frequent(self, my_list):
        return np.bincount(my_list).argmax()

    def __predict_using(self, df, model_file):

        # Set some standard parameters upfront
        pd.options.display.float_format = '{:.1f}'.format
        # The number of steps within one time segment
        TIME_PERIODS = Constants.ml_time_periods
        # The steps to take from one segment to the next; if this value is equal to
        # TIME_PERIODS, then there is no overlap between the segments
        STEP_DISTANCE = Constants.ml_step_distance

        # Normalize features for training data set (values between 0 and 1)
        # Surpress warning for next 3 operation
        pd.options.mode.chained_assignment = None  # default='warn'
        df['x-axis']    = df['x-axis'] / df['x-axis'].max()
        df['y-axis']    = df['y-axis'] / df['y-axis'].max()
        df['z-axis']    = df['z-axis'] / df['z-axis'].max()
        df['timestamp'] = df['timestamp'] / df['timestamp'].max()
        df['phone-position'] = df['phone-position'] / 6

        # Round numbers
        df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4, 'timestamp': 4, 'phone-position': 4})

        # Create segments
        x_pred = self.__create_segments(df, TIME_PERIODS, STEP_DISTANCE)

        # Set input & output dimensions
        num_time_periods, num_sensors = x_pred.shape[1], x_pred.shape[2]
        num_classes = self.__num_activity()

        # compress two-dimensional data in one-dimensional data
        input_shape = (num_time_periods*num_sensors)
        x_pred = x_pred.reshape(x_pred.shape[0], input_shape)

        # convert data to float: keras want float data
        x_pred = x_pred.astype('float32')

        # Ugly but functional workaround
        # To fix a Keras bug with last version of tensorflow
        # https://github.com/keras-team/keras/issues/13353#issuecomment-545459472
        keras.backend.tensorflow_backend._SYMBOLIC_SCOPE.value = True

        # load json and create model
        loaded_model = load_model(model_file)

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Check Against Test Data
        y_pred_test = loaded_model.predict(x_pred)

        # Take the class with the highest probability from the test predictions
        max_y_pred_test = np.argmax(y_pred_test, axis=1)

        # Take more frequent result
        prediction = self.__most_frequent(max_y_pred_test)

        # Calculate accuracy
        accuracy = None

        # Result
        print('[RESULT] Prediction', prediction)

        return prediction, accuracy

    def predict(self, data):

        # Uncompress data
        data_acc    = data[Constants.sensor_type_accelerometer]
        #data_gyro   = data[Constants.sensor_type_gyroscope]

        # Create dataframes from lists
        df_acc  = pd.DataFrame(data=data_acc)
        #df_gyro = pd.DataFrame(data=data_gyro)

        # Predictions
        prediction_acc, accuracy_acc = self.__predict_using(df_acc, Constants.models_path + Constants.sensor_type_accelerometer + '.h5')
        #prediction_gyro, accuracy_gyro = self.__predict_using(df_gyro, Constants.models_path + Constants.sensor_type_gyroscope + '.h5')

        # Select best prediction
        

        prediction = prediction_acc
        accuracy = accuracy_acc


        return prediction, accuracy
