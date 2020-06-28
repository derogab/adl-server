from constants import Constants
import h5py
import urllib.request, json 
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report
import keras
from keras.utils import np_utils
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Reshape

# Debug mode
debug = Constants.debug

if debug:
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    from IPython.display import display, HTML
    from keras.layers import Conv2D, MaxPooling2D
    from scipy import stats

    sns.set() # Default seaborn look and feel
    plt.style.use('ggplot')

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
# Surpress warning for some operation
pd.options.mode.chained_assignment = None  # default='warn'

class Power:

    ### Init ###
    def __init__(self):
        pass
    
    ### Useful functions ###
    
    # Function to get the activity num
    def __num_activity(self):
        
        total_activity_num = 0
        with urllib.request.urlopen("https://api.adl.derogab.com/activities") as url:
            data = json.loads(url.read().decode())
            # get activities
            activities = data['activities']
            # get num of all activities
            total_activity_num = len(data['activities'])

        return total_activity_num

    # Function to get the phone positions num
    def __num_phone_position(self):

        positions = [
            'left_hand', 
            'right_hand', 
            'front_left_pocket', 
            'back_left_pocket', 
            'front_right_pocket', 
            'back_right_pocket'
        ]

        return len(positions)
    
    # Function to convert a value to float
    def __convert_to_float(self, x):
        try:
            return np.float(x)
        except:
            return np.nan

    # Function to read a csv file
    def __read_data(self, file_path):

        column_names = ['archive',
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

    def __split_dataframe(self, df):
        
        # Get archives num
        archives_num = df['archive-encoded'].nunique()
        
        # Calculate the point where to divide
        division_point = archives_num / 3 # 1/3 to test, 2/3 to train 

        # Differentiate between test set and training set
        df_train = df[df['archive-encoded'] > division_point]
        df_test = df[df['archive-encoded'] <= division_point]

        return df_train, df_test

    # Function to prepare the dataframe
    # Normalize and round the data 
    def __prepare_dataframe(self, df):

        # Normalize features for data set (values between 0 and 1)
        df['x-axis'] = df['x-axis'] / df['x-axis'].max()
        df['y-axis'] = df['y-axis'] / df['y-axis'].max()
        df['z-axis'] = df['z-axis'] / df['z-axis'].max()
        df['timestamp'] = df['timestamp'] / df['timestamp'].max()
        df['phone-position'] = df['phone-position'] / self.__num_phone_position()

        # Round numbers
        df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4, 'timestamp': 4, 'phone-position': 4})

        return df

    def __plot_activity(self, activity, data):

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
        self.__plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')
        self.__plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')
        self.__plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')
        plt.subplots_adjust(hspace=0.2)
        fig.suptitle(activity)
        plt.subplots_adjust(top=0.90)
        plt.show()

    def __plot_axis(self, ax, x, y, title):

        ax.plot(x, y, 'r')
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
        ax.set_xlim([min(x), max(x)])
        ax.grid(True)

    def __show_confusion_matrix(self, validations, predictions):

        matrix = metrics.confusion_matrix(validations, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    annot=True,
                    fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def __show_dataset_graphs(self, df):

        # Show how many data for each activity
        df['activity'].value_counts().plot(kind='bar', title='Data by Activity Type')
        plt.show()

        for activity in np.unique(df['activity']):
            subset = df[df['activity'] == activity][:180]
            self.__plot_activity(activity, subset)

    def __timestamp_to_distance_helper(self, df):

        relatives = []
        min_timestamp = df['timestamp'].min()
        for time in df['timestamp']:
            relatives.append(np.subtract(time, min_timestamp))

        distances = []
        previous_time = 0
        for time in relatives:
            distances.append(np.subtract(time, previous_time))
            previous_time = time

        df['timestamp'] = distances

        return df


    def __timestamp_to_distance(self, df):

        frames = []
        # For each group, grouped by distinct archive-encoded
        for group in df.groupby(by=['archive-encoded']):
            
            # Get
            index, df_group = group
            # Transform timestamp to distance
            df_group = self.__timestamp_to_distance_helper(df_group)
            # Set
            group = index, df_group
            # Associate the edited groups
            frames.append(df_group)
        
        # Concat the associated groups
        df = pd.concat(frames, axis=0, ignore_index=False)

        return df

    ### Main method ###
    def __teach_using(self, dataset_file, model_file):

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

        # show data
        if debug:
            self.__show_dataset_graphs(df)
            pass

        # Transform non numeric column in numeric
        df['archive-encoded'] = preprocessing.LabelEncoder().fit_transform(df['archive'].values.ravel())

        # Convert in float
        df['phone-position'] = [self.__convert_to_float(x) for x in df['phone-position'].to_numpy()]
        df['activity'] = [self.__convert_to_float(x) for x in df['activity'].to_numpy()]

        # Split the dataframe
        df_train, df_test = self.__split_dataframe(df)

        # Convert timestamp to distance
        df_train = self.__timestamp_to_distance(df_train)        

        # Prepare dataframe for training data
        df_train = self.__prepare_dataframe(df_train)

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
        model_m.add(Dense(100, activation='relu'))
        model_m.add(Dense(100, activation='relu'))
        model_m.add(Flatten())
        model_m.add(Dense(num_classes, activation='softmax'))

        # Callback list
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath=Constants.tmp_path+'best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
        ]

        # Model compile
        model_m.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

        # Train
        trained = False
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

            # check successfull train
            trained = True                    

        except:
            print('[Warning] Still little data to learn...')


        if trained:
            
            # Convert timestamp to distance
            df_test = self.__timestamp_to_distance(df_test)

            # Prepare dataframe for testing data
            df_test = self.__prepare_dataframe(df_test)

            # Create segments and labels
            x_test, y_test = self.__create_segments_and_labels(df_test, TIME_PERIODS, STEP_DISTANCE, 'activity')

            # Set input & output dimensions
            num_time_periods, num_sensors = x_test.shape[1], x_test.shape[2]
            num_classes = self.__num_activity()

            # compress two-dimensional data in one-dimensional data
            input_shape = (num_time_periods*num_sensors)
            x_test = x_test.reshape(x_test.shape[0], input_shape)

            # convert data to float: keras want float data
            x_test = x_test.astype('float32')
            y_test = y_test.astype('float32')

            # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
            y_test_hot = np_utils.to_categorical(y_test, num_classes)

            # Predict using test data
            y_pred_test = model_m.predict(x_test)

            # Take the class with the highest probability from the test predictions
            max_y_pred_test = np.argmax(y_pred_test, axis=1)
            max_y_test = np.argmax(y_test_hot, axis=1)

            # Show classification report
            print(classification_report(max_y_test, max_y_pred_test))

            # Show after-train graphs in debug mode
            if debug:

                # show accuracy and loss graph
                plt.figure(figsize=(6, 4))
                plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
                plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
                plt.plot(history.history['loss'], 'r--', label='Loss of training data')
                plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
                plt.title('Model Accuracy and Loss')
                plt.ylabel('Accuracy and Loss')
                plt.xlabel('Training Epoch')
                plt.ylim(0)
                plt.legend()
                plt.show()

                # show confusion matrix
                self.__show_confusion_matrix(max_y_test, max_y_pred_test)


    def teach(self):
        self.__teach_using(Constants.datasets_path + Constants.sensor_type_accelerometer + '.csv', Constants.models_path + Constants.sensor_type_accelerometer + '.h5')
        self.__teach_using(Constants.datasets_path + Constants.sensor_type_gyroscope + '.csv', Constants.models_path + Constants.sensor_type_gyroscope + '.h5')

    # Function to find most frequent  
    # element in a list 
    def __most_frequent(self, my_list):
        
        response = np.bincount(my_list).argmax()
        counter = 0

        # count element
        for element in my_list:
            if element == response:
                counter = counter + 1
        # accuracy % 
        # conter : accuracy = len(my_list) : 100
        accuracy = (counter * 100) / len(my_list)
        
        return response, accuracy

    # Function to predict 
    # using a pretrained model and a set of live data
    def __predict_using(self, df, model_file):

        # The number of steps within one time segment
        TIME_PERIODS = Constants.ml_time_periods
        # The steps to take from one segment to the next; if this value is equal to
        # TIME_PERIODS, then there is no overlap between the segments
        STEP_DISTANCE = Constants.ml_step_distance

        # Prepare dataframe for training data
        df = self.__prepare_dataframe(df)

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
        loaded = False
        try:
            loaded_model = load_model(model_file)
            loaded = True
        except:
            print('[Error] Model load failed.')

        # Init return value
        prediction = None
        accuracy = None

        if loaded:
        
            # evaluate loaded model on test data
            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Check Against Test Data
            predicted = False
            try:
                y_pred_test = loaded_model.predict(x_pred)
                predicted = True
            except:
                print('[Warning] Still little data to predict...')

            # Check results
            if predicted:

                # Take the class with the highest probability from the test predictions
                prediction_results = False
                try:
                    max_y_pred_test = np.argmax(y_pred_test, axis=1)
                    prediction_results = True
                except:
                    print('[Warning] Still little data to get results...')
                
                if prediction_results:
                    # Get prediction result and accuracy
                    prediction, accuracy = self.__most_frequent(max_y_pred_test)

        # Return results
        return prediction, accuracy

    def predict(self, data):

        # Uncompress data
        data_acc    = data[Constants.sensor_type_accelerometer]
        data_gyro   = data[Constants.sensor_type_gyroscope]

        # Create dataframes from lists
        df_acc  = pd.DataFrame(data=data_acc)
        df_gyro = pd.DataFrame(data=data_gyro)

        # Convert timestamp to distance
        df_acc  = self.__timestamp_to_distance_helper(df_acc)
        df_gyro = self.__timestamp_to_distance_helper(df_gyro)

        # Predictions
        prediction_acc, accuracy_acc    = self.__predict_using(df_acc, Constants.models_path + Constants.sensor_type_accelerometer + '.h5')
        prediction_gyro, accuracy_gyro  = self.__predict_using(df_gyro, Constants.models_path + Constants.sensor_type_gyroscope + '.h5')

        # Results
        print('[RESULT] Prediction ACC', prediction_acc, ' ', accuracy_acc, '%')
        print('[RESULT] Prediction GYRO', prediction_gyro, ' ', accuracy_gyro, '%')

        # Select result
        if accuracy_gyro and not accuracy_acc:
            prediction = prediction_gyro
            accuracy = accuracy_gyro
        elif accuracy_acc:
            # main prediction input
            prediction = prediction_acc
            accuracy = accuracy_acc
        else:
            # no prediction
            prediction = None
            accuracy = 0

        # and then return
        return prediction, accuracy
