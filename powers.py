from constants import Constants
from api import API
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten

# Debug mode
debug = Constants.debug

if debug:
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    from IPython.display import display, HTML
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
        self.api = API()
        pass
    
    ### Useful functions ###

    # Function to get activity informations
    def __get_activities(self):
        return self.api.get_activities()

    # Function to get activities labels
    def __get_activities_labels(self):

        # Create labels
        labels = []
        # Get labels
        for item in self.__get_activities():
            labels.append(item['activity'])

        return labels

    # Function to get activity name by id
    def __get_activity_by_id(self, aid):
        
        # get activities
        activities = self.__get_activities()
        # search by id
        activity = aid
        for item in activities:
            if item['id'] == aid:
                activity = item['activity']

        return activity
    
    # Function to get the activity num
    def __num_of_activities(self):
        
        total_activity_num = 0
        # get activities
        activities = self.__get_activities()
        # get num of all activities
        total_activity_num = len(activities)

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

    # Function to get activity ID by name
    def __get_activity_id_by_name(self, name):
        
        for item in self.api.get_activities():
            if item['activity'] == name:
                return item['id']

        return Constants.no_id_value

    # Function to get activity name by ID
    def __get_activity_name_by_id(self, aid):
        
        for item in self.api.get_activities():
            if item['id'] == aid:
                return item['activity']

        return aid
    
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
        features = ['x', 'y', 'z', 't']
        return len(features)

    # Function to create segments
    def __create_segments(self, df, time_steps, step):
        segments, labels = self.__create_segments_and_labels(df, time_steps, step, None)
        return segments

    # Function to create segments and labels
    def __create_segments_and_labels(self, df, time_steps, step, label_name):

        # features = x, y, z, t
        # x, y, z, timestamp as features
        N_FEATURES = self.__num_features()
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
            
            # Create segments
            segments.append([xs, ys, zs, ts])

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
        division_point = archives_num / 5 # 1/5 to test, 4/5 to train 

        # Differentiate between test set and training set
        df_train = df[df['archive-encoded'] > division_point]
        df_test = df[df['archive-encoded'] <= division_point]

        return df_train, df_test

    # Function to prepare the dataframe
    # Normalize and round the data 
    def __prepare_dataframe(self, df):

        # Normalize features for data set (values between 0 and 1)
        df['x-axis']    = df['x-axis']      / df['x-axis'].max()
        df['y-axis']    = df['y-axis']      / df['y-axis'].max()
        df['z-axis']    = df['z-axis']      / df['z-axis'].max()
        df['timestamp'] = df['timestamp']   / df['timestamp'].max()        

        # Round numbers
        df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4, 'timestamp': 4})

        return df

    def __plot_activity(self, activity, data):

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
        self.__plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')
        self.__plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')
        self.__plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')
        plt.subplots_adjust(hspace=0.2)

        activity = self.__get_activity_by_id(activity)

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

        LABELS = self.__get_activities_labels()

        matrix = metrics.confusion_matrix(validations, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    annot=True,
                    fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def __show_dataset_graphs(self, df):

        # get activities
        activities = self.__get_activities()

        # Dataframe to show
        df_plt = df.copy()
        
        # Replace activity id with name
        df_plt['activity'] = [self.__get_activity_by_id(aid) for aid in df_plt['activity']]

        # Show how many data for each activity
        df_plt['activity'].value_counts().plot(kind='bar', title='Data by Activity Type')
        plt.show()

        for activity in np.unique(df_plt['activity']):
            subset = df_plt[df_plt['activity'] == activity][:180]
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
            archive, df_group = group
            # Transform timestamp to distance
            df_group = self.__timestamp_to_distance_helper(df_group)
            # Set
            group = archive, df_group
            # Associate the edited groups
            frames.append(df_group)
        
        # Concat the associated groups
        df = pd.concat(frames, axis=0, ignore_index=False)

        return df

    ### Main method ###
    def __teach_using(self, df, model_file):

        # The number of steps within one time segment
        TIME_PERIODS = Constants.ml_time_periods
        # The steps to take from one segment to the next; if this value is equal to
        # TIME_PERIODS, then there is no overlap between the segments
        STEP_DISTANCE = Constants.ml_step_distance

        # Hyper-parameters
        BATCH_SIZE = Constants.ml_batch_size # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        EPOCHS = Constants.ml_epoch

        # show data
        if debug:
            self.__show_dataset_graphs(df)
            pass

        # Transform non numeric column in numeric
        df['archive-encoded'] = preprocessing.LabelEncoder().fit_transform(df['archive'].values.ravel())

        # Transform activity name in numerical (id)
        df['activity'] = df['activity'].apply(self.__get_activity_id_by_name)
        # and then convert to float
        df['activity'] = df['activity'].apply(self.__convert_to_float)

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
        num_classes = self.__num_of_activities()

        # convert data to float: keras want float data
        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
        y_train_hot = np_utils.to_categorical(y_train, num_classes)

        # machine learning
        model_m = Sequential()
        # Set layers
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
            keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
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
            num_classes = self.__num_of_activities()

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
            print(classification_report(max_y_test, max_y_pred_test, zero_division=0))

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

        sensors = [
            Constants.sensor_type_accelerometer,
            Constants.sensor_type_gyroscope
        ]
        
        for sensor in sensors:
            
            # Get dataset path
            dataset_file = Constants.datasets_path + sensor + '.csv'

            # Load data set containing all the data from csv
            df = self.__read_data(dataset_file)

            # For each phone-position, grouped by distinct phone-position
            for group in df.groupby(by=['phone-position']):
                
                # Get
                position, df_group = group
                # Transform timestamp to distance
                df_group = self.__timestamp_to_distance_helper(df_group)
                # Train
                self.__teach_using(df_group, Constants.models_path + sensor + '_' + position + '.h5')

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
        num_classes = self.__num_of_activities()

        # convert data to float: keras want float data
        x_pred = x_pred.astype('float32')

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

    def predict(self, features, position):

        # Uncompress data
        features_acc    = features[Constants.sensor_type_accelerometer]
        features_gyro   = features[Constants.sensor_type_gyroscope]

        # Create dataframes from lists
        df_acc  = pd.DataFrame(data=features_acc)
        df_gyro = pd.DataFrame(data=features_gyro)

        # Convert timestamp to distance
        df_acc  = self.__timestamp_to_distance_helper(df_acc)
        df_gyro = self.__timestamp_to_distance_helper(df_gyro)

        # Init return values
        prediction, accuracy = None, 0
        # Predictions
        # Secondary prediction
        if len(df_gyro) >= Constants.ml_time_periods:
            prediction, accuracy = self.__predict_using(df_gyro, Constants.models_path + Constants.sensor_type_gyroscope + '_' + position + '.h5')
            # Results
            print('[Result][Gyro] Prediction', prediction, ' ', accuracy, '%')
        # Main prediction
        if len(df_acc) >= Constants.ml_time_periods:
            prediction, accuracy = self.__predict_using(df_acc, Constants.models_path + Constants.sensor_type_accelerometer + '_' + position + '.h5')
            # Results
            print('[Result][Acc] Prediction', prediction, ' ', accuracy, '%')

        # and then return
        return prediction, accuracy
