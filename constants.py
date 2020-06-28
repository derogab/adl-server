# Constants
# Class w/ useful constants
class Constants:

    def __init__(self):
        raise TypeError("cannot create 'Constants' instances")

    # Server
    server_host = ""
    server_port = 3000
    # Connection
    connection_max_timeout = 9999999
    something_value = 200
    # ML params
    ml_time_periods = 200
    ml_step_distance = 20
    ml_batch_size = 100
    ml_epoch = 50
    # Sensor type
    sensor_type_accelerometer = "accelerometer"
    sensor_type_gyroscope = "gyroscope"
    # Request
    request_status_success = "OK"
    request_mode_analyzer = "analyzer"
    request_mode_learning = "learning"
    request_type_data = "data"
    request_type_close = "close"
    request_type_destroy = "destroy"
    # Paths
    datasets_path = "/datasets/"
    models_path = "/models/"
    tmp_path = "/tmp/"
    # Debug mode
    debug = False