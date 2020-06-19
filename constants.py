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
    something_value = 500
    # Sensor type
    sensor_type_accelerometer = "accelerometer"
    sensor_type_gyroscope = "gyroscope"
    # Request
    request_status_success = "OK"
    request_mode_analyzer = "analyzer"
    request_mode_learning = "learning"
    request_type_data = "data"
    request_type_close = "close"
    # Paths
    datasets_path = "/datasets/"
    models_path = "/models/"
    tmp_path = "/tmp/"
