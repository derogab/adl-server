# Constants
# Class w/ useful constants
class Constants:

    def __init__(self):
        raise TypeError("cannot create 'Constants' instances")

    # Server
    server_host = ""
    server_port = 3000
    # API
    api_path = "https://api.adl.derogab.com"
    # Connection
    connection_max_timeout = 9999999
    something_value = 200
    # ML params
    ml_time_periods = 80
    ml_step_distance = 40
    ml_batch_size = 400
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
    # No ID value
    no_id_value = -1
    # Debug mode
    debug = False
    # Debug graph axis num value
    debug_graph_quantity_values = 300