import json
import threading
import time
from replies import Reply
from constants import Constants

class Worker(threading.Thread):

    user = 0

    def __init__(self, socket, data_received, wizard, teacher):
        threading.Thread.__init__(self)
        self.socket = socket
        self.data_received = data_received
        self.wizard = wizard
        self.teacher = teacher
        
    def run(self):
        msg = self.data_received.strip().decode()

        if msg:

            array = msg.split('\n')

            for x in array:

                request = False
                try:
                    request = json.loads(x)
                except:
                    print('[Error] JSON decode failed.')
                
                if request and request['status'] == Constants.request_status_success:

                    data = request['data']

                    if data['type'] == Constants.request_type_data:
                        
                        self.socket.send(Reply.ack())

                        if request['mode'] == Constants.request_mode_analyzer:
                            
                            # get data
                            archive = data['archive']
                            index = data['info']['index']
                            sensor = data['info']['sensor']
                            position = data['info']['position']
                            values = data['values']

                            # insert data in my wizard
                            self.wizard.collect(archive, index, sensor, position, values)
                            
                            

                        if request['mode'] == Constants.request_mode_learning:

                            # get data
                            archive = data['archive']
                            index = data['info']['index']
                            activity = data['info']['activity']
                            sensor = data['info']['sensor']
                            position = data['info']['position']
                            values = data['values']

                            # do something with learning data
                        

                    if data['type'] == Constants.request_type_close:

                        self.socket.send(Reply.close())

                
                else:
                    print('[Error] Data received error.')
