import json
import threading
import time
from replies import Reply
from constants import Constants

class Worker(threading.Thread):

    def __init__(self, socket, data_received, wizard, teacher):
        # Init thread
        threading.Thread.__init__(self)
        # Init data
        self.socket = socket
        self.data_received = data_received
        self.wizard = wizard
        self.teacher = teacher

    def __send_message(self, msg):
        try:
            self.socket.send(msg)
        except:
            print('[Warning] Connection already closed by client.')

    def __at_the_end(self):
        print('[Info] All data received. Starting TODO work list...')
        
        # send close message
        try:
            self.__send_message(Reply.close())
        except:
            print('[Warning] Connection already closed by client.')

        # save collected data to datasets
        if self.teacher:
            self.teacher.save()
        # train all the data
        if self.teacher:
            self.teacher.start()

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
                        
                        # send ack
                        self.__send_message(Reply.ack())

                        # mode
                        if request['mode'] == Constants.request_mode_analyzer:

                            # kill teacher
                            self.teacher = None
                            
                            # get data
                            archive = data['archive']
                            index = data['info']['index']
                            sensor = data['info']['sensor']
                            position = data['info']['position']
                            values = data['values']

                            # collect data in my wizard
                            self.wizard.collect(archive, index, sensor, position, values)

                            # predict sometimes
                            if index % Constants.something_value == 0:
                                prediction, accuracy = self.wizard.predict()
                                # send prediction
                                self.__send_message(Reply.prediction(prediction))
                           

                        if request['mode'] == Constants.request_mode_learning:

                            # kill wizard
                            self.wizard = None

                            # get data
                            archive = data['archive']
                            index = data['info']['index']
                            activity = data['info']['activity']
                            sensor = data['info']['sensor']
                            position = data['info']['position']
                            values = data['values']

                            # collect data in my teacher
                            self.teacher.collect(archive, index, activity, sensor, position, values)
                        

                    if data['type'] == Constants.request_type_close:

                        self.__at_the_end()
                
                else:
                    print('[Error] Data received error.')                    

    def kill(self):
        print('[Info] Killing worker...')
        # On kill worker start after learning todo list
        self.__at_the_end()
