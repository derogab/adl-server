import json
import threading
import time
from replies import Reply

class Worker(threading.Thread):

    running = True
    user = 0

    def __init__(self, socket, data_received):
        threading.Thread.__init__(self)
        self.socket = socket
        self.data_received = data_received
        running = True
        

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
                
                if request and request['status'] == 'OK':

                    data = request['data']
                    print(data)

                    if data['type'] == 'data':
                        
                        self.socket.send(Reply.ack())

                        # do something with data

                    if data['type'] == 'close':

                        self.socket.send(Reply.close())

                        self.running = False
                
                else:
                    print('[Error] Data received error.')
