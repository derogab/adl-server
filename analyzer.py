import json
import threading

class Analyzer(threading.Thread):

    running = True

    def __init__(self, socket, data_received):
        threading.Thread.__init__(self)
        self.socket = socket
        self.data_received = data_received
        running = True
        

    def run(self):
        msg = self.data_received.strip().decode()
        reply = self.socket.send

        if msg:

            array = msg.split('\n')

            for x in array:

                try:
                    info = json.loads(x)
                except:
                    print('[Error] JSON decode failed.')
                
                if info and info['status'] == 'OK':

                    response = info['response']

                    if response['type'] == 'data':
                        
                        res = {
                            "status": "OK",
                            "type": "data",
                            "data": {
                                "x": str(response['data']['x']), 
                                "y": str(response['data']['y']), 
                                "z": str(response['data']['z'])
                            }
                        }

                    if response['type'] == 'close':

                        res = {
                            "status": "OK",
                            "type": "close"
                        }

                        self.running = False         

                    reply(bytes(str(res)+"\n", 'UTF-8'))
                
                else:
                    print('[Error] Data received error.')
