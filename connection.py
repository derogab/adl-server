import json
import threading
from worker import Worker
from replies import Reply

# Connection
# Class for a single connection
class Connection(threading.Thread):
    
    def __init__(self, clientAddress, clientsocket):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.caddress = clientAddress
        print ("New connection added: ", clientAddress)
    
    def run(self):
        print ("Connection from : ", self.caddress)

        # Send back a message
        try:
            self.csocket.send(Reply.ack())
        except:
            print('[Error] No handshake')

        # Get all data from this connection
        while True:
            data = self.csocket.recv(1024)

            # create a separate thread for data analysis
            my_worker = Worker(self.csocket, data)
            my_worker.start()

            if not my_worker.running:
                print("Force disconnect")
                break
         
        print ("Client at ", self.caddress , " disconnected...")