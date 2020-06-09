import json
import threading
from worker import Worker
from replies import Reply
from constants import Constants
from wizard import Wizard
from teacher import Teacher

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
            self.csocket.send(Reply.handshake())
        except:
            print('[Error] No handshake')

        # Init timeout
        timeout = 0

        # Create teacher and wizard for each connection
        teacher = Teacher()
        wizard = Wizard()

        # Get all data from this connection
        while True:
            data = self.csocket.recv(2048)

            if data:
                # Create a separate thread for data analysis
                my_worker = Worker(self.csocket, data, wizard, teacher)
                my_worker.start()
                # Reinit timeout
                timeout = 0
            else:
                # Increse timeout
                timeout = timeout + 1

            # Check timeout
            if timeout > Constants.connection_max_timeout:
                print("Force disconnect")
                break
         

        # Send closing a message
        try:
            self.csocket.send(Reply.goodbye())
        except:
            print('[Warning] Connection already closed by client.')

        print ("Client at ", self.caddress , " disconnected...")