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
        # Init thread
        threading.Thread.__init__(self)
        # Init data
        self.csocket = clientsocket
        self.caddress = clientAddress
        print ('[Info] New connection added: ', clientAddress)

    def __send_message(self, msg):
        try:
            self.csocket.send(msg)
        except:
            print('[Warning] Connection already closed by client.')
    
    def run(self):
        print ('[Info] Connection from : ', self.caddress)

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

        # create queue
        queue = ''

        # Get all data from this connection
        while True:

            try:
                data = self.csocket.recv(2048)
            except:
                print('[Warning] Connection closed by client.')

            if data:
                # decode data
                data = data.decode()
                # add data to queue
                queue = queue + data
                # Reinit timeout
                timeout = 0
            else:
                # Increse timeout
                timeout = timeout + 1

            if queue:
                # Cut point
                cut = queue.rfind('\n')+1
                # Create a separate thread for data analysis
                my_worker = Worker(self.csocket, queue[:cut], wizard, teacher)
                my_worker.start()
                # Pop queue
                queue = queue[cut:]

            # Check timeout
            if timeout > Constants.connection_max_timeout:
                print('[Info] Force disconnect')
                break

        # Send closing a message
        self.__send_message(Reply.goodbye())
        
        print ('[Info] Client at ', self.caddress , ' disconnected...')