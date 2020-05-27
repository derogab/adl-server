import json
import threading
from analyzer import Analyzer

class Receiver(threading.Thread):
    
    def __init__(self, clientAddress, clientsocket):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        self.caddress = clientAddress
        print ("New connection added: ", clientAddress)
    
    def run(self):
        print ("Connection from : ", self.caddress)

        try:
            self.csocket.send(bytes(str({"status": "OK", "type": "connection"})+"\n", 'UTF-8'))
        except:
            print('[Error] No handshake')

        while True:
            data = self.csocket.recv(1024)

            mythread = Analyzer(self.csocket, data)
            mythread.start()

            if not mythread.running:
                break
         
        print ("Client at ", self.caddress , " disconnected...")