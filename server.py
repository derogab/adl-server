import socket, threading

class ClientThread(threading.Thread):
    def __init__(self,clientAddress,clientsocket):
        threading.Thread.__init__(self)
        self.csocket = clientsocket
        print ("New connection added: ", clientAddress)
    def run(self):
        print ("Connection from : ", clientAddress)
        self.csocket.send(bytes("Hi, Server connected..",'utf-8'))
        msg = ''
        while True:
            data = self.csocket.recv(8192)
            msg = data.decode()
            if msg:
                print ("from client", msg)
            self.csocket.send(bytes(msg,'UTF-8'))
        print ("Client at ", clientAddress , " disconnected...")

LOCALHOST = ""
PORT = 8090
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LOCALHOST, PORT))
print("Server started")
print("Waiting for client request..")
while True:
    server.listen(1)
    clientsock, clientAddress = server.accept()
    newthread = ClientThread(clientAddress, clientsock)
    newthread.start()