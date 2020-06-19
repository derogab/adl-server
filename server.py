import socket
from constants import Constants
from connection import Connection

# server socket
LOCALHOST = Constants.server_host
PORT = Constants.server_port
# initialize & bind socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LOCALHOST, PORT))
# start the server
print('[Info] Server started')
print('[Info] Waiting for client request..')
while True:
    # listen for connections
    server.listen(1)
    # accept a connection
    clientsock, clientAddress = server.accept()
    # create a separate thread for this connection
    new_connection = Connection(clientAddress, clientsock)
    new_connection.start()
