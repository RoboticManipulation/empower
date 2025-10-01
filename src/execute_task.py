import socket
import argparse

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)
client_socket.connect(server_address)

argparse.ArgumentParser(description='Send a message to the server.')
parser = argparse.ArgumentParser()
parser.add_argument('--message', type=str, help='Message to send', required=True)
args = parser.parse_args()
message = args.message

try:
    print('Sending: ', message)
    client_socket.sendall(message.encode())
finally:
    client_socket.close()