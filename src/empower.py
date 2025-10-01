import socket
import loader 
import importlib
import pipeline
import sys

if __name__ == '__main__':
    ###ACQUIRE DATA
    loader_instance = loader.Loader()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10000)
    server_socket.bind(server_address)
    server_socket.listen(1)

    while True:
        try:
            print('Waiting for a connection...')
            connection, client_address = server_socket.accept()
    
            data = connection.recv(1024)
            message = data.decode()
            if message != "":
                print(message)
            
            if "detection" in message:
                importlib.reload(pipeline)
                task = message.split("|")[1]
                pipeline_instance = pipeline.PipelineFinal(task)
                pipeline_instance.set_loader(loader_instance)
                pipeline_instance.run_pipeline()
            elif "exit" in message:
                sys.exit(0)
        except Exception as e:
            print("Error occurred")
            print("IN the script the following error occurred: ", e)
         
