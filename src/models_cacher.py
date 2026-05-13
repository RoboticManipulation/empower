import loader 
import detection
import socket
import importlib
import sys


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: python3 models_cacher.py <use_case> [grasp_object]")
    use_case = args[0]
    grasp_object = " ".join(args[1:]).strip() if len(args) > 1 else None

    loader_instance = loader.Loader(use_case)
    if grasp_object:
        loader_instance.grasp_object = grasp_object
        print(f"Semantic placement grasp object: {grasp_object}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = ('localhost', 10000)
    server_socket.bind(server_address)

    server_socket.listen(1)

    while True:
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()

        try:        
            data = connection.recv(1024)
            message = data.decode()
            if message != "":
                print(message)
            
            if "detection" in message:
                importlib.reload(detection)
                detection_instance = detection.Detection()
                detection_instance.set_loader(loader_instance)
                print("Detection completed")
            else:
                pass
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
