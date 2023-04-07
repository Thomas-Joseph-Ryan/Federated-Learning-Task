import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import socket
import logging
import base64
import pickle
from MultinominalLogisticRegression import MultinominalLogisticRegression

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

SERVER_PORT = 6000
HEADER_FORMAT = "{type}:{length}:"
HEADER_SIZE = 40  # Choose a fixed size for the header

def main():
    if len(sys.argv) != 4:
        print("Usage: python COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>")
        sys.exit(1)

    client_id = sys.argv[1]
    if client_id not in ["client1", "client2", "client3", "client4", "client5"]:
        print("Invalid Client-id. Valid options: client1, client2, client3, client4, client5")
        sys.exit(1)

    try:
        port_client = int(sys.argv[2])
        if port_client < 6001 or port_client > 6005:
            raise ValueError()
    except ValueError:
        print("Invalid Port-Client. Port number should be an integer between 6001 and 6005.")
        sys.exit(1)

    try:
        opt_method = int(sys.argv[3])
        if opt_method not in [0, 1]:
            raise ValueError()
    except ValueError:
        print("Invalid Opt-Method. Choose 0 for Gradient Descent or 1 for Mini-Batch Gradient Descent.")
        sys.exit(1)

    #SET UP LOGGING
    file_handler = logging.FileHandler("./logs/server.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    # Disable the StreamHandler that is attached to the root logger by default
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    X_train, y_train, X_test, y_test, train_samples, test_samples= get_data(client_id)
    send_handshake(client_id, train_samples)
    

def send_handshake(id: str, data_size: int):
        '''
        Handshake message will be of the form
        "hello <client_id> <data_size>" where data size is the number
        of samples in the training data
         '''
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Trying to connect to server on port {SERVER_PORT}")
                s.connect(('localhost', SERVER_PORT))
                logging.info(f"Sending handshake")
                encoded_data = f"hello {id} {data_size}".encode('utf-8')
                header = HEADER_FORMAT.format(type="string", length=len(encoded_data)).encode('utf-8').ljust(HEADER_SIZE)            
                s.sendall(header + encoded_data)
        except ConnectionRefusedError:
            logging.error(f"Connection was refused by server on port {SERVER_PORT}")
        except Exception as e:
            logging.error(f"Exception in broadcast: {e}")

def send_local_model(model : MultinominalLogisticRegression):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Trying to connect to server on port {SERVER_PORT}")
                s.connect(('localhost', SERVER_PORT))
                logging.info(f"Sending local model")
                encoded_data = base64.b64encode(model.serialize())
                header = HEADER_FORMAT.format(type="pickle", length=len(encoded_data)).encode('utf-8').ljust(HEADER_SIZE)            
                s.sendall(header + encoded_data)
        except ConnectionRefusedError:
            logging.error(f"Connection was refused by server on port {SERVER_PORT}")
        except Exception as e:
            logging.error(f"Exception in broadcast: {e}")

def listen(port: int):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            s.listen()
            logging.info(f"Listening on port {port}")

            conn, (client_addr, client_port) = s.accept()

            with conn:
                logging.info(f"Connected by {client_port}")

                header = conn.recv(HEADER_SIZE).decode('utf-8').strip()
                data_type, data_length = header.split(':')
                data_length = int(data_length)

                encoded_data = conn.recv(data_length)
                if data_type == "pickle":
                    decoded_data = base64.b64decode(encoded_data)
                elif data_type == "string":
                    decoded_data = encoded_data.decode('utf-8')
                else:
                    raise ValueError("Invalid data type")
                    
                logging.debug(f"Recieved data: {decoded_data}")
                return decoded_data
    except Exception as e:
        logging.error(f"Exception in client listening")

def get_data(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_" + str(id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_" + str(id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples

if __name__ == "__main__":
    main()
