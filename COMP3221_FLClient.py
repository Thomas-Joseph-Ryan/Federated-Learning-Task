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
from torch.utils.data import DataLoader
import copy
import threading
import signal

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

SERVER_PORT = 6000
HEADER_FORMAT = "{type}:{length}:{port}"
HEADER_SIZE = 40  # Choose a fixed size for the header

INPUT_SIZE = 28*28
NUM_CLASSES = 10

stop_event = threading.Event()

# This class is provided by the W6 tutorial
class UserAVG():
    def __init__(self, client_id, model, learning_rate, batch_size, x_train, y_train, x_test, y_test, train_samples, test_samples):

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = x_train, y_train, x_test, y_test, train_samples, test_samples 
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        self.trainloader = DataLoader(self.train_data, batch_size) # type: ignore
        self.testloader = DataLoader(self.test_data, self.test_samples) # type: ignore

        self.loss = nn.NLLLoss()

        self.model = copy.deepcopy(model)

        self.id = client_id

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            
    def train(self, epochs):
        LOSS = 0
        self.model.train() # Tells the model that you are starting to train it
        for epoch in range(1, epochs + 1):
            self.model.train() # Only need this once 
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad() #Sets the gradients of all optimised code to zero
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data #type: ignore
    
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print(str(self.id) + ", Accuracy of client ",self.id, " is: ", test_acc)
        return test_acc

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
    file_handler = logging.FileHandler(f"./logs/{client_id}.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    # Disable the StreamHandler that is attached to the root logger by default
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    X_train, y_train, X_test, y_test, train_samples, test_samples= get_data(client_id)
    send_handshake(client_id, train_samples, port_client)
    while True:
        server_data, data_type = listen(port_client) # type: ignore
        if data_type == "string":
            if server_data == "end":
                break
        global_model = MultinominalLogisticRegression.deserialize(server_data, INPUT_SIZE, NUM_CLASSES)
        # Create a UserAVG instance

        user = UserAVG(client_id, global_model, learning_rate=0.01, batch_size=32, x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, train_samples=train_samples, test_samples=test_samples)

        # Train the client model
        loss = user.train(epochs=10)
        print(f"Training loss for client {client_id}: {loss}")

        # Test the client model
        test_accuracy = user.test()
        print(f"Test accuracy for client {client_id}: {test_accuracy}")

        # Send the local model back to the server
        send_local_model(user.model, port_client)


    

def send_handshake(id: str, data_size: int, listening_port: int):
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
                header = HEADER_FORMAT.format(type="string", length=len(encoded_data), port=listening_port).encode('utf-8').ljust(HEADER_SIZE)            
                s.sendall(header + encoded_data)
        except ConnectionRefusedError:
            logging.error(f"Connection was refused by server on port {SERVER_PORT}")
        except Exception as e:
            logging.error(f"Exception in broadcast: {e}")

def send_local_model(model : MultinominalLogisticRegression, listening_port: int):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Trying to connect to server on port {SERVER_PORT}")
                s.connect(('localhost', SERVER_PORT))
                logging.info(f"Sending local model")
                encoded_data = base64.b64encode(model.serialize())
                header = HEADER_FORMAT.format(type="pickle", length=len(encoded_data), port=listening_port).encode('utf-8').ljust(HEADER_SIZE)            
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

            s.settimeout(0.5)

            while True:
                try:
                    conn, (client_addr, client_port) = s.accept()
                    break
                except socket.timeout:
                    check_stop_event()
                    continue
            with conn:
                logging.info(f"Connected by {client_port}")

                header = conn.recv(HEADER_SIZE).decode('utf-8').strip()
                data_type, data_length, server_port = header.split(':')
                data_length = int(data_length)
                server_port = int(server_port)

                encoded_data = conn.recv(data_length)
                if data_type == "pickle":
                    decoded_data = base64.b64decode(encoded_data)
                elif data_type == "string":
                    decoded_data = encoded_data.decode('utf-8')
                else:
                    raise ValueError("Invalid data type")
                    
                logging.debug(f"Recieved data: {decoded_data}")
                return (decoded_data, data_type)
    except Exception as e:
        logging.error(f"Exception in client listening: e")

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

def check_stop_event():
    if stop_event.is_set():
        sys.exit(0)

def quit_gracefully(signum, frame) -> None:
    logging.info(f"Received signal {signum}, quitting gracefully...")
    stop_event.set()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, quit_gracefully)
    signal.signal(signal.SIGTERM, quit_gracefully)
    main()
