import sys
import logging
import torch
import torch.nn as nn
from MultinominalLogisticRegression import MultinominalLogisticRegression
import socket
import threading
import signal
import queue
import time
import random
import base64

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

stop_event = threading.Event()

clients: dict[int, dict[str, str | int]] = {}

inbound_information = queue.Queue()

SUB_CLIENT_NUMBER = 2

HEADER_FORMAT = "{type}:{length}:{port}"
HEADER_SIZE = 40  # Choose a fixed size for the header

SERVER_PORT = 6000


def main():
    if len(sys.argv) != 3:
        print("Usage: python COMP3221_FLServer.py <Port-Server> <Sub-client>")
        sys.exit(1)

    try:
        port_server = int(sys.argv[1])
        sub_client = int(sys.argv[2])
    except ValueError:
        print("Error: Both <Port-Server> and <Sub-client> must be integers.")
        sys.exit(1)

    if port_server != 6000:
        print("Error: <Port-Server> must be 6000.")
        sys.exit(1)

    if sub_client not in (0, 1):
        print("Error: <Sub-client> must be either 0 or 1.")
        sys.exit(1)

    #SET UP LOGGING
    file_handler = logging.FileHandler("./logs/server.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    # SERVER
    logging.info(f"Starting server on port {port_server} with sub-client flag {sub_client}...")
    
    input_size = 28 * 28  # for 28x28 pixel images
    num_classes = 10  # for digits 0 to 9
    global_model = MultinominalLogisticRegression(input_size, num_classes)

    listen_thread = threading.Thread(target=thread_listen, args=(port_server,))
    listen_thread.start()

    while True:
        # Wait indefinately for client to join
        try:
            inbound_info = inbound_information.get(block=True, timeout=0.5)
        except queue.Empty:
            check_stop_event()
            continue
        message, client_port, data_type = inbound_info[0], inbound_info[1], inbound_info[2]
        logging.info(f"Message recieved from client {message}, {client_port}, {data_type}")
        if data_type == "string":
            if message.split(" ")[0] == "hello":
                logging.info(f"New client added {message}")
                add_client(message, client_port)
            else: 
                continue
        else:
            continue

        # Once a single client joins, 30 second timer begins
        start_time = time.time()
        timeout = 30  
        while True:
            check_stop_event()
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time >= timeout:
                break

            try:
                inbound_info = inbound_information.get(block=True, timeout=0.5)
                message, client_port = inbound_info[0], inbound_info[1]
                if message.split(" ")[0] == "hello":
                    add_client(message, client_port)
            except queue.Empty:
                continue
        break

    iteration = 1
    iteration_data = []
    broadcast_model(global_model)
    while iteration <= 100:
        print(f"Global Iteration {iteration}:")
        num_clients = len(clients)
        print(f"Total Number of clients: {num_clients}")
        while True:
            try:
                inbound_info = inbound_information.get(block=True, timeout=0.5)
            except queue.Empty:
                check_stop_event()
                continue
            message, client_port, data_type = inbound_info[0], inbound_info[1], inbound_info[2]
            if data_type == "string":
                if message.split(" ")[0] == "hello":
                    add_client(message, client_port)
            else:
                client_data = clients[client_port]
                client_id = client_data["id"]
                print(f"Getting local model from {client_id}")
                local_model = MultinominalLogisticRegression.deserialize(message, input_size, num_classes)
                iteration_data.append((local_model, client_port))
            
            if inbound_information.qsize() == 0 and len(iteration_data) == num_clients:
                break
        print("Aggregating new global model")
        global_model = aggregate_models(iteration_data, global_model, sub_client)
        print("Broadcasting new global model\n")
        broadcast_model(global_model)
    broadcast_end()
        
def aggregate_models(client_models: list[tuple[MultinominalLogisticRegression, int]], 
                     global_model: MultinominalLogisticRegression,
                     sub_client: int) -> MultinominalLogisticRegression:
    if sub_client == 0 or len(client_models) < SUB_CLIENT_NUMBER:
        num_clients = len(client_models)

        # Initialize the aggregated weights and biases to zero
        aggregated_weights = torch.zeros_like(client_models[0][0].linear.weight)
        aggregated_biases = torch.zeros_like(client_models[0][0].linear.bias)

        # Sum the weights and biases from all client models
        for model in client_models:
            aggregated_weights += model[0].linear.weight
            aggregated_biases += model[0].linear.bias

        # Calculate the average of the weights and biases
        aggregated_weights /= num_clients
        aggregated_biases /= num_clients

        # Update the global model with the averaged weights and biases
        global_model.linear.weight.data = aggregated_weights
        global_model.linear.bias.data = aggregated_biases

    elif sub_client == 1 and len(client_models) >= SUB_CLIENT_NUMBER:
        num_clients = SUB_CLIENT_NUMBER

        # Initialize the aggregated weights and biases to zero
        aggregated_weights = torch.zeros_like(client_models[0][0].linear.weight)
        aggregated_biases = torch.zeros_like(client_models[0][0].linear.bias)

        # Choose 2 client models at random
        selected_client_models = random.sample(client_models, num_clients)
        # Sum the weights and biases from all client models
        for model in selected_client_models:
            aggregated_weights += model[0].linear.weight
            aggregated_biases += model[0].linear.bias

        # Calculate the average of the weights and biases
        aggregated_weights /= num_clients
        aggregated_biases /= num_clients

        # Update the global model with the averaged weights and biases
        global_model.linear.weight.data = aggregated_weights
        global_model.linear.bias.data = aggregated_biases
    return global_model

def thread_listen(port: int):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            s.listen()
            logging.info(f"Listening on port {port}")

            s.settimeout(5)
            while not stop_event.is_set():

                try:
                    conn, (client_addr, client_port) = s.accept()
                except socket.timeout:
                    continue
                except Exception as e:
                    logging.error(f"Exception when accepting client: {e}")
                    continue

                with conn:
                    logging.info(f"Connected by {client_port}")

                    try:
                        header = conn.recv(HEADER_SIZE).decode('utf-8').strip()
                        logging.debug(f"Header: {header}")
                        data_type, data_length, client_port = header.split(':')
                        data_length = int(data_length)
                        client_port = int(client_port)
                    except Exception as e:
                        logging.error(f"Exception while processing header: {e}")
                        continue

                    try:
                        encoded_data = conn.recv(data_length)
                        if data_type == "pickle":
                            decoded_data = base64.b64decode(encoded_data)
                        elif data_type == "string":
                            decoded_data = encoded_data.decode('utf-8')
                        else:
                            raise ValueError("Invalid data type")
                    except Exception as e:
                        logging.error(f"Exception while processing data: {e}")
                        continue
                        
                    logging.debug(f"Recieved data: {decoded_data}")
                    inbound_information.put((decoded_data, client_port, data_type))
    except Exception as e:
        logging.error(f"Exception in server listening: {e}")

def broadcast_model(model : MultinominalLogisticRegression):
    data = model.serialize()
    data = base64.b64encode(data)
    for client in clients:
        client_port : int = client
        client_data = clients[client_port]
        client_id = client_data["id"]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Trying to connect to {client_id} on port {client_port}")
                s.connect(('localhost', client_port))
                header = HEADER_FORMAT.format(type="pickle", length=len(data), port=SERVER_PORT).encode('utf-8').ljust(HEADER_SIZE)
                s.sendall(header + data)
        except ConnectionRefusedError:
            logging.error(f"Connection was refused by {client_id} on port {client_port}")
        except Exception as e:
            logging.error(f"Exception in broadcast: {e}")

def broadcast_end():
    data = "end"
    data = data.encode('utf-8')
    for client in clients:
        client_port : int = client
        client_data = clients[client_port]
        client_id = client_data["id"]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                logging.info(f"Trying to connect to {client_id} on port {client_port} to end client process")
                s.connect(('localhost', client_port))
                header = HEADER_FORMAT.format(type="string", length=len(data), port=SERVER_PORT).encode('utf-8').ljust(HEADER_SIZE)
                s.sendall(header + data)
        except ConnectionRefusedError:
            logging.error(f"Connection was refused by {client_id} on port {client_port}")
        except Exception as e:
            logging.error(f"Exception in broadcast: {e}")

def add_client(hand_shake_message : str, client_port : int):
    '''
        Handshake message will be of the form
        "hello <client_id> <data_size> <client_listening_port>" where data size is the number
        of samples in the training data
    '''
    split_msg = hand_shake_message.split(" ")
    clients[client_port] = {"id": split_msg[1], "data_size": int(split_msg[2])}


def quit_gracefully(signum, frame) -> None:
    logging.info(f"Received signal {signum}, quitting gracefully...")
    stop_event.set()
    broadcast_end()
    sys.exit(0)

def check_stop_event():
    if stop_event.is_set():
        broadcast_end()
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, quit_gracefully)
    signal.signal(signal.SIGTERM, quit_gracefully)
    main()
