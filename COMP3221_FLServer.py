import sys
import logging
import torch
import torch.nn as nn
from MultinominalLogisticRegression import MultinominalLogisticRegression
import socket

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def aggregate_models(client_models, global_model: MultinominalLogisticRegression) -> MultinominalLogisticRegression:
    num_clients = len(client_models)

    # Initialize the aggregated weights and biases to zero
    aggregated_weights = torch.zeros_like(client_models[0].linear.weight)
    aggregated_biases = torch.zeros_like(client_models[0].linear.bias)

    # Sum the weights and biases from all client models
    for model in client_models:
        aggregated_weights += model.linear.weight
        aggregated_biases += model.linear.bias

    # Calculate the average of the weights and biases
    aggregated_weights /= num_clients
    aggregated_biases /= num_clients

    # Update the global model with the averaged weights and biases
    global_model.linear.weight.data = aggregated_weights
    global_model.linear.bias.data = aggregated_biases

    return global_model


def start():
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
    # Disable the StreamHandler that is attached to the root logger by default
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    # The arguments are valid; you can now start the federated learning server
    logging.info(f"Starting server on port {port_server} with sub-client flag {sub_client}...")
    
    # Your server implementation goes here
    input_size = 28 * 28  # for 28x28 pixel images
    num_classes = 10  # for digits 0 to 9
    model = MultinominalLogisticRegression(input_size, num_classes)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port_server))
            s.listen()
            logging.info(f"Listening on port {port_server}")

            while True:
                pass
    except Exception as e:
        logging.error(f"Exception in server listening")


if __name__ == "__main__":
    start()
