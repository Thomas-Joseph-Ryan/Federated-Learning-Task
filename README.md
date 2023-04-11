# Federated learning Task

In order to run the program, use the commands outlined in the spec.

Ensure you start the server first, then the clients.

After the process has ended, the server will send a 'end' message to the clients which will end their processes too.

Please ensure you do not end a client process before the server process has finished, as this will cause the server process
to block waiting for the client to send a message.