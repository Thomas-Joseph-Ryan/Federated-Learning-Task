# Federated learning Task

I have not uploaded the FLdata in the submission to reduce package size, please insert the directory 
FLdata from the assignment spec into the same directory of the program or download the MNIST training data
externally (possible link -> https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

In order to run the program, use the commands outlined in the spec.

Ensure you start the server first, then the clients.

After the process has ended, the server will send a 'end' message to the clients which will end their processes too.

Please ensure you do not end a client process before the server process has finished, as this will cause the server process
to block waiting for the client to send a message.

# NOTE

When no subsampling, the model can achieve >90% accuracy on both mini-batch GD and GD. 
With subsampling however, the model cannot achieve this as discussed in the report. Since,
my model can comfortably achieve > 90% accuracy when using consistent methods, I believe
my implementation does complete all of the criteria for the code in the canvas.