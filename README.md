# Distributed_neural_network_using_Tensorflow


It runs the Keras MNIST mlp example across multiple servers.

This sample code runs multiple processes on a single host. It can be configured to run on multiple hosts simply by chaning the host names given in *ClusterSpec*.

##Training the model

Start the parameter server
  python keras_distributed.py --job_name="ps" --task_index=0
  
Start the three workers
  python keras_distributed.py --job_name="worker" --task_index=0
  python keras_distributed.py --job_name="worker" --task_index=1
  python keras_distributed.py --job_name="worker" --task_index=2