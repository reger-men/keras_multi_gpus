## keras_multi_gpus

Simple exapmle of data parallelism on keras.

The `multi_gpu_model` keras function replicates a model on different GPUs.

Each GPU take a sub-batches of the training set pool.
    -Apply a model copy on each sub-batch. 
    -Every model copy is executed on a dedicated GPU.
    -The result of each GPU will concatenate on the CPU into one big batch.
