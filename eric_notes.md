 - need to run `module load cuda/12.4.1` at the start of interactive session
 - need to run `module load ompi/4.1.0` at the start of interactive session
 - if you make any changes to the openrlhf code, you need to run `python setup.py install` in the openrlhf directory
 - getting an initial version working with llama7b takes 4 a40 gpus and about 19 hours
 - bloom1b7 takes 1 a40 gpu and about 8 hours
 - ultrafeedback used `gpt-4-0613`


 ## Arguments to Play With
 - pretrain (initial model)
 - train_batch_size
 - micro_train_batch_size
 - learning rate
 - max_epochs
