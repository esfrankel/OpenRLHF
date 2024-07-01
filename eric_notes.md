 - need to run `module load cuda/12.4.1` at the start of interactive session
 - need to run `module load ompi/4.1.0` at the start of interactive session
 - if you make any changes to the openrlhf code, you need to run `python setup.py install` in the openrlhf directory

 ## Arguments to Play With
 - pretrain (initial model)
 - train_batch_size
 - micro_train_batch_size
 - learning rate
 - max_epochs
 - max_len
 - zero_stage ?