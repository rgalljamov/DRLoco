
## SB2 to SB3 Migration Log 

### Changes to the algorithm

1. No longer using a custom policy and custom PPO implentation
2. No longer using custom distributions and therefore not STD clipping anymore, which should be fine. 
3. There are no 'nminibatches' parameter any more in SB3, which might be relevant for the mirroring approaches.


### Use CPU instead of GPU for training

1. Fool Python to think there is no CUDA device available with
`
	from os import environ
 	environ["CUDA_VISIBLE_DEVICES"]="" `

2. To avoid massive slow-down when using torch with cpu, play with the number of threads torch is using
`
    import torch
    torch.set_num_threads(1) `

    a. INFO: We found setting the number of threads to the number of parallel environments we use to collect the experiences to result in the highest training speed. 