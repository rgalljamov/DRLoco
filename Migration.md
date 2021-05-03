
## SB2 to SB3 Migration Log 

### Changes to the algorithm

1. No longer using a custom policy and custom PPO implentation
2. No longer using custom distributions and therefore not STD clipping anymore, which should be fine. 
3. There are no 'nminibatches' parameter any more in SB3, which might be relevant for the mirroring approaches.