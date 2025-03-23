from PPO_alg import *

# Average for learning rate and discount_factor 
lr=0.0002
df=0.99

for run in range(10):
    PPO_run(lr, df)



