from DQN_alg import *

# Gridsearch for learning rate and discount_factor 
# 0.002,0.99 or 0.005
learning_rate=[0.00075]
discount_factor=[0.99,.5,0.9,0.99]

count=1
for lr in learning_rate:
    for df in discount_factor:
        print(f"Learning rate:{lr}, Discount factor:{df}")
        DQN_run(lr, df)
        break
    break
        

# learning_rate= 0.0001

# for df in discount_factor:
#     PPO_run(learning_rate, df)

