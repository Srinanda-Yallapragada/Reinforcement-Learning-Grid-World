from DQN_alg import *

# Gridsearch for learning rate and discount_factor 
learning_rate=[0.01,0.0001,0.0002]
discount_factor=[0.999,0.99]

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

