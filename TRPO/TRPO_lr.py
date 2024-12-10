from TRPO_alg import *

# Gridsearch for learning rate and discount_factor 
# Ideal- 0.005, 0.9
learning_rate=[0.001]
discount_factor=[0.1,0.5,0.9,0.99]

count=1
for lr in learning_rate:
    for df in discount_factor:
        print(f"Learning rate:{lr}, Discount factor:{df}")
        TRPO_run(lr, df)

# learning_rate= 0.0001

# for df in discount_factor:
#     PPO_run(learning_rate, df)

