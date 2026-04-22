import itertools
import subprocess
import datetime

# Define hyperparameter values
lr_values = [0.01, 0.0075, 0.005]
batch_sizes = [50]
weight_decay_values = [0.01, 0.0075, 0.005]
scheduler_gamma_values = [0.2]

train_path = "Training_Perlin40k_Pperlin50k_MatchNorm.mat"
train_size = 80000
train_test_split = 0.8
loss_function = 'mse'
epochs = 250

# Generate all combinations
hyperparameter_combinations = list(itertools.product(lr_values, batch_sizes,
                                                     weight_decay_values, scheduler_gamma_values))

for lr, batch_size, weight_decay, scheduler_gamma in hyperparameter_combinations:
    exp_name = f"Test_{lr}lr_{batch_size}bs_{weight_decay}l2reg_{scheduler_gamma}gamma_{loss_function}"
    command = (f"python -m specunet_pkg main.py --train --test_exp "
               f"--exp_name {exp_name} "
               f"--train_path {train_path} "
               f"--train_size {train_size} "
               f"--train_test_split {train_test_split}"
               f"--lr {lr} --bs {batch_size} "
               f"--weight_decay {weight_decay} --scheduler_gamma {scheduler_gamma} "
               f"--loss_fn {loss_function} "
               f"--epochs {epochs} ")
    subprocess.run(command, shell=True)

print("Finished at time: ", datetime.datetime.now())