import argparse

import os
import csv
import math


def run_command(command):
    res = os.system(command)

    if res != 0:
        raise Exception(f"Failure during execution of \n{command}")

# Preliminary definitions
num_particles=100
num_iterations=100
dataset="BDD-Objects"
method='Adaptive'
data_dir='inputs/Models2Repair/BDD100K-Classification'


# Run approach
def run_Adaptive(model='EnetB7', class_id=0, repetition=0, criterion='Stagnation', window=10, threshold=10,gpu=0):
    negative_inputs_dir = f"{data_dir}/models/{model}/negative/{class_id}"
    positive_inputs_dir = f"{data_dir}/models/{model}/positive/"
    model_dir = f"{data_dir}/models/{model}"

    if criterion is None:
        output_dir = f"outputs/Adaptive{criterion}/{model}/C_{class_id}_R_{repetition}"
    else:
        output_dir = f"outputs/Adaptive{criterion}/{model}/C_{class_id}_R_{repetition}_{criterion}_W{window}_T{threshold}"

    # Create output directory if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    command = f"python3 cli.py optimize --dataset={dataset} --method={method} \
    --model_dir={model_dir} \
    --target_data_dir={negative_inputs_dir}\
    --positive_inputs_dir={positive_inputs_dir} --num_input_pos_sampled={positives}\
    --output_dir={output_dir} --num_particles={num_particles} --num_iterations={num_iterations}  \
              --switch_criterion={criterion.lower()} --window={window} --threshold={threshold} --normalized_fitness=on --gpu_num={gpu}"

    print(command)
    run_command(command)

# Fixed values to try
classes = [0, 1, 2, 3, 6, 7, 8, 9, 12]

# Parse arguments
parser = argparse.ArgumentParser(
                    prog = 'AdaptiveRunner',
                    description = 'Runs an adaptive repair experiment')

parser.add_argument('-g','--gpu',default=0,type=int)
parser.add_argument('-b','--begin',default=0,type=int)
parser.add_argument('-e','--end',default=10,type=int)
parser.add_argument('-s','--switch',default='Stagnation')
parser.add_argument('-w','--window',default=10,type=int)
parser.add_argument('-t','--threshold',default=0.01,type=float)
parser.add_argument('-p','--positives',default=200,type=int)

args = vars(parser.parse_args())

gpu_num=args['gpu']
begin=args['begin']
end=args['end']
switch_criterion=args['switch']
window = args['window']
threshold = args['threshold']
positives = args['positives']

if switch_criterion == 'Stagnation':
    window_set = [10, 15, 20]
    threshold_set = [0.1, 0.02, 0.01, 0.005]
else:
    window_set = [10,15,20]
    threshold_set = [0.8,0.6,0.7]

if not (window is None):
    window_set = [window]

if not (threshold is None):
    threshold_set = [threshold]

# Run experiments
print(f"Doing runs {begin}-{end}, switching by {switch_criterion}")
print(f"Testing windows {window_set}, thresholds {threshold_set}")

for t in threshold_set:

    for w in window_set:

        for rep in range(begin,end):

            print(f"REPETITION {rep}, WINDOW {w}, THRESHOLD {t}")

            for c in classes:
                print(f"\tRepairing class {c}", end='...')
                run_Adaptive('EnetB7', class_id=c, repetition=rep, criterion=switch_criterion, window=w, threshold=t,gpu=gpu_num)
                print("DONE")
