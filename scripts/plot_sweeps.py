import json
import os
import matplotlib.pyplot as plt

# Base directory where the outputs are saved
output_base_dir = "/n/holylabs/LABS/kempner_dev/Users/hsafaai/results/dendritic_modeling"
# Job ID directory
job_id_dir = "dendritic_modeling_sweep_57509333"
# Sweep values for `ie_synapses_per_branch_per_layer`
sweep_values = [1, 2, 3, 4, 5, 10, 20, 40]

train_accuracies = []
valid_accuracies = []
test_accuracies = []

for i, sweep_value in enumerate(sweep_values):
    # Path to each task-specific folder
    task_dir = f"{job_id_dir}/57509333_{i}"
    output_dir = os.path.join(output_base_dir, task_dir)

    # Load the losses.json file for this run
    try:
        with open(os.path.join(output_dir, 'losses.json'), 'r') as f:
            performance = json.load(f)

        # Assuming `losses.json` contains accuracy values:
        train_accuracies.append(performance.get('train accuracy', None))
        valid_accuracies.append(performance.get('valid accuracy', None))
        test_accuracies.append(performance.get('test accuracy', None))

    except FileNotFoundError:
        print(f"Performance file not found for sweep value {sweep_value}")
        train_accuracies.append(None)
        valid_accuracies.append(None)
        test_accuracies.append(None)

# Plot accuracy vs `ie_synapses_per_branch_per_layer`
plt.figure()
plt.plot(sweep_values, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(sweep_values, valid_accuracies, label='Validation Accuracy', marker='o')
plt.plot(sweep_values, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('ie_synapses_per_branch_per_layer')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. ie_synapses_per_branch_per_layer')
plt.legend()
plt.show()