
from network_utils import *

# Downloading MNIST data
train_data = dsets.MNIST(root='./data', train=True,
                         transform=transforms.ToTensor(), download=True)

test_data = dsets.MNIST(root='./data', train=False,
                        transform=transforms.ToTensor())

# Loading the data
batch_size = 500  # the size of input data taken for one iteration

train_gen = torch.utils.data.DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True)

test_gen = torch.utils.data.DataLoader(dataset=test_data,
                                       batch_size=batch_size,
                                       shuffle=False)

# Define Hyperparameters
input_size = 2 * 784  # img_size = (28,28) ---> 28*28=784 in total x 2 for side by side images
num_classes = 10  # number of output classes, discrete range [0,9]
num_epochs = 200  # number of times which the entire dataset is passed through the model
lr = 1e-2  # learning rate

# Define Dendrite Hyperparameters
excitatory_synapses_per_dendrite_branch = 50
inhibitory_synapses_per_dendrite_branch = 1
dendrite_layers_n = 1
dendrite_branch_per_dendrite = [1]
dendrite_per_soma = 3400
e_to_i_ratio = excitatory_synapses_per_dendrite_branch / inhibitory_synapses_per_dendrite_branch
soma_n = num_classes
synapses_per_inhibitory_unit = 200
inhibitory_units_n = 1
sigmoid_offset = 5
dendrite_branch_n = [None] * dendrite_layers_n
dendrite_branch_n[0] = dendrite_per_soma * soma_n

for i in range(1, dendrite_layers_n):
    dendrite_branch_n[i] = dendrite_branch_n[i - 1] * dendrite_branch_per_dendrite[i - 1]

shunting = True

print(dendrite_branch_n)
sum(dendrite_branch_n)

# Index arrays
index_list1 = torch.tensor(range(batch_size))
index_list2 = torch.tensor(list(range(1, batch_size)) + [0])
modulo_base = num_classes

def sum_labels(labels_in):
    return torch.remainder(labels_in[index_list1] + labels_in[index_list2], modulo_base)

def transform_images(images_in):
    return torch.cat((images_in[index_list1, :], images_in[index_list2, :]), 1)

# Build the models
net = Net().to(device)
net.initWeights()

net_with_attention = NetWithAttention().to(device)
net_with_attention.initWeights()

# Train and evaluate the network without attention
trained_net, training_losses = train_network(net, train_gen, test_gen, num_epochs, lr, 0.98, use_new_input_for_inhibition=False)
accuracy, all_preds, all_labels = evaluate_network(trained_net, test_gen)

# Plot confusion matrix
plot_confusion_matrix(all_labels, all_preds, accuracy)

# Analyze weights, variance, and information flow
weights = analyze_weights(trained_net)
inputs, outputs = next(iter(train_gen))
inputs, outputs = transform_images(inputs.view(-1, 28*28)).to(device), sum_labels(outputs).to(device)
info_flow = compute_information_flow(trained_net, inputs, outputs, 'dendrite_layers')
mutual_info = compute_mutual_information(inputs.cpu().numpy(), outputs.cpu().numpy())
input_variance, output_variance = analyze_variance_signals(inputs, outputs)

# Visualize the trained model's attention weights
attention_weights = visualize_attention_weights(net_with_attention, 'attention_layers')

# Train and evaluate the network with attention
net_with_attention, training_losses_attention = train_network(net_with_attention, train_gen, test_gen, num_epochs, lr, 0.98, use_new_input_for_inhibition=False)
accuracy_attention, all_preds_attention, all_labels_attention = evaluate_network(net_with_attention, test_gen)

# Plot confusion matrix for the network with attention
plot_confusion_matrix(all_labels_attention, all_preds_attention, accuracy_attention)

# Analyze weights, variance, and information flow for the network with attention
weights_attention = analyze_weights(net_with_attention)
info_flow_attention = compute_information_flow(net_with_attention, inputs, outputs, 'dendrite_layers')
mutual_info_attention = compute_mutual_information(inputs.cpu().numpy(), outputs.cpu().numpy())
input_variance_attention, output_variance_attention = analyze

variance_signals(inputs, outputs)

# Visualize the trained model's attention weights again after training
attention_weights_after_training = visualize_attention_weights(net_with_attention, 'attention_layers')

# Plot training metrics
plot_training_metrics([], [], training_losses, [])
plot_training_metrics([], [], training_losses_attention, [])