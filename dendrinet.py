#@title Import Dependencies

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

%matplotlib inline





#@title Function to plot an image with the scale color bar
def plotWithColor(data, title=''):
    if torch.is_tensor(data):
    data=data.cpu().numpy()
    ax = plt.subplot()
    im = ax.imshow(data)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    if len(title)>0:
    plt.title(title)

    plt.show()






#@title Downloading MNIST data

train_data = dsets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = dsets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())






#@title Loading the data

batch_size = 100 # the size of input data took for one iteration

train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = batch_size,
                                             shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = batch_size,
                                      shuffle = False)







#@title Define Hyperparameters
input_size = 784 # img_size = (28,28) ---> 28*28=784 in total x 2 for side by side images
num_classes = 10 # number of output classes discrete range [0,9]
num_epochs = 5 # 20 # number of times which the entire dataset is passed throughout the model
lr = 1e-2 # size of step





#@title Define Dendrite Hyperparameters

excitatory_synapses_per_dendrite_branch=50
inhibitory_synapses_per_dendrite_branch=1

dendrite_layers_n = 3
dendrite_branch_per_dendrite = [3, 3] # list with one fewer entries than dendrite_layers_n
dendrite_per_soma=1


# dendrite_branch_per_soma=dendrite_branch_per_dendrite*dendrite_per_soma

e_to_i_ratio=excitatory_synapses_per_dendrite_branch/inhibitory_synapses_per_dendrite_branch

soma_n=num_classes
synapses_per_inhibitory_unit=200
inhibitory_units_n=40

sigmoid_offset=5

# dendrite_n=soma_n*dendrite_per_soma
# dendrite_branch_n=dendrite_n*dendrite_branch_per_dendrite

dendrite_branch_n = [None] * dendrite_layers_n
dendrite_branch_n[0] = dendrite_per_soma * soma_n

for i in range(1, dendrite_layers_n):
    dendrite_branch_n[i] = dendrite_branch_n[i-1] * dendrite_branch_per_dendrite[i-1]

shunting=False







#@title Index arrays
index_list1=torch.tensor(range(batch_size))
index_list2=torch.tensor(list(range(1,batch_size))+[0])
modulo_base=num_classes

def sum_labels(labels_in):
  return torch.remainder(labels_in[index_list1]+labels_in[index_list2], modulo_base)

def transform_images(images_in):
  return torch.cat((images_in[index_list1, :], images_in[index_list2, :]), 1)







#@title Define layer class that only keeps K top weights
class TopKLinear(nn.Linear):
  """ Custom Linear layer that only uses the stongest K weights """
  def __init__(self, in_features, out_features, keep_in_features, bias=False):
    super().__init__(in_features, out_features, bias=bias)
    self.keep_in_features = keep_in_features
    self.topk_indices=[]
    self.mask=0*self.weight.data
    self.net_weight=0*self.weight.data

  def forward(self, x):
    self.mask=0*self.weight.data
    self.topk_indices=torch.topk(self.weight, self.keep_in_features, 1, largest=True)[1]
    self.mask[torch.arange(self.weight.shape[0])[:,None], self.topk_indices] = 1  ##  Change those values to 1
    self.net_weight=self.mask*self.weight
    w_times_x= torch.mm(x, self.net_weight.t())
    return w_times_x






from math import floor
#@title Define layer class that uses a block structure
class BlockLinear(nn.Linear):
  """ Custom Linear layer that only uses the stongest K weights """
  def __init__(self, in_features, out_features, bias=False):
    super().__init__(in_features, out_features, bias=bias)
    block_size=floor(in_features/out_features)
    assert(block_size*out_features==in_features)
    self.block_size = block_size

    self.mask=torch.zeros(out_features, in_features, device='cuda:0')
    for counter in range(self.out_features):
      self.mask[counter, torch.arange(counter*self.block_size, (1+counter)*self.block_size)]=1

  def forward(self, x):
    self.net_weight=self.weight*self.mask
    w_times_x= torch.mm(x, self.net_weight.t())
    return w_times_x








#@title Define dendrite branch layer
class DendriteBranchLayer(nn.Module):
  """ """
  def __init__(self, inhibitory_units_n, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch,
               excitatory_synapses_per_dendrite_branch, shunting = False, branch_input = False, input_dendrite_branch_n = None):
    super(DendriteBranchLayer, self).__init__()

    self.branch_input = branch_input
    self.shunting = shunting

    self.inputs_to_dendrite_branch = TopKLinear(input_size, dendrite_branch_n, excitatory_synapses_per_dendrite_branch)

    self.inhibition_to_dendrite_branch = TopKLinear(inhibitory_units_n, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch)

    if self.branch_input:
      self.dendrite_branch_inputs_to_dendrite_branch = BlockLinear(input_dendrite_branch_n, dendrite_branch_n)
      self.dendrite_branch_inputs_to_dendrite_branch.requires_grad_(False) # we don't train this.  All dendrites have the same weight

    self.sigmoid=nn.Sigmoid()

  def forward(self, input, inhibition_outputs, dendrite_branch_outputs=None):

    self.dendrite_inhibition = self.inhibition_to_dendrite_branch(inhibition_outputs)
    self.dendrite_excitation = self.inputs_to_dendrite_branch(input)


    if self.branch_input:
      self.dendrite_depolarization = self.dendrite_branch_inputs_to_dendrite_branch(dendrite_branch_outputs)
      total_excitatory_input = self.dendrite_excitation + self.dendrite_depolarization
    else:
      total_excitatory_input = self.dendrite_excitation

    if self.shunting:
      self.dendrite_act = (total_excitatory_input)/     \
       (total_excitatory_input + e_to_i_ratio*self.dendrite_inhibition + 1)
      self.dendrite_out=self.dendrite_act
    else:
      self.dendrite_act = total_excitatory_input - e_to_i_ratio*self.dendrite_inhibition - 3
      self.dendrite_out = self.sigmoid(self.dendrite_act)

    return self.dendrite_out








#@title Define network structure

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()

    # inhibitory cell layer
    self.inputs_to_inhibition=TopKLinear(input_size, inhibitory_units_n, synapses_per_inhibitory_unit) # only XX inputs to each inhibitory cell

    # dendrite layers
    self.dendrite_layers = nn.ModuleList([None]*dendrite_layers_n)
    for i in range(dendrite_layers_n - 1 , -1, -1):

        is_outermost_layer = (i == (dendrite_layers_n - 1))
        branch_input = False if is_outermost_layer else True
        input_dendrite_branch_n = None if is_outermost_layer else dendrite_branch_n[i+1]

        self.dendrite_layers[i] = DendriteBranchLayer(inhibitory_units_n, dendrite_branch_n[i], inhibitory_synapses_per_dendrite_branch,
               excitatory_synapses_per_dendrite_branch, shunting,
               branch_input, input_dendrite_branch_n)

    # soma
    self.dendrite_to_soma=BlockLinear(dendrite_branch_n[0], soma_n)
    self.dendrite_to_soma.requires_grad_(False) # we don't train this.  All dendrites have the same weight

    self.relu=nn.ReLU()
    self.sigmoid=nn.Sigmoid()

  def initWeights(self):

    with torch.no_grad():

      self.inputs_to_inhibition.weight.data=torch.sign(self.inputs_to_inhibition.weight)*self.inputs_to_inhibition.weight

      for i in range(dendrite_layers_n - 1 , -1, -1):

          self.dendrite_layers[i].inputs_to_dendrite_branch.weight.data=torch.sign(self.dendrite_layers[i].inputs_to_dendrite_branch.weight)*self.dendrite_layers[i].inputs_to_dendrite_branch.weight # all +
          self.dendrite_layers[i].inhibition_to_dendrite_branch.weight.data=torch.sign(self.dendrite_layers[i].inhibition_to_dendrite_branch.weight)*self.dendrite_layers[i].inhibition_to_dendrite_branch.weight # all +

          is_outermost_layer = (i == (dendrite_layers_n - 1))
          if not is_outermost_layer:
              self.dendrite_layers[i].dendrite_branch_inputs_to_dendrite_branch.weight.data=1.0*self.dendrite_layers[i].dendrite_branch_inputs_to_dendrite_branch.mask # all 1

      self.dendrite_to_soma.weight.data=1.0*self.dendrite_to_soma.mask # all 1

  def trimWeights(self):

      with torch.no_grad():
         self.inputs_to_inhibition.weight.data=torch.maximum(self.inputs_to_inhibition.weight, 0*self.inputs_to_inhibition.weight) # all +

         for i in range(dendrite_layers_n - 1 , -1, -1):
             self.dendrite_layers[i].inputs_to_dendrite_branch.weight.data=torch.maximum(self.dendrite_layers[i].inputs_to_dendrite_branch.weight, 0*self.dendrite_layers[i].inputs_to_dendrite_branch.weight) # all +
             self.dendrite_layers[i].inhibition_to_dendrite_branch.weight.data=torch.maximum(self.dendrite_layers[i].inhibition_to_dendrite_branch.weight, 0*self.dendrite_layers[i].inhibition_to_dendrite_branch.weight) # all +


  def forward(self, input):

    self.inhibition_act = self.inputs_to_inhibition(input)
    if shunting:
      self.inhibition_out = self.inhibition_act / (1 + self.inhibition_act)
    else:
      self.inhibition_out = self.sigmoid(self.inhibition_act - sigmoid_offset)

    self.dendrite_branch_out = [None] * dendrite_layers_n

    for i in range(dendrite_layers_n - 1 , -1, -1):

        is_outermost_layer = (i == (dendrite_layers_n - 1))
        this_dendrite_branch_out = None if is_outermost_layer else self.dendrite_branch_out[i + 1]

        self.dendrite_branch_out[i] = self.dendrite_layers[i](input, self.inhibition_out, this_dendrite_branch_out)

    self.soma_act = self.dendrite_to_soma(self.dendrite_branch_out[0])
    # self.soma_out = self.sigmoid(self.soma_act - sigmoid_offset)

    return self.soma_act





#@title Build the model
net = Net()
if torch.cuda.is_available():
  net.cuda()
net.initWeights()





#@title Define loss-function & optimizer
lr=0.01
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
lossi=[]

print(lr)





#@title Training the model

for epoch in range(3*num_epochs):
  for i ,(images,labels) in enumerate(train_gen):
    # images = transform_images(images_raw.view(-1,28*28)).cuda()
    # labels = sum_labels(labels_raw).cuda()
    images = images.view(-1,28*28).cuda()
    labels = labels.cuda()

    optimizer.zero_grad()
    outputs = net(images)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    net.trimWeights()
    lossi.append(loss.item())

    if (i+1) % 100 == 0:
      scheduler.step()
      print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.data))





#@title Evaluating the accuracy of the model

correct = 0
total = 0
for images,labels in test_gen:
  # images = transform_images(images_raw.view(-1,28*28)).cuda()
  # labels = sum_labels(labels_raw).cuda()
  images = images.view(-1,28*28).cuda()
  labels = labels.cuda()
  with torch.no_grad():
    output = net(images)
    _, predicted = torch.max(output,1)
    correct += (predicted == labels).sum()
    total += labels.size(0)

print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))





# Proportion of labels in training data
processed_labels = np.empty((1,))
for images_raw,labels_raw in train_gen:
  labels = sum_labels(labels_raw).cuda()
  processed_labels = np.concatenate((processed_labels, labels.cpu().numpy()))

processed_labels = processed_labels[1:]





#@title dendrite weights
plt.rcParams["figure.figsize"] = (20,3)

print("inputs_to_inhibition")
plt.xlabel("pixel")
plt.ylabel("inh unit")
plotWithColor(net.inputs_to_inhibition.weight.data)

print("inputs_to_inhibition mask")
plt.xlabel("pixel")
plt.ylabel("inh unit")
plotWithColor(net.inputs_to_inhibition.mask.data)

# plt.rcParams["figure.figsize"] = (3,20)

print("inhibition_to_dendrite_branch")
plt.ylabel("inh unit")
plt.xlabel("dendrite")
plotWithColor(net.inhibition_to_dendrite_branch.weight.data.t())

print("inhibition_to_dendrite_branch net")
plt.ylabel("inh unit net")
plt.xlabel("dendrite")
plotWithColor(net.inhibition_to_dendrite_branch.net_weight.data.t())

print("inhibition_to_dendrite_branch mask")
plt.ylabel("inh unit mask")
plt.xlabel("dendrite")
plotWithColor(net.inhibition_to_dendrite_branch.mask.data.t())

print("inhibition_to_dendrite")
plt.ylabel("inh unit")
plt.xlabel("dendrite")
plotWithColor(net.inhibition_to_dendrite.weight.data.t())

print("inhibition_to_dendrite net")
plt.ylabel("inh unit net")
plt.xlabel("dendrite")
plotWithColor(net.inhibition_to_dendrite.net_weight.data.t())

print("inhibition_to_dendrite mask")
plt.ylabel("inh unit mask")
plt.xlabel("dendrite")
plotWithColor(net.inhibition_to_dendrite.mask.data.t())





# #@title inhibition weights
plt.rcParams["figure.figsize"] = (20,6)

print("inputs_to_dendrite_branch")
plt.xlabel("pixel")
plt.ylabel("branch")
plotWithColor(net.inputs_to_dendrite_branch.weight.data)

# print("inputs_to_dendrite_branch net")
# plt.xlabel("pixel")
# plt.ylabel("branch net")
# plotWithColor(net.inputs_to_dendrite_branch.net_weight.data)

print("inputs_to_dendrite_branch mask")
plt.xlabel("pixel")
plt.ylabel("branch")
plotWithColor(net.inputs_to_dendrite_branch.mask.data)

print("inputs_to_dendrite")
plt.xlabel("pixel")
plt.ylabel("dendrite")
plotWithColor(net.inputs_to_dendrite.weight.data)

print("inputs_to_dendrite mask")
plt.xlabel("pixel")
plt.ylabel("dendrite")
plotWithColor(net.inputs_to_dendrite.mask.data)

print("dendrite_branch_to_dendrite")
plt.xlabel("dendrite branch")
plt.ylabel("dendrite")
plotWithColor(net.dendrite_branch_to_dendrite.weight.data)

print("dendrite_to_soma")
plt.xlabel("dendrite")
plt.ylabel("soma")
plotWithColor(net.dendrite_to_soma.weight.data)









# plotWithColor(net.dendrites_to_soma.mask.data.cpu())







labels_sort, labels_sort_ind=torch.sort(labels)

images_sort=images[labels_sort_ind]
plt.plot(labels_sort.cpu().numpy())

images_solo=images_raw.view(-1,28*28).cuda()
labels_solo_sort, labels_solo_sort_ind=torch.sort(labels_raw)
images_solo_sort=images_solo[labels_solo_sort_ind]
plt.plot(labels_solo_sort.cpu().numpy())







with torch.no_grad():
  output=net(transform_images(images_solo_sort))






plt.rcParams["figure.figsize"] = (20,3)
plotWithColor(net.inhibition_act.t(), 'inhibition_act')
plotWithColor(net.inhibition_out.t(), 'inhibition_out')






plotWithColor(net.dendrite_branch_inhibition, 'dendrite_branch_inhibition')
plotWithColor(net.dendrite_branch_excitation, 'dendrite_branch_excitation')
plotWithColor(net.dendrite_branch_act, 'dendrite_branch_act')
plotWithColor(net.dendrite_branch_out, 'dendrite_branch_out')






plotWithColor(net.dendrite_inhibition.t(), 'dendrite_inhibition')
plotWithColor(net.dendrite_excitation.t(), 'dendrite_excitation')
plotWithColor(net.dendrite_depolarization.t(), 'dendrite_depolarization')
plotWithColor(net.dendrite_act.t(), 'dendrite_act')
plotWithColor(net.dendrite_out.t(), 'dendrite_out')






plt.rcParams["figure.figsize"] = (10,3)
plotWithColor(net.soma_act.t(), 'soma_act')
# plotWithColor(net.soma_out.t())






#@title The shape of the total excitatory input to each main dendrite of each soma


#fig.suptitle('The shape of the total excitatory each main dendrite of each soma', fontsize=16)
for soma_counter in range(soma_n):
  fig, axs = plt.subplots(1, dendrite_per_soma)
  fig.set_figwidth(20)

  for dendrite_counter in range(dendrite_per_soma):
    imAvg=0*net.inputs_to_dendrite_branch.net_weight.data[0,:].reshape(56,28);
    for counter in range(dendrite_branch_per_dendrite):
      start_dendrite_branch=soma_counter*dendrite_branch_per_soma +  dendrite_counter*dendrite_branch_per_dendrite
      imAvg+=net.inputs_to_dendrite_branch.net_weight.data[start_dendrite_branch+counter,:].reshape(56,28)
    axs[dendrite_counter].imshow(imAvg.cpu().numpy())
    axs[dendrite_counter].set_xlabel(dendrite_counter)
  axs[0].set_ylabel(soma_counter)







#@title The shape of the total excitatory input to all the dendrites of each soma
fig, axs = plt.subplots(1, 10)
fig.set_figwidth(20)
fig.suptitle('The shape of the total excitatory input to each soma', fontsize=16)

for soma_counter in range(10):
  imAvg=0*net.inputs_to_dendrite_branch.net_weight.data[0,:].reshape(56,28);
  for counter in range(soma_counter*dendrite_branch_per_soma, (soma_counter+1)*dendrite_branch_per_soma):
    imAvg+=net.inputs_to_dendrite_branch.net_weight.data[counter,:].reshape(56,28)
  axs[soma_counter].imshow(imAvg.cpu().numpy())






#@title The shape of the total inhibitory input directly to the main dendrites of each soma
inhIndices=net.inhibition_to_dendrite.topk_indices.flatten()
fig, axs = plt.subplots(1, soma_n)
fig.set_figwidth(20)
for soma_index in range(soma_n):
  imAvg=0*net.inputs_to_inhibition.weight.data[0,:].reshape(28,56);
  for counter in range(soma_index*dendrite_per_soma, (soma_index+1)*dendrite_per_soma):
    imAvg+=net.inputs_to_inhibition.net_weight.data[inhIndices[counter],:].reshape(56,28)
  fig.suptitle('The shape of the total inhibitory input directly onto dendrites of each soma', fontsize=16)
  axs[soma_index].imshow(imAvg.cpu().numpy())

inhIndices=net.inhibition_to_dendrite_branch.topk_indices.flatten()
fig, axs = plt.subplots(1, soma_n)
fig.set_figwidth(20)
for soma_index in range(soma_n):
  imAvg=0*net.inputs_to_inhibition.weight.data[0,:].reshape(56,28);
  for counter in range(soma_index*dendrite_branch_per_soma, (soma_index+1)*dendrite_branch_per_soma):
    imAvg+=net.inputs_to_inhibition.net_weight.data[inhIndices[counter],:].reshape(56,28)
  fig.suptitle('The shape of the total inhibitory input to all dendrite branches of each soma', fontsize=16)
  axs[soma_index].imshow(imAvg.cpu().numpy())






#@title The shape of the input to each inhibitory cell
plt.rcParams["figure.figsize"] = (20,9)
from math import ceil

inhIndices=net.inhibition_to_dendrite_branch.topk_indices.flatten()
fig, axs = plt.subplots(ceil(inhibitory_units_n/10), 10)
fig.set_figwidth(20)
row_counter=0
for inh_index in range(inhibitory_units_n):
  fig.suptitle('The shape of the input to each inhibitory cell', fontsize=16)
  axs[row_counter, inh_index % 10].imshow(net.inputs_to_inhibition.net_weight.data[inh_index,:].reshape(56,28).cpu().numpy())
  if torch.where(inhIndices==inh_index)[0].numel()==0:
    axs[row_counter, inh_index % 10].set_title('not used')
  else:
    axs[row_counter, inh_index % 10].set_title('used')

  if inh_index % 10==9:
    row_counter+=1






#@title The shape of the excitatory and inhibitory input to each dendrite of each soma
inhIndices=net.inhibition_to_dendrite_branch.topk_indices.flatten()

plt.rcParams["figure.figsize"] = (20,3.5)
for soma_index in range(soma_n):
  fig, axs = plt.subplots(2, dendrite_branch_per_soma)
  fig.set_figwidth(20)
  offset=soma_index*dendrite_branch_per_soma
  for counter in range(dendrite_branch_per_soma):
    fig.suptitle(soma_index, fontsize=16)
    axs[0, counter].imshow(net.inputs_to_dendrite_branch.net_weight.data[offset+counter,:].reshape(56,28).cpu().numpy())
    axs[1, counter].imshow(net.inputs_to_inhibition.weight.data[inhIndices[offset+counter],:].reshape(56,28).cpu().numpy())
    axs[1, counter].set_title(inhIndices[offset+counter].item())
  axs[0, 0].set_ylabel('exc')
  axs[1, 0].set_ylabel('inh')











































