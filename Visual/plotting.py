import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable



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
    imAvg=0*net.inputs_to_dendrite_branch.net_weight.data[0,:].reshape(56,28)
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
    imAvg=0*net.inputs_to_inhibition.weight.data[0,:].reshape(28,56)
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









