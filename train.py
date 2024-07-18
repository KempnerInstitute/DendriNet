import torch








if __name__ == '__main__':

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
                    % (epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.data))
                



    # Proportion of labels in training data
    processed_labels = np.empty((1,))
    for images_raw,labels_raw in train_gen:
        labels = sum_labels(labels_raw).cuda()
        processed_labels = np.concatenate((processed_labels, labels.cpu().numpy()))

    processed_labels = processed_labels[1:]









