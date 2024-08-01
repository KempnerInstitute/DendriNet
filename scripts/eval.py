import torch




if __name__ == '__main__':

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








