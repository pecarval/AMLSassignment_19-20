import matplotlib.pyplot as plt


def plot_acc_curve(accs, num_epochs, title):
    '''
    Plots 2 curves of accuracies in training and validation phases

    Keyword arguments:
        - accs : Array with accuracy at each epoch for training and validation
        - num_epochs : Number of epochs used in training
        - title : String with title to use on the plot and save the figure
    '''

    plt.figure(1, figsize=(6, 6))
    plt.plot(range(1,num_epochs+1), accs['train']*100, '-ro', label='Training Accuracy')
    plt.plot(range(1,num_epochs+1), accs['val']*100, '-bo', label='Validation Accuracy')
    plt.ylabel('Accuracy %')
    plt.xlabel('Epoch number')
    plt.legend()
    plt.title(title)
    plt.savefig(title + '.png')
    plt.clf()
    plt.close()


def plot_loss_curve(losses, num_epochs, title):
    '''
    Plots 2 curves of losses in training and validation phases

    Keyword arguments:
        - losses : Array with loss at each epoch for training and validation
        - num_epochs : Number of epochs used in training
        - title : String with title to use on the plot and save the figure
    '''

    plt.figure(1, figsize=(6, 6))
    plt.plot(range(1,num_epochs+1), losses['train'], '-ro', label='Training Loss')
    plt.plot(range(1,num_epochs+1), losses['val'], '-bo', label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch number')
    plt.legend()
    plt.title(title)
    plt.savefig(title + '.png')
    plt.clf()
    plt.close()