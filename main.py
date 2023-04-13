from model import *
import matplotlib.pyplot as plt

def load_data(input_type = MODEL_INPUT.FIVE_DAYS, output_type = MODEL_OUTPUT.ONE_DAY, batch_size = 32):
    # randomly generate data of images in gray scale
    # if input_type == MODEL_INPUT.FIVE_DAYS:
    # the input is a picture of 15*32 pixels
    # the output is a float number
    # if input_type == MODEL_INPUT.TWENTY_DAYS:
    # the input is a picture of 60*64 pixels
    # the output is a float number
    # the size of dataset is 1000
    if input_type == MODEL_INPUT.FIVE_DAYS:
        train_dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 32, 15), torch.rand(1000, 1))
        val_dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 32, 15), torch.rand(1000, 1))
        test_dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 32, 15), torch.rand(1000, 1))
    elif input_type == MODEL_INPUT.TWENTY_DAYS:
        train_dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 64, 60), torch.rand(1000, 1))
        val_dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 64, 60), torch.rand(1000, 1))
        test_dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 64, 60), torch.rand(1000, 1))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
    


def twenty_days():
    # load data
    train_loader, val_loader, test_loader = load_data(MODEL_INPUT.TWENTY_DAYS, MODEL_OUTPUT.ONE_DAY)
    # show example data
    for i, (inputs, labels) in enumerate(train_loader):
        # plot 4 images as gray scale
        plt.subplot(221)
        plt.imshow(inputs[0][0], cmap=plt.get_cmap('gray'))
        plt.subplot(222)
        plt.imshow(inputs[1][0], cmap=plt.get_cmap('gray'))
        plt.subplot(223)
        plt.imshow(inputs[2][0], cmap=plt.get_cmap('gray'))
        plt.subplot(224)
        plt.imshow(inputs[3][0], cmap=plt.get_cmap('gray'))
        # show the plot
        plt.show()
        if i == 2:
            break


    # train model
    model = CNN20d()
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, batch_size=32, device='cpu', weight_decay=0.0)
    test_model(model, test_loader, device='cpu')

def five_days():
    # load data
    train_loader, val_loader, test_loader = load_data(MODEL_INPUT.FIVE_DAYS, MODEL_OUTPUT.ONE_DAY)
    # show example data
    for i, (inputs, labels) in enumerate(train_loader):
        # plot 4 images as gray scale
        plt.subplot(221)
        plt.imshow(inputs[0][0], cmap=plt.get_cmap('gray'))
        plt.subplot(222)
        plt.imshow(inputs[1][0], cmap=plt.get_cmap('gray'))
        plt.subplot(223)
        plt.imshow(inputs[2][0], cmap=plt.get_cmap('gray'))
        plt.subplot(224)
        plt.imshow(inputs[3][0], cmap=plt.get_cmap('gray'))
        # show the plot
        plt.show()
        break


    # train model
    model = CNN5d()
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, batch_size=32, device='cpu', weight_decay=0.0)
    test_model(model, test_loader, device='cpu')


if __name__ == '__main__':
    five_days()
    twenty_days()