from __init__ import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_n_epochs(n_epochs, model, label_type, train_loader, valid_loader, criterion, optimizer, savefile, early_stop_epoch):
    valid_loss_min = np.Inf # track change in validation loss
    train_loss_set = []
    valid_loss_set = []
    train_acc_set = []
    valid_acc_set = []
    invariant_epochs = 0
    
    for epoch_i in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0
        
        #### Model for training 
        model.train()
        for i, (data, ret5, ret20) in enumerate(train_loader):
            assert label_type in ['RET5', 'RET20'], f"Wrong Label Type: {label_type}"
            if label_type == 'RET5':
                target = ret5
            else:
                target = ret20

            target = (1-target).unsqueeze(1) @ torch.LongTensor([1., 0.]).unsqueeze(1).T + target.unsqueeze(1) @ torch.LongTensor([0, 1]).unsqueeze(1).T
            target = target.to(torch.float32)

            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            # update training acc
            train_acc += (output.argmax(1) == target.argmax(1)).sum()


        #### Model for validation
        model.eval()
        for i, (data, ret5, ret20) in enumerate(valid_loader):
            assert label_type in ['RET5', 'RET20'], f"Wrong Label Type: {label_type}"
            if label_type == 'RET5':
                target = ret5
            else:
                target = ret20
                
            target = (1-target).unsqueeze(1) @ torch.LongTensor([1., 0.]).unsqueeze(1).T + target.unsqueeze(1) @ torch.LongTensor([0, 1]).unsqueeze(1).T
            target = target.to(torch.float32)
                
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
            valid_acc += (output.argmax(1) == target.argmax(1)).sum()
        
        # Compute average loss
        train_loss = train_loss/len(train_loader.sampler)
        train_loss_set.append(train_loss)
        valid_loss = valid_loss/len(valid_loader.sampler)
        valid_loss_set.append(valid_loss)

        train_acc = train_acc/len(train_loader.sampler)
        train_acc_set.append(train_acc.cpu().numpy())
        valid_acc = valid_acc/len(valid_loader.sampler)
        valid_acc_set.append(valid_acc.cpu().numpy())
            
        print('Epoch: {} Training Loss: {:.6f} Validation Loss: {:.6f} Training Acc: {:.5f} Validation Acc: {:.5f}'.format(epoch_i, train_loss, valid_loss, train_acc, valid_acc))
        
        # if validation loss gets smaller, save the model
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            valid_loss_min = valid_loss
            invariant_epochs = 0
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, savefile)
        else:
            invariant_epochs = invariant_epochs + 1
        
        if invariant_epochs == early_stop_epoch:
            print(f"Early Stop at Epoch [{epoch_i}]: Performance hasn't enhanced for {early_stop_epoch} epochs")
            break

    return train_loss_set, valid_loss_set, train_acc_set, valid_acc_set



def plot_loss_and_acc(loss_and_acc_dict):
    _, axes = plt.subplots(1, 2, figsize=(20, 6))
    tmp = list(loss_and_acc_dict.values())
    maxEpoch = len(tmp[0][0])

    maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
    minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

    for name, lossAndAcc in loss_and_acc_dict.items():
        axes[0].plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(range(0, maxEpoch + 1, maxEpoch//10))
    axes[0].axis([0, maxEpoch, minLoss, maxLoss])
    axes[0].legend()
    axes[0].set_title("Error")

    maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
    minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

    for name, lossAndAcc in loss_and_acc_dict.items():
        axes[1].plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xticks(range(0, maxEpoch + 1, maxEpoch//10))
    axes[1].axis([0, maxEpoch, minAcc, maxAcc])
    axes[1].legend()
    axes[1].set_title("Accuray")