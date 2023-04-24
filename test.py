from __init__ import *
import model as _M
reload(_M)
import train as _T
reload(_T)
import dataset as _D
reload(_D)
import utils as _U
reload(_U)


def model_test(model, label_type, classes, criterion, setting):
    # track test loss
    test_loss = 0.0
    class_correct = [0., 0.]
    class_total = [0., 0.]

    model.eval()
    # iterate over test data
    sub_points = [setting.TEST.START_DATE] + [int(setting.TEST.END_DATE//1e4 * 1e4) + i*100 + 1 for i in range(4, 13, 3)] + [setting.TEST.END_DATE]
    
    for m_idx in range(len(sub_points)-1):
        print(f"Testing: {sub_points[m_idx]} - {sub_points[m_idx+1]}")
        test_dataset = _D.ImageDataSet(win_size = setting.DATASET.LOOKBACK_WIN, \
                            start_date = sub_points[m_idx], \
                            end_date = sub_points[m_idx+1], \
                            mode = 'default', \
                            label = setting.TRAIN.LABEL, \
                            indicators = setting.DATASET.INDICATORS, \
                            show_volume = setting.DATASET.SHOW_VOLUME, \
                            parallel_num=setting.DATASET.PARALLEL_NUM)
        test_imageset = test_dataset.generate_images(1.0)
        test_loader = torch.utils.data.DataLoader(dataset=test_imageset, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=False)
            
        for i, (data, ret5, ret20) in enumerate(test_loader):
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
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            pred = torch.argmax(output, 1)    
            # compare predictions to true label
            correct_tensor = pred.eq(torch.argmax(target, 1).data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not device == 'cuda' else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(target.shape[0]):
                label = torch.argmax(target.data[i])
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models via YAML files')
    parser.add_argument('setting', type=str, \
                        help='Experiment Settings, should be yaml files like those in /configs')

    args = parser.parse_args()

    with open(args.setting, 'r') as f:
        setting = _U.Dict2ObjParser(yaml.safe_load(f)).parse()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert setting.MODEL in ['CNN5d', 'CNN20d'], f"Wrong Model Template: {setting.MODEL}"
    
    
    if setting.MODEL == 'CNN5d':
        model = _M.CNN5d()
    else:
        model = _M.CNN20d()
    model.to(device)

    state_dict = torch.load(setting.TRAIN.MODEL_SAVE_FILE)
    model.load_state_dict(state_dict['model_state_dict'])

    criterion = nn.BCELoss()

    model_test(model, setting.TRAIN.LABEL, ['down', 'up'], criterion, setting)