from importlib import reload
from model import *
from train import *
from dataloader import *
import dataloader as _D
reload(_D)
import utils as _U
reload(_U)
from collections import OrderedDict
import yaml
import argparse

parser = argparse.ArgumentParser(description='Train Models via YAML files')
parser.add_argument('setting', type=str, \
                    help='Experiment Settings, should be yaml files like those in /configs')

args = parser.parse_args()

with open(args.setting, 'r') as f:
    setting = _U.Dict2ObjParser(yaml.safe_load(f)).parse()


dataset = _D.ImageDataSet(win_size = setting.DATASET.LOOKBACK_WIN, \
                            start_date = setting.DATASET.START_DATE, \
                            end_date = setting.DATASET.END_DATE, \
                            mode = setting.DATASET.MODE, \
                            indicators = setting.DATASET.INDICATORS, \
                            show_volume = setting.DATASET.SHOW_VOLUME, \
                            parallel_num=setting.DATASET.PARALLEL_NUM)

image_set = dataset.generate_images(setting.DATASET.SAMPLE_RATE)

train_loader_size = int(len(image_set)*(1-setting.TRAIN.VALID_RATIO))
valid_loader_size = len(image_set) - train_loader_size

train_loader, valid_loader = torch.utils.data.random_split(image_set, [train_loader_size, valid_loader_size])
train_loader = torch.utils.data.DataLoader(dataset=train_loader, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_loader, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert setting.MODEL in ['CNN5d', 'CNN20d'], f"Wrong Model Template: {setting.MODEL}"

if 'models' not in os.listdir('./'):
    os.system('mkdir models')
if setting.TRAIN.MODEL_SAVE_FILE.split('/')[1] not in os.listdir('./models/'):
    os.system(f"cd models && mkdir {setting.TRAIN.MODEL_SAVE_FILE.split('/')[1]}")
if 'logs' not in os.listdir('./'):
    os.system('mkdir logs')
if setting.TRAIN.LOG_SAVE_FILE.split('/')[1] not in os.listdir('./logs/'):
    os.system(f"cd logs && mkdir {setting.TRAIN.LOG_SAVE_FILE.split('/')[1]}")


if __name__ == '__main__':
    if setting.MODEL == 'CNN5d':
        model = CNN5d()
    else:
        model = CNN20d()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=setting.TRAIN.LEARNING_RATE, weight_decay=setting.TRAIN.WEIGHT_DECAY)
    
    train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = train_n_epochs(setting.TRAIN.NEPOCH, model, setting.TRAIN.LABEL, train_loader, valid_loader, criterion, optimizer, setting.TRAIN.MODEL_SAVE_FILE)
    
    log = pd.DataFrame([train_loss_set, train_acc_set, valid_loss_set, valid_acc_set], index=['train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
    log.to_csv(setting.TRAIN.LOG_SAVE_FILE)
    
    