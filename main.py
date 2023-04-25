from __init__ import *
import utils as _U
reload(_U)
import model as _M
reload(_M)
import train as _T
reload(_T)
import dataset as _D
reload(_D)
import sys


parser = argparse.ArgumentParser(description='Train Models via YAML files')
parser.add_argument('setting', type=str, \
                    help='Experiment Settings, should be yaml files like those in /configs')

args = parser.parse_args()

with open(args.setting, 'r') as f:
    setting = _U.Dict2ObjParser(yaml.safe_load(f)).parse()

if 'models' not in os.listdir('./'):
    os.system('mkdir models')
if setting.TRAIN.MODEL_SAVE_FILE.split('/')[1] not in os.listdir('./models/'):
    os.system(f"cd models && mkdir {setting.TRAIN.MODEL_SAVE_FILE.split('/')[1]}")
if 'logs' not in os.listdir('./'):
    os.system('mkdir logs')
if setting.TRAIN.LOG_SAVE_FILE.split('/')[1] not in os.listdir('./logs/'):
    os.system(f"cd logs && mkdir {setting.TRAIN.LOG_SAVE_FILE.split('/')[1]}")

dir = setting.TRAIN.MODEL_SAVE_FILE.split('/')[0] + '/' + setting.TRAIN.MODEL_SAVE_FILE.split('/')[1]
if setting.TRAIN.MODEL_SAVE_FILE.split('/')[2] in os.listdir(dir):
    print(f'Pretrained Model: {args.setting} Already Exist')
    sys.exit(0)

dataset = _D.ImageDataSet(win_size = setting.DATASET.LOOKBACK_WIN, \
                            start_date = setting.DATASET.START_DATE, \
                            end_date = setting.DATASET.END_DATE, \
                            mode = 'train', \
                            label = setting.TRAIN.LABEL, \
                            indicators = setting.DATASET.INDICATORS, \
                            show_volume = setting.DATASET.SHOW_VOLUME, \
                            parallel_num=setting.DATASET.PARALLEL_NUM)

image_set = dataset.generate_images(setting.DATASET.SAMPLE_RATE)

train_loader_size = int(len(image_set)*(1-setting.TRAIN.VALID_RATIO))
valid_loader_size = len(image_set) - train_loader_size

train_loader, valid_loader = torch.utils.data.random_split(image_set, [train_loader_size, valid_loader_size])
train_loader = torch.utils.data.DataLoader(dataset=train_loader, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_loader, batch_size=setting.TRAIN.BATCH_SIZE, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert setting.MODEL in ['CNN5d', 'CNN20d'], f"Wrong Model Template: {setting.MODEL}"


if __name__ == '__main__':
    
    if setting.MODEL == 'CNN5d':
        model = _M.CNN5d()
    else:
        model = _M.CNN20d()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=setting.TRAIN.LEARNING_RATE, weight_decay=setting.TRAIN.WEIGHT_DECAY)
    
    train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = _T.train_n_epochs(setting.TRAIN.NEPOCH, model, setting.TRAIN.LABEL, train_loader, valid_loader, criterion, optimizer, setting.TRAIN.MODEL_SAVE_FILE, setting.TRAIN.EARLY_STOP_EPOCH)
    
    log = pd.DataFrame([train_loss_set, train_acc_set, valid_loss_set, valid_acc_set], index=['train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
    log.to_csv(setting.TRAIN.LOG_SAVE_FILE)
    
    