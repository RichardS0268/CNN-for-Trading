import torch
from model import *

def load_data_of_years(year_begin, year_end, input_type, output_type):
    """This is a temporary function to generate random data for testing purpose."""
    dataset = None
    if input_type == MODEL_INPUT.FIVE_DAYS:
        dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 32, 15), torch.rand(1000, 1))
    elif input_type == MODEL_INPUT.TWENTY_DAYS:
        dataset = torch.utils.data.TensorDataset(torch.rand(1000, 1, 64, 60), torch.rand(1000, 1))
    return dataset

# roll = [((2010, 2012), (2012, 2013)),
#         ((2011, 2013), (2013, 2014)),
#         ((2012, 2014), (2014, 2015))]

def generate_roll(year_begin, year_end, step):
    roll = []
    for i in range(year_begin, year_end, step):
        roll.append(((i, i+step), (i+step, i+step+1)))
    return roll

def load_data_for_roll_item(roll_item, input_type, output_type, batch_size, val):
    train_dataset = load_data_of_years(roll_item[0][0], roll_item[0][1], input_type, output_type)
    train_loader, val_loader = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * (1 - val)), int(len(train_dataset) * val)])

    test_dataset = load_data_of_years(roll_item[1][0], roll_item[1][1], input_type, output_type)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_in_roll(roll, input_type, output_type, batch_size=64, device='cpu', num_epochs=10, learning_rate=0.001, weight_decay=0.0, val_size=0.2):
    for i in roll:
        train_loader, val_loader, test_loader = load_data_for_roll_item(i, input_type, output_type, batch_size, val_size)
        model = CNN20d().to(device) if input_type == MODEL_INPUT.TWENTY_DAYS else CNN5d().to(device)
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, device=device , weight_decay=weight_decay)
        test_model(model, test_loader, device=device)

if __name__ == '__main__':
    roll = generate_roll(2010, 2021, 2)
    train_in_roll(roll, MODEL_INPUT.FIVE_DAYS, MODEL_OUTPUT.ONE_DAY, 32)
    train_in_roll(roll, MODEL_INPUT.FIVE_DAYS, MODEL_OUTPUT.FIVE_DAYS, 32)
    train_in_roll(roll, MODEL_INPUT.TWENTY_DAYS, MODEL_OUTPUT.ONE_DAY, 32)
    train_in_roll(roll, MODEL_INPUT.TWENTY_DAYS, MODEL_OUTPUT.FIVE_DAYS, 32)
