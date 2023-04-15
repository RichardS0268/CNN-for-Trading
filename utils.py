
from contextlib import contextmanager
import time
import matplotlib.pyplot as plt


@contextmanager 
def timer(name: str, _align): # ⏱
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f"{ '[' + name + ']' :{_align}} | {time.strftime('%Y-%m-%d %H:%M:%S')} Done | Using {elapsed: .3f} seconds")
    
    

def display_image(entry):
    assert (type(entry) == list) and (len(entry) == 4), "Type error, expected a list with length of 4"
    plt.imshow(entry[0], cmap=plt.get_cmap('gray'))
    plt.ylim((0,entry[0].shape[0]-1))
    plt.xlim((0,entry[0].shape[1]-1))
    plt.title(f'ret1: {entry[1]}\nret5: {entry[2]}\nret20: {entry[2]}')
    


