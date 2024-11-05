import torch
import os
import numpy
from matplotlib import pyplot
import numpy as np

cuda = torch.cuda.is_available()
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')

def get_init_state(batch_size, cell_size, squeeze=False):
    h_init = torch.zeros((1, batch_size, cell_size)).to(device)
    c_init = torch.zeros((1, batch_size, cell_size)).to(device)
    
    if squeeze:
        h_init, c_init = h_init.squeeze(0), c_init.squeeze(0)
        
    return h_init, c_init


def save_checkpoint(epoch, model, validation_loss, optimizer, scheduler, directory, \
                    filename='best.pt'):
    checkpoint=({'epoch': epoch+1,
    'model': model.state_dict(),
    'validation_loss': validation_loss,
    'optimizer' : optimizer.state_dict(),
    'scheduler' : None if scheduler is None else scheduler.state_dict()
    })
    try:
        torch.save(checkpoint, os.path.join(directory, filename))
        
    except:
        os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory, filename))


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    #stroke[:, :3] = (stroke[:, :3] * np.array([1., 42.903453, 37.58289517])) + np.array([0., 8.37023456, 0.1114528])
    #stroke[:, :3] = (stroke[:, :3] * np.array([1., 11.86284, 6.452663])) + np.array([0., 1.4611896, 0.03389655])
    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])
    #print(list(x))
    #print(list(y))
    #x = stroke[:, 1]
    #y = stroke[:, 2]

    #size_x = x.max() - x.min() + 1.
    #size_y = y.max() - y.min() + 1.

    #f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0
    for cut_value in cuts:
        color = 'black'
        if stroke.shape[1] == 4:
            color = 'red' if (stroke[start:cut_value, 3] == 1).any() else 'black'
            
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3, color=color)
        start = cut_value + 1
    ax.axis('equal')
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: " + save_name)

    pyplot.close()
