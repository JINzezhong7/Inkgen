import torch
import os
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np

# cuda = torch.cuda.is_available()
# device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

def get_init_state(batch_size, cell_size, device, squeeze=False):
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
    model_path = os.path.join(directory, filename)
    try:
        torch.save(checkpoint, model_path)
        
    except:
        os.mkdir(directory)
        torch.save(checkpoint, model_path)
    return model_path

def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()
    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    if len(stroke) not in cuts:
        cuts = np.append(cuts, len(stroke))
    start = 0
    for cut_value in cuts:
        color = 'black'
        if stroke.shape[1] == 4:
            color = 'red' if (stroke[start:cut_value, 3] == 1).any() else 'black'
            
        ax.plot(x[start:cut_value], y[start:cut_value],linewidth=3, color=color)
        start = cut_value + 1
    ax.axis('equal')

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

    pyplot.close('all')
