import argparse
import os
import numpy as np
from PyInk import *
from h5InkSuit import *
from scipy.signal import savgol_filter
import torch
from random import shuffle
from tqdm import tqdm

def get_stroke_sets(data_root):
    stroke_sets = []

    for setname in ['train', 'test']:
        for suit_path in tqdm(os.listdir(os.path.join(data_root, setname))):
            suit = h5InkSuit(os.path.join(data_root, setname, suit_path))
            for panel in suit.panels:
                stroke_set = []
                for stroke in panel.GetAllStrokes():
                    stroke_set.append(stroke.T)
            
                stroke_sets.append((stroke_set, panel.GetLabel(), setname == 'train'))
        
    return stroke_sets

def align(coords):
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords

def denoise(coords):
    coords = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []
    for stroke in coords:
        if len(stroke) != 0:
            x_new = savgol_filter(stroke[:, 0], 7, 3, mode='nearest')
            y_new = savgol_filter(stroke[:, 1], 7, 3, mode='nearest')
            xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])
            stroke = np.concatenate([xy_coords, stroke[:, 2].reshape(-1, 1)], axis=1)
            new_coords.append(stroke)

    coords = np.vstack(new_coords)
    return coords

def normalize(offsets):
    offsets = np.copy(offsets)
    offsets[:, :2] /= np.median(np.linalg.norm(offsets[:, :2], axis=1))
    return offsets

def preprocess_data(stroke_sets, max_len):
    train_set, val_set = [], []
    train_lens, val_lens = [], []
    train_labels, val_labels = [], []
    chars = set()
    for stroke_set, label, is_train in tqdm(stroke_sets):
        stroke_arr = []
        for stroke in stroke_set:
            stroke_arr.extend([x, -y, 1 if i == len(stroke) - 1 else 0] for i, (x, y) in enumerate(stroke))
        stroke_arr = np.array(stroke_arr).astype(np.float32)
        
        stroke_arr = align(stroke_arr)
        stroke_arr = denoise(stroke_arr)

        stroke_arr[1:, :2] -= stroke_arr[:-1, :2]
        stroke_arr[0, :2] = [0, 0]
        stroke_arr = normalize(stroke_arr)
    
        stroke_arr = stroke_arr[:max_len]
        stroke_arr[-1, 2] = 1
        if not ~np.any(np.linalg.norm(stroke_arr[:, :2], axis=1) > 60):
            continue
        
        padded_stroke_arr = np.zeros((max_len, 3)).astype(np.float32)
        padded_stroke_arr[:len(stroke_arr)] = stroke_arr

        if is_train:
            train_set.append(padded_stroke_arr)
            train_lens.append(len(stroke_arr))
            train_labels.append(label)
            chars = chars.union(set(label))
        else:
            val_set.append(padded_stroke_arr)
            val_lens.append(len(stroke_arr))
            val_labels.append(label)

    train_set, val_set = np.array(train_set), np.array(val_set)
    train_lens, val_lens = np.array(train_lens), np.array(val_lens)
    train_labels, val_labels = np.array(train_labels), np.array(val_labels)

    chars = sorted(list(chars))
    charset = {c:i for i,c in enumerate(chars)}

    return train_set, val_set, train_lens, val_lens, train_labels, val_labels, charset

def save(args, data_set, all_lens, labels, idxs, charset, setname):
    x_tmp = data_set[idxs]
    lens = all_lens[idxs]
    
    strokes = np.zeros_like(x_tmp)
    strokes[:, :, 0], strokes[:, :, 1:] = x_tmp[:, :, -1], x_tmp[:, :, :-1]

    masks = np.zeros((len(idxs), max(all_lens)-1))
    for i in range(len(masks)): masks[i, :lens[i]-1] = 1

    texts = labels[idxs]
    onehots = np.zeros((len(idxs), max([len(t) for t in texts]), len(charset)))
    for i, text in enumerate(texts):
        label = np.array([charset[c] for c in text])
        onehots[i, np.arange(len(label)), label] = 1

    strokes = strokes[:, :max(all_lens)]
    assert (np.array([strokes[i][:masks[i].sum().astype(int)+1][-1, 0] for i in range(len(idxs))]) == 1).all()
    
    print(strokes.shape, masks.shape, onehots.shape)
    np.save(args.out_dir + '\\%s_strokes_700.npy' % setname, strokes)
    np.save(args.out_dir + '\\%s_masks_700.npy' % setname, masks)
    np.save(args.out_dir + '\\%s_onehot_700.npy' % setname, onehots)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    print("loading ink samples")
    stroke_sets = get_stroke_sets(args.data_root)
    print("processing ink samples")
    train_set, val_set, train_lens, val_lens, train_labels, val_labels, charset = preprocess_data(stroke_sets, args.max_len)
    
    train_len = int(train_set.shape[0] * .95)
    idxs = np.arange(train_set.shape[0])
    shuffle(idxs)
    train_idxs, valid_idxs = sorted(idxs[:train_len]), sorted(idxs[train_len:])
    print('saving results')
    save(args, train_set, train_lens, train_labels, train_idxs, charset, 'train')
    save(args, train_set, train_lens, train_labels, valid_idxs, charset, 'validation')
    torch.save(charset, args.out_dir + '\\char_to_code.pt')

    
if __name__ == '__main__':
    main()
