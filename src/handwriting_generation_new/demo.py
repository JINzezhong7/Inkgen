from generate import generate_conditionally

import PyInk
import numpy as np
from scipy.signal import savgol_filter
import torch

import tkinter
import tkinter.messagebox


def main():
    def clear():
        global strokes, points
        canvas.delete("all")
        strokes = []
        points = []

    def start_paint(event):
        global points
        points = [event.x, event.y]

    def paint(event):
        global points
        x1, y1 = points[-2], points[-1]
        x2, y2 = event.x, event.y
        canvas.create_line(x1, y1, x2, y2, fill="#000000")
        points.extend([x2, y2])

    def end_paint(event):
        global points, strokes
        strokes.append(points)
        
    def denoise(coords):
        coords = np.split(coords, np.where(coords[:, 0] == 1)[0] + 1, axis=0)
        new_coords = []
        for stroke in coords:
            if len(stroke) != 0:
                x_new = savgol_filter(stroke[:, 1], 7, 3, mode='nearest')
                y_new = savgol_filter(stroke[:, 2], 7, 3, mode='nearest')
                xy_coords = np.hstack([x_new.reshape(-1, 1), y_new.reshape(-1, 1)])
                stroke = np.concatenate([stroke[:, 0].reshape(-1, 1), xy_coords], axis=1)
                new_coords.append(stroke)

        coords = np.vstack(new_coords)
        return coords

    def normalize(offsets):
        offsets = np.copy(offsets)
        offsets[:, 1:] /= np.median(np.linalg.norm(offsets[:, 1:], axis=1))
        return offsets

    def key_press(event):
        global strokes, points
        if event.char == 'g':
            stroke_list = PyInk.InkStrokeList()
            for stroke in strokes:
                x, y = stroke[::2], stroke[1::2]
                stroke_list.append(np.array([x, y]).squeeze())

            seq = []
            for stroke in PyInk.UniformInkStrokes(stroke_list, 128):
                stroke_arr = np.zeros((len(stroke[0]), 3))
                stroke_arr[-1, 0] = 1
                stroke_arr[:, 1], stroke_arr[:, 2] = stroke[0], -stroke[1]
                seq.append(stroke_arr)
            seq = np.vstack(seq)

            seq = denoise(seq)
            seq[1:, 1:] = seq[1:, 1:] - seq[:-1, 1:]
            seq[0, 1:] = 0.
            seq = normalize(seq)
            
            generate_conditionally('test generation ', state_dict_file='save_pureeng_full_rms_continuewithlr0.0001flat\\epoch_120.pt', cell_size=512, char_to_code_file='datasets\\data_inkwell_preprocessed\\char_to_code.pt', bias=1., bias2=1., priming_x=torch.Tensor(seq).unsqueeze(0).cuda(), priming_text='hello there')
            #generate_conditionally('test generation ', state_dict_file='save_se_adam_lr0.001\\epoch_80.pt', cell_size=512, char_to_code_file='datasets\\data_inkwell_preprocessed\\char_to_code.pt', bias=1., bias2=1., z_size=256, x_r=torch.Tensor(seq).unsqueeze(0).cuda())#, priming_text='hello there')
        elif event.char == 'c':
            clear()

    global strokes
    strokes = []

    root = tkinter.Tk()
    root.title("Handwriting Demo App")
    root.geometry("3000x2100")
    root.bind("<KeyPress>", key_press)

    canvas = tkinter.Canvas(root, width=1800, height=1200)
    canvas.bind("<ButtonPress>", start_paint)
    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<ButtonRelease>", end_paint)
    canvas.pack()

    tkinter.mainloop()

if __name__ == "__main__":
    main()