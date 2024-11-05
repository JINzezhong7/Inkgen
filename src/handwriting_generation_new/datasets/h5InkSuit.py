import h5py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath( __file__ + "/../" )))
import PyInk
import numpy as np

class h5InkWord:
    def __init__(self, strokes, label):
        self.strokes = strokes
        self.label = label

class h5InkPanel:
    def __init__(self, h5file, index):
        self.h5file = h5file
        self.index = index

    def GetAllStrokes(self):
        strokes = PyInk.InkStrokeList()
        strokedata = self.h5file['strokes'][self.index]
        strokedata = strokedata.reshape((strokedata.size // 2, 2))
        stroke = []
        for row in strokedata:
            if row[0] == row[1] == -1.0:
                if len(stroke) > 0:
                    strokes.append(np.array(stroke, dtype = np.float32).T)
                stroke = []
            else:
                stroke.append(row)
        return strokes

    def GetLabel(self):
        labelarr  = self.h5file['labels'][self.index]
        label = []
        for val in labelarr:
            label.append(chr(val))
        return "".join(label)
    @property
    def words(self):
        if "strokesegments" in self.h5file.keys():
            strokesegments = self.h5file['strokesegments'][self.index]
            if len(strokesegments) == 0:
                return []
            wordsegments = self.h5file['wordsegments'][self.index]
            allstrokes = self.GetAllStrokes()
            alllabels = self.GetLabel()
            allwords = []
            strokeidx = 0
            charidx = 0
            for i in range(len(strokesegments)):
                strokes = []
                chars = []
                for j in range(strokesegments[i]):
                    strokes.append(allstrokes[strokeidx + j])
                strokeidx += strokesegments[i]
                for j in range(wordsegments[i]):
                    chars.append(alllabels[charidx + j])
                charidx += wordsegments[i]
                while (charidx< len(alllabels) and alllabels[charidx] == ' '):
                    charidx += 1
                wordlabel = "".join(chars)
                allwords.append(h5InkWord(strokes, wordlabel))
            return allwords
        else:
            return []

class h5InkSuit:
    def __init__(self, h5file):
        self.h5file = h5py.File(h5file, 'r')
        self.valid = True
        keys = self.h5file.keys()
        if "labels" not in keys or 'strokes' not in keys:
            self.valid = False
        self.panels = [h5InkPanel(self.h5file, i) for i in range(len(self.h5file['strokes']))]