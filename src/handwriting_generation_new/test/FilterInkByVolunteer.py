
import os
import sys
import PyInk
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image


def Recognize(processor, model, strokeslist, tmpPath):
    for i, strokes in enumerate(strokeslist):
        box = PyInk.BoundingRect(strokes)
        width = box.right - box.left + 1
        height = box.bottom - box.top + 1
        if width < height*2:
            width = 2*height
        dpi = width/10
        padding = height/20
        fig, ax = plt.subplots(figsize=( width/dpi + 1, height/dpi + 1))
        ax.axis('off')
        #plt.xlim((box.left - padding, box.right + padding))
        #plt.ylim((-box.bottom  -padding, -box.top + padding))
        ax.set_aspect('equal')
        for stroke in strokes:
            ax.plot(stroke[0, :], -stroke[1, :], color='b', linewidth=2)
        plt.savefig(f'{tmpPath}/ink_{i}.jpg')
        plt.close('all')
    images = [Image.open(f"{tmpPath}/ink_{i}.jpg").convert("RGB") for i in range(len(strokeslist)) ]
    pixel_values = processor(images, return_tensors="pt").pixel_values.to('cuda')
    generated_ids = model.generate(pixel_values)
    recResult = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return recResult


def plots(strokes):
    fig, ax = plt.subplots(1, 1)
    for stroke in strokes:
        ax.plot(stroke[0,:], -stroke[1,:])
    ax.grid()
    return fig

if __name__ == '__main__':
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to('cuda')

    writerId = sys.argv[1]
    path = f"singleLine/{writerId}/panels.xml"
    suit = PyInk.InkSuit(path)
    goodPanels = PyInk.InkPanelList()
    labelworker  = PyInk.LabelWorker('en-US')
    for i in range(0,len(suit.panels),32):
        bgn = i
        end = i + 32 if i + 32 < len(suit.panels) else len(suit.panels) 
        panels = [suit.panels[j] for j in range(bgn,end)]
        strokesList = [panel.GetAllStrokes() for panel in panels]
        recStrs = Recognize(processor, model, strokesList, f"singleLine/{writerId}")
        for j, panel in enumerate(panels):
            stats = PyInk.EvaluateRecoString(labelworker.RelaxString(panel.GetLabel()), labelworker.RelaxString(recStrs[j]))
            if stats.errWord == 0:
                goodPanels.append(panel)

    goodSuit = PyInk.InkSuit(goodPanels)
    os.makedirs(f"filtered/{writerId}", exist_ok=True)
    goodSuit.Export(f"filtered/{writerId}/panels.xml")