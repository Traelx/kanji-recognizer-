import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps
from main import Model, jis_to_char


checkpoint = torch.load('model.pt', map_location='cpu') #safety 
labels = [jis_to_char(code) for code in checkpoint['unique_labels']] #get kanji / hiragana
model = Model(len(labels)) # init 
model.load_state_dict(checkpoint['model']) #load weights
model.eval()

def preprocess(rgba):
    
    alpha = rgba[:, :, 3]
    gray = rgba[:, :, :3].mean(axis=2) # gray scale
    ink_mask = ((alpha > 128) & (gray < 128)).astype(np.uint8) * 255 #only dark pixels
    
    img = Image.fromarray(ink_mask)
    
   
    bbox = img.getbbox() #measure

    if bbox is None:
        return None
    
    cropped = img.crop(bbox) #crop
    
    
    #resize 
    centered = ImageOps.pad(cropped, (64, 63), method=Image.BILINEAR, color=0)
    
    
    return (np.array(centered) > 64).astype(np.uint8) #convert to binary


def predict(sketch):
    if sketch is None or sketch['composite'] is None:
        return {}
    img = preprocess(sketch['composite'])
    if img is None:
        return {}

    
    x = torch.from_numpy(img).float()[None, None]  # shape (1, 1, 63, 64)
    with torch.no_grad(): # don't track 
        probs = torch.softmax(model(x), dim=1)[0]
    top5 = probs.topk(5)
    return {labels[i]: float(p) for p, i in zip(top5.values, top5.indices)} #return dict w kanji and prob


demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        canvas_size=(400, 400),
        brush=gr.Brush(default_size=16, colors=['#000000'], color_mode='fixed'),
        label='Draw a kanji or hiragana',
    ),
    outputs=[
        gr.Label(num_top_classes=5, label='Top 5 predictions'),
    ],

    live=True,
    title='Handwritten Kanji Recognizer',
    description='A CNN trained on the ETL8B dataset (956 kanji and hiragana)',
)

if __name__ == '__main__':
    demo.launch()
