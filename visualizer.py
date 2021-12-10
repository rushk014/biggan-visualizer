import argparse
import moviepy.editor as mpy
import torch
from torch.functional import cdist
from tqdm import tqdm
import librosa
from utils import generate_vectors, random_classes, to_np
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, convert_to_images)
from urllib.request import urlopen
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--song",required=True)
parser.add_argument("--resolution", default='512')
parser.add_argument("--duration", type=int)
parser.add_argument("--pitch_sensitivity", type=int, default=220)
parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
parser.add_argument("--classes", nargs='+', type=int)
parser.add_argument("--num_classes", type=int, default=12)
parser.add_argument("--jitter", type=float, default=0.5)
parser.add_argument("--frame_length", type=int, default=512)
parser.add_argument("--smooth_factor", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--output_file", default="output.mp4")
parser.add_argument("--preload", action="store_true", default=False)
args = parser.parse_args()

song = args.song
frame_length = args.frame_length
pitch_sensitivity = (300-args.pitch_sensitivity) * 512 / frame_length
tempo_sensitivity = args.tempo_sensitivity * frame_length / 512
model_name='biggan-deep-' + args.resolution
num_classes=args.num_classes
jitter=args.jitter
batch_size = args.batch_size
outname = args.output_file
smooth_factor = int(args.smooth_factor * 512 / frame_length)
preload = args.preload

print('\nReading audio \n')
y, sr = librosa.load(song)

if args.duration:
    seconds = args.duration
    frame_lim = int(np.floor(seconds*22050/frame_length/batch_size))
else:
    frame_lim = int(np.floor(len(y)/sr*22050/frame_length/batch_size))
    
# Load pre-trained model
model = BigGAN.from_pretrained(model_name)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.classes:
    classes = args.classes
    if len(classes) not in [12,num_classes]:
        raise ValueError("The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")
else:
    classes = random_classes(num_classes=num_classes)

# print class names
with open('imagenet_labels.txt','r') as labels:
    c_dict = ast.literal_eval(labels.read())

for c in classes:
    print(c_dict[c])

class_vectors, noise_vectors = generate_vectors(y, sr, tempo_sensitivity, pitch_sensitivity, classes=classes, preload=False)    
noise_vectors = torch.Tensor(np.array(noise_vectors))      
class_vectors = torch.Tensor(np.array(class_vectors))      


#Generate frames in batches of batch_size

print('\n\nGenerating frames \n')

#send to CUDA if running on GPU
model=model.to(device)
noise_vectors=noise_vectors.to(device)
class_vectors=class_vectors.to(device)


frames = []

for i in tqdm(range(frame_lim)):
    if (i+1)*batch_size > len(class_vectors):
        torch.cuda.empty_cache()
        break
    
    noise_vector = noise_vectors[i*batch_size:(i+1)*batch_size]
    class_vector = class_vectors[i*batch_size:(i+1)*batch_size]

    with torch.no_grad():
        output = model(noise_vector, class_vector, 1)

    output_images = convert_to_images(output)
    frames.extend(np.array([to_np(i) for i in output_images]))
    torch.cuda.empty_cache()

#Save video  
aud = mpy.AudioFileClip(song, fps=44100) 

if args.duration:
    aud.duration = args.duration

clip = mpy.ImageSequenceClip(frames, fps=22050/frame_length)
clip = clip.set_audio(aud)
clip.write_videofile(outname, audio_codec='aac')