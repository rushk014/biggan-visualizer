import argparse
import moviepy.editor as mpy
import torch
from torch.functional import cdist
from tqdm import tqdm
import librosa
from utils import generate_vectors, get_frame_lim, random_classes, semantic_classes, to_np
import numpy as np
from pytorch_pretrained_biggan import (BigGAN, convert_to_images)
import ast
import os

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song",required=True)
    parser.add_argument("--resolution", default='512')
    parser.add_argument("--duration", type=int)
    parser.add_argument("--pitch_sensitivity", type=int, default=220)
    parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
    parser.add_argument("--classes", nargs='+', type=int)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--smooth_factor", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--use_last", action="store_true", default=False)
    parser.add_argument("--lyrics")
    return parser


def visualize(cvs, nvs, model, batch_size, frame_lim):
    frames = []
    for i in tqdm(range(frame_lim)):
        if (i+1)*batch_size > len(class_vectors):
            torch.cuda.empty_cache()
            break
        
        noise_vector = nvs[i*batch_size:(i+1)*batch_size]
        class_vector = cvs[i*batch_size:(i+1)*batch_size]

        with torch.no_grad():
            output = model(noise_vector, class_vector, 1)

        output_images = convert_to_images(output.cpu())
        frames.extend(np.array([to_np(i) for i in output_images]))
        torch.cuda.empty_cache()
    return frames

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    song = args.song
    frame_length = args.frame_length
    pitch_sensitivity = (300-args.pitch_sensitivity) * 512 / frame_length
    tempo_sensitivity = args.tempo_sensitivity * frame_length / 512
    model_name = 'biggan-deep-' + args.resolution
    jitter = args.jitter
    batch_size = args.batch_size
    smooth_factor = int(args.smooth_factor * 512 / frame_length)
    use_last = args.use_last
    num_classes = 12 # --classes must use 12
    if args.output_file:
        outname = 'output/' + args.output_file
    else:
        outname = 'output/' + os.path.basename(args.song).split('.')[0] + '.mp4'

    print(outname)
    print('Reading audio\n')
    y, sr = librosa.load(song)

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load class names
    with open('imagenet_labels.txt','r') as labels:
        c_dict = ast.literal_eval(labels.read())

    if args.duration:
        frame_lim = get_frame_lim(args.duration, frame_length, batch_size)
    else:
        frame_lim = get_frame_lim(len(y)/sr, frame_length, batch_size)
    if args.classes and args.lyrics:
        raise ValueError("Must use either semantic similarity on lyrics or chosen classes")
    elif args.classes:
        classes = args.classes
        assert len(classes) == 12, "must select 12 unique classes"
    elif args.lyrics:
        classes = semantic_classes(args.lyrics, c_dict, num_classes=num_classes, device=device)
    else:
        classes = random_classes(num_classes=num_classes)
    
    # print class names
    

    print('Chosen classes: \n')
    for cid, c in enumerate(classes):
        print(cid, c_dict[c])
    
    # Load pre-trained model
    print('Loading BigGAN \n')
    model = BigGAN.from_pretrained(model_name)

    print('Generating vectors \n')
    class_vectors, noise_vectors = generate_vectors(y, sr, tempo_sensitivity, pitch_sensitivity, classes=classes, preload=use_last)    
    noise_vectors = torch.Tensor(np.array(noise_vectors))      
    class_vectors = torch.Tensor(np.array(class_vectors))      

    # Generate frames in batches of batch_size
    print('Generating frames \n')
    model = model.to(device)
    noise_vectors = noise_vectors.to(device)
    class_vectors = class_vectors.to(device)
    frames = visualize(class_vectors, noise_vectors, model, batch_size, frame_lim)

    #Save video  
    aud = mpy.AudioFileClip(song, fps=44100) 
    if args.duration:
        aud.duration = args.duration
    clip = mpy.ImageSequenceClip(frames, fps=22050/frame_length)
    clip = clip.set_audio(aud)
    clip.write_videofile(outname, audio_codec='aac')