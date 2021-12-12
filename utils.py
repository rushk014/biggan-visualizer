import librosa
import numpy as np
import random
from pytorch_pretrained_biggan import truncated_noise_sample

CV_SIZE = 1000
NV_SIZE = 128

def generate_power(y, sr, frame_length=512):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=frame_length)
    spec_mean = np.mean(spec,axis=0)
    grad_mean = np.gradient(spec_mean)
    grad_mean = np.clip(grad_mean/np.max(grad_mean), 0, None)
    spec_mean = (spec_mean-np.min(spec_mean))/np.ptp(spec_mean)
    return spec_mean, grad_mean

def generate_chroma(y, sr, frame_length=512):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)
    chromasort = np.argsort(np.mean(chroma, axis=1))[::-1]
    return chroma, chromasort

def random_classes(num_classes=12):
    classes = list(range(CV_SIZE))
    random.shuffle(classes)
    return classes[:num_classes]

def get_sensitivity(jitter=0.5):
    return np.random.choice([1, 1-jitter], size=NV_SIZE)

def get_frame_lim(seconds, frame_length, batch_size):
    return int(np.floor(seconds*22050/frame_length/batch_size))

def default_cv(chroma, chromasort, classes):
    cv = np.zeros(CV_SIZE)
    for pi, p in enumerate(chromasort[:len(classes)]):
        if len(classes) < 12:
            cv[classes[pi]] = chroma[p][np.min([np.where(ch>0)[0][0] for ch in chroma])]
        else:
            cv[classes[p]] = chroma[p][np.min([np.where(ch>0)[0][0] for ch in chroma])]
    return cv

def normalize_cv(cv):
    min_cv = min(cv[cv != 0])
    cv[cv == 0] = min_cv
    return (cv-min_cv)/np.ptp(cv)

def new_update_dir(nv, update_dir, tempo_sensitivity, truncation):
    update_dir[nv >= 2*truncation - tempo_sensitivity] = -1
    update_dir[nv < -2*truncation + tempo_sensitivity] = 1
    return update_dir

def smooth(class_vectors,smooth_factor):
    class_vectors_sm = []
    for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):  
        ci = int(c*smooth_factor)          
        cva = np.mean(class_vectors[ci:ci+smooth_factor],axis=0)
        cvb = np.mean(class_vectors[ci+smooth_factor:ci+smooth_factor*2],axis=0)
        for j in range(smooth_factor):
            terp_frac = j/(smooth_factor-1)
            cvc = cva * (1-terp_frac) + cvb * terp_frac                                          
            class_vectors_sm.append(cvc)
    return np.array(class_vectors_sm)

def generate_vectors(y, sr, tempo_sensitivity, pitch_sensitivity, classes, preload, truncation):
    if preload:
        return np.load('saved_vectors/class_vectors.npy'), np.load('saved_vectors/noise_vectors.npy')
    if classes is None:
        classes = random_classes()
    spec_mean, grad_mean = generate_power(y, sr)
    chroma, chromasort = generate_chroma(y, sr)
    cv1, nv1 = default_cv(chroma, chromasort, classes), truncated_noise_sample(truncation=truncation)[0]
    cvs, nvs = [cv1], [nv1]
    update_dir = np.where(nv1 < 0, 1, -1)
    update_last = np.zeros(NV_SIZE)
    for f in range(len(spec_mean)):
        cvlast, nvlast = cvs[f], nvs[f]
        if f%200 == 0:
            jitters = get_sensitivity()
        
        update = np.full(NV_SIZE, tempo_sensitivity) * (grad_mean[f]+spec_mean[f]) * update_dir * jitters 
        update = (update+update_last*3)/4
        nv = nvlast + update
        cv = cvlast
        for cid in range(len(classes)):
            cv[classes[cid]] += ((chroma[chromasort[cid]][f])/(pitch_sensitivity))/(1+(1/(pitch_sensitivity)))
        
        if np.std(cv[cv != 0]) < 1e-7:
            cv[classes[0]] = cv[classes[0]]+0.01
        cv = normalize_cv(cv)

        nvs.append(nv)
        cvs.append(cv)
        update_dir = new_update_dir(nv, update_dir, tempo_sensitivity, truncation)
        update_last = update
    
    np.save('saved_vectors/class_vectors.npy', cvs)
    np.save('saved_vectors/noise_vectors.npy', nvs)
    return cvs, nvs

def to_np(p_img):
    return np.array(p_img)