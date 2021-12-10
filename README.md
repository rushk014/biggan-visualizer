
BigGAN Audio Visualizer
=======

# Description

Visualizer using audio and semantic analysis to explore BigGAN (Brock et al., 2018) latent space.

# Usage:


```bash
usage: visualize.py [-h] -s SONG [--resolution {128,256,512}] [-d DURATION]
               [-ps [200-295]] [-ts [0.05-0.8]]
               [--classes CLASSES [CLASSES ...]] [-n NUM_CLASSES]
               [--jitter [0-1]] [--frame_length i*2^6] [--truncation [0.1-1]]
               [--smooth_factor [10-30]] [--batch_size BATCH_SIZE]
               [-o OUTPUT_FILE] [--use_last_vectors] [--use_last_classes]
               [-l LYRICS]

```
# Arguments

|short|long|default|range|help|
| :--- | :--- | :--- | :--- | :--- |
|`-h`|`--help`|||show this help message and exit|
|`-s`|`--song`|`input/romantic.mp3`||path to input audio file|
||`--resolution`|`512`|`{128,256,512}`|output video resolution|
|`-d`|`--duration`|`None`||output video duration|
|`-ps`|`--pitch_sensitivity`|`220`|`[200-295]`|controls the sensitivity of the class vector to changes in pitch|
|`-ts`|`--tempo_sensitivity`|`0.25`|`[0.05-0.8]`|controls the sensitivity of the noise vector to changes in volume and tempo|
||`--classes`|`None`||manually specify [--num_classes] ImageNet classes|
|`-n`|`--num_classes`|`12`||number of unique classes to use|
||`--jitter`|`0.5`|`[0-1]`|controls jitter of the noise vector to reduce repitition|
||`--frame_length`|`512`|`i*2^6`|number of audio frames to video frames in the output|
||`--truncation`|`1`|`[0.1-1]`|BigGAN truncation parameter controls complexity of structure within frames|
||`--smooth_factor`|`20`|`[10-30]`|controls interpolation between class vectors to smooth rapid flucations|
||`--batch_size`|`30`||BigGAN batch_size|
|`-o`|`--output_file`|||name of output file stored in output/, defaults to [--song] path base_name|
||`--use_last_vectors`|`False`||set flag to use previous saved class/noise vectors|
||`--use_last_classes`|`False`||set flag to use previous classes|
|`-l`|`--lyrics`|`None`||path to lyrics file; setting [--lyrics LYRICS] computes classes by semantic similarity under BERT encodings|
