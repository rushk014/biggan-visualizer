
BigGAN Audio Visualizer
=======
[![biggan-visualizer-example](https://res.cloudinary.com/marcomontalbano/image/upload/v1639193761/video_to_markdown/images/youtube--30RK_HeV3Is-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=30RK_HeV3Is "biggan-visualizer-example")

# Description

This visualizer explores [BigGAN (Brock et al., 2018)](https://arxiv.org/abs/1809.11096) latent space by using pitch/tempo of an audio file to generate and interpolate between noise/class vector inputs to the model. Classes are chosen manually or optionally using semantic similarity on BERT encodings of a lyrics corpus.

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

In order to speed up runtime, code can be run on [Google Colab](https://research.google.com/colaboratory/) GPUs (or other cloud notebook providers) using `deep_music_visualizer.ipynb` (hosted [here](https://colab.research.google.com/github/rushk014/biggan-visualizer/blob/master/biggan_music_visualizer.ipynb)).

# Arguments

|short|long|default|range|help|
| :--- | :--- | :--- | :--- | :--- |
|`-h`|`--help`|||show this help message and exit|
|`-s`|`--song`|`input/romantic.mp3`||path to input audio file `[REQUIRED]`|
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
||`--batch_size`|`20`||BigGAN batch_size|
|`-o`|`--output_file`|||name of output file stored in output/, defaults to [--song] path base_name|
||`--use_last_vectors`|`False`||set flag to use previous saved class/noise vectors|
||`--use_last_classes`|`False`||set flag to use previous classes|
|`-l`|`--lyrics`|`None`||path to lyrics file; setting [--lyrics LYRICS] computes classes by semantic similarity under BERT encodings|

# Acknowledgments

Thanks to Matt Siegelman for providing the [inspiration](https://towardsdatascience.com/the-deep-music-visualizer-using-sound-to-explore-the-latent-space-of-biggan-198cd37dac9a) as well as a [boilerplate](https://github.com/msieg/deep-music-visualizer) for the project.

# References

- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
- [The Deep Music Visualizer: Using sound to explore the latent space of BigGAN](https://towardsdatascience.com/the-deep-music-visualizer-using-sound-to-explore-the-latent-space-of-biggan-198cd37dac9a)
- [BigGanEx: A Dive into the Latent Space of BigGan](https://thegradient.pub/bigganex-a-dive-into-the-latent-space-of-biggan/)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)