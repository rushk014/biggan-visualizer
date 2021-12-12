BigGAN Audio Visualizer
=======
[![biggan-visualizer-example](https://res.cloudinary.com/marcomontalbano/image/upload/v1639287854/video_to_markdown/images/youtube--3m8v3Rt9-YE-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=3m8v3Rt9-YE "biggan-visualizer-example")

# Description

This visualizer explores [BigGAN (Brock et al., 2018)](https://arxiv.org/abs/1809.11096) latent space by using pitch/tempo of an audio file to generate and interpolate between noise/class vector inputs to the model. Classes are chosen manually or optionally using semantic similarity on BERT encodings of a lyrics corpus.

# Usage:

```bash
usage: visualizer.py [-h] -s SONG [-r {128,256,512}] [-d DURATION]
                     [-ps [200-295]] [-ts [0.05-0.8]]
                     [-c CLASSES [CLASSES ...]] [-n NUM_CLASSES] [-j [0-1]]
                     [-fl i*2^6] [-t [0.1-1]] [-sf [10-30]] [-bs BATCH_SIZE]
                     [-o OUTPUT_FILE] [--use_last_vectors]
                     [--use_last_classes] [--sort_pitch] [-l LYRICS]
                     [-e {sbert,doc2vec}] [-es {best,random,ransac}]

```

- In order to speed up runtime, code can be run on [Google Colab](https://research.google.com/colaboratory/) GPUs (or other cloud notebook providers) using `biggan_music_visualizer.ipynb` (hosted [here](https://colab.research.google.com/github/rushk014/biggan-visualizer/blob/master/biggan_music_visualizer.ipynb)).
- The `[-n NUM_CLASSES]` parameter selects the number of classes to interpolate between. 
- Default behavior is to select `[-n NUM_CLASSES]` random classes. The `[-c CLASSES [CLASSES ...]]` parameter can be used to select specific ImageNet classes. A full list can be found [here](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/), and a list categorized by coarse descriptors [here](https://github.com/noameshed/novelty-detection/blob/master/imagenet_categories.csv). Be sure to use the `int` ids and not the `string` labels, and set `[-n NUM_CLASSES]` to the number of chosen classes.
- Use the `[--sort_by_power]` flag to map classes to the `[-n NUM_CLASSES]` highest power pitches. By default, classes are mapped to a chromatic scale.
- The `[-d DURATION]` parameter can be useful to generate short videos while tweaking other parameters. Once the desired parameters are set, use the `[--use_last_vector]` flag and remove the `[-d DURATION]` parameter to generate the same video at full length.
- Reducing the output resolution with `[-r {128,256,512}]` and/or increasing the frame length with `[-fl i*2^6]` can help reduce the runtime.
- To compute classes through semantic similarity to a lyrics file, use the `[-l LYRICS]` parameter. The embedding technique and strategy for choosing classes can be set with `[-e {sbert,doc2vec}]` and `[-es {best,random,ransac}]` respectively.
- Pitch and tempo sensitivity can be set with  `[-ps [200-295]]` and `[-ts [0.05-0.8]]` respectively. Jitter, truncation and smooth factor can be set with `[-j [0-1]]`, `[-t [0.1-1]]` and `[-sf [10-30]]` respectively.
- See the help column of the [`arguments`](#arguments) section for details on all parameters.

# Arguments

|short|long|default|range|help|
| :--- | :--- | :--- | :--- | :--- |
|`-h`|`--help`|||show this help message and exit|
|`-s`|`--song`|||path to input audio file `[REQUIRED]`|
|`-r`|`--resolution`|`512`|`{128,256,512}`|output video resolution|
|`-d`|`--duration`|`None`|`int`|output video duration|
|`-ps`|`--pitch_sensitivity`|`220`|`[200-295]`|controls the sensitivity of the class vector to changes in pitch|
|`-ts`|`--tempo_sensitivity`|`0.25`|`[0.05-0.8]`|controls the sensitivity of the noise vector to changes in volume and tempo|
|`-c`|`--classes`|`None`||manually specify `[--num_classes]` ImageNet classes|
|`-n`|`--num_classes`|`12`|`[1-12]`|number of unique classes to use|
|`-j`|`--jitter`|`0.5`|`[0-1]`|controls jitter of the noise vector to reduce repitition|
|`-fl`|`--frame_length`|`512`|`i*2^6`|number of audio frames to video frames in the output|
|`-t`|`--truncation`|`1`|`[0.1-1]`|BigGAN truncation parameter controls complexity of structure within frames|
|`-sf`|`--smooth_factor`|`20`|`[10-30]`|controls interpolation between class vectors to smooth rapid flucations|
|`-bs`|`--batch_size`|`20`|`int`|BigGAN batch_size|
|`-o`|`--output_file`|||name of output file stored in `output/`, defaults to `[--song]` path base_name|
||`--use_last_vectors`|`False`|`bool`|set flag to use previous saved class/noise vectors|
||`--use_last_classes`|`False`|`bool`|set flag to use previous classes|
||`--sort_pitches`|`False`|`bool`|set flag to sort pitches by the ordering of classes|
|`-l`|`--lyrics`|`None`||path to lyrics file; setting `[--lyrics LYRICS]` computes classes by semantic similarity under BERT encodings|
|`-e`|`--encoding`|`sbert`|`{sbert,doc2vec}`|controls choice of sentence embeddings technique|
|`-es`|`--encoding_strategy`|`None`|`{random,best,ransac}`|controls strategy for choosing classes: `[-e sbert]` can use `best` or `random` while `[-e doc2vec]` can use `ransac`|

# Acknowledgments

Thanks to Matt Siegelman for providing the [inspiration](https://towardsdatascience.com/the-deep-music-visualizer-using-sound-to-explore-the-latent-space-of-biggan-198cd37dac9a) as well as a [boilerplate](https://github.com/msieg/deep-music-visualizer) for the project.

# References

- [The Deep Music Visualizer: Using sound to explore the latent space of BigGAN](https://towardsdatascience.com/the-deep-music-visualizer-using-sound-to-explore-the-latent-space-of-biggan-198cd37dac9a)
- [BigGanEx: A Dive into the Latent Space of BigGan](https://thegradient.pub/bigganex-a-dive-into-the-latent-space-of-biggan/)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)