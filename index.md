# BigGAN Audio Visualizer

## Inspiration

This project was inspired by my desire to explore the intersection of audio and image processing. After doing some research, I came across [videos](https://youtu.be/A55NzPmB5PE?t=96) exploring the latent space of [BigGAN (Brock et al., 2018)](https://arxiv.org/abs/1809.11096). BigGAN differs from traditional GAN in that is it truly ***big***, containing over 300 million trainable parameters. As a result, interpolating across the latent image space contains a lot of rich, complex structure. My project uses audio processing (as well as NLP) to control the interpolation within this latent space, and is deeply inspired by the Matt Siegelman's approach [here](https://towardsdatascience.com/the-deep-music-visualizer-using-sound-to-explore-the-latent-space-of-biggan-198cd37dac9a). Using this technique, we can produce trippy synthetic music videos as seen below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/3m8v3Rt9-YE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Audio Processing

The BigGAN architectures takes two vectors of input:

1. a class vector of shape (1000,) representing the weights corresponding to [1000 ImageNet classes](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
2. a noise vector of shape (128,) with weights {$-2 \leq 2$}

In order to process our audio into a sequence of these vectors, we first compute a chromagram and spectrogram of our input file. 
- We filter the chromagram to find the highest power pitch class for each frame, each of which is associated with some ImageNet `cid`. We then construct the class vectors using random directions weighted by the respective `cid` for each frame.
- We use the spectrogram mean gradient and power at each frame to compute the noise vectors, applying appropriate jitter and smoothing to both vectors to ensure a somewhat random walk across the latent space.

## Frame Generation

Once our noise and class vectors are defined, frame generation is as simple as piping our vectors, alongside a tunable truncation parameter, through the pretrained BigGAN. 

Given the size of BigGAN, the computational intensity of this operation is significant. On my local machine, generating a minute worth of frames at `512x512` resolution takes ~7 hours. To produce the examples below, I utilized cloud GPU providers to greatly reduce runtime.

## Hyperparameters

There are a number of hyperparameters responsible for controlling the output video, they are:
- `pitch sensitivity`
- `tempo sensitivity`
- `jitter`
- `truncation`
- `smooth factor`

`Pitch sensitivity` and `tempo sensitivity` control the rate of change of the class/noise vectors respectively, while `jitter` controls the magnitude of update to the noise vectors in each frame. `Truncation` controls the variability of output images, while the `smooth factor` controls the smoothness of interpolation between the class vectors.

## Class Generation

In order to implement a smarter choice of classes, my idea was to use a similarity metric between encodings of the lyrics corpus and each ImageNet class. Following the framework of this [paper](https://arxiv.org/abs/1908.10084), I choose a Siamese BERT network to encode the sentences, and compared semantic similarity using a cosine-similarity metric. I then select the most similar `[num_classes]` unique classes to use for frame generation.

<iframe width="560" height="315" src="https://www.youtube.com/embed/badjh3FQuUA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In the above example, semantic similarity between the lyrics `dough` and `chicken chili` causes the algorithm to select several ImageNet classes associated with food. This is a failure to account for the contextual meaning of the text. An interesting future development would be to implement sentiment analysis, alongside other contextual NLP techniques, to further account for subjective differences.

## Examples

<iframe width="560" height="315" src="https://www.youtube.com/embed/8JY0UdOaHfs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/gxf-4iyurvE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/badjh3FQuUA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/30RK_HeV3Is" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/3m8v3Rt9-YE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/JZMo-WK-xbs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Making Your Own

Use the provided [Google Colab notebook](https://colab.research.google.com/github/rushk014/biggan-visualizer/blob/master/biggan_music_visualizer.ipynb) to generate your own Deep Music Videos!