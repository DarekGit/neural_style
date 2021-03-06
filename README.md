# neural_style

This is Colab notebook implementation based on [ProGamerGov](https://github.com/ProGamerGov/neural-style-pt)
following paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The code is based on Justin Johnson's [Neural-Style](https://github.com/jcjohnson/neural-style).

The code was adopted to perform easy in  Colab framework. Weight normalization was fixed and implemented in Losses backwards.
Set of parameters was created in the module for easy usage as well auto manager of results directories helps to compare progress of modified images.

There was chosen best fitted models to human perception. Other Google model adaptations you can find on https://github.com/rdcolema/deepdream-neural-style-transfer.

To start you can download [notebook](neural_style.ipynb).

<div align="center">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/ZD_face.jpg" height="300px">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/DD.jpg" height="300px">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/DD.jpg" height="300px">
</div>

<div align="center">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/frida_kahlo.jpg" height="300px">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/Z.jpg" height="300px">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/Zfrid.jpg" height="300px">
</div>

<br>
The paper presents an algorithm for combining the content of one image with the style of another image using
convolutional neural networks. Here's an example that maps the artistic style of
[The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night)
onto a night-time photograph of the Stanford campus:

<div align="center">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/starry_night_google.jpg" height="223px">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/hoovertowernight.jpg" height="223px">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/starry_stanford_bigger.png" width="710px">
</div>

Applying the style of different images to the same content image gives interesting results.
Here we reproduce Figure 2 from the paper, which renders a photograph of the Tubingen in Germany in a
variety of styles:

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/tubingen.jpg" height="250px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_shipwreck.png" height="250px">

<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_starry.png" height="250px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_scream.png" height="250px">

<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_seated_nude.png" height="250px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_composition_vii.png" height="250px">
</div>

Here are the results of applying the style of various pieces of artwork to this photograph of the
golden gate bridge:


<div align="center"
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/golden_gate.jpg" height="200px">

<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/frida_kahlo.jpg" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_kahlo.png" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/escher_sphere.jpg" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_escher.png" height="160px">
</div>

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/woman-with-hat-matisse.jpg" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_matisse.png" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/the_scream.jpg" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_scream.png" height="160px">
</div>

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/starry_night_crop.png" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry.png" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/seated-nude.jpg" height="160px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_seated.png" height="160px">
</div>

### Content / Style Tradeoff

The algorithm allows the user to trade-off the relative weight of the style and content reconstruction terms,
as shown in this example where we port the style of [Picasso's 1907 self-portrait](http://www.wikiart.org/en/pablo-picasso/self-portrait-1907) onto Brad Pitt:

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/picasso_selfport1907.jpg" height="220px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/inputs/brad_pitt.jpg" height="220px">
</div>

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/pitt_picasso_content_5_style_100.png" height="220px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/pitt_picasso_content_1_style_100.png" height="220px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/pitt_picasso_content_01_style_100.png" height="220px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/pitt_picasso_content_0025_style_100.png" height="220px">
</div>

### Style Scale

By resizing the style image before extracting style features, we can control the types of artistic
features that are transfered from the style image; you can control this behavior with the style_scale params.
Below we see three examples of rendering the Golden Gate Bridge in the style of The Starry Night.
From left to right, params.style_scale is 2.0, 1.0, and 0.5.

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry_scale2.png" height=175px>
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry_scale1.png" height=175px>
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry_scale05.png" height=175px>
</div>

### Multiple Style Images
You can use more than one style image to blend multiple artistic styles.

Clockwise from upper left: "The Starry Night" + "The Scream", "The Scream" + "Composition VII",
"Seated Nude" + "Composition VII", and "Seated Nude" + "The Starry Night"

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_starry_scream.png" height="250px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_scream_composition_vii.png" height="250px">

<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_starry_seated.png" height="250px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_seated_nude_composition_vii.png" height="250px">
</div>


### Style Interpolation
When using multiple style images, you can control the degree to which they are blended:

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry_scream_3_7.png" height="175px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry_scream_5_5.png" height="175px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/golden_gate_starry_scream_7_3.png" height="175px">
</div>

### Transfer style but not color
If you set the params.original_colors to 1 then the output image will retain the colors of the original image.

<div align="center">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_starry.png" height="185px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_scream.png" height="185px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/tubingen_composition_vii.png" height="185px">

<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/original_color/tubingen_starry.png" height="185px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/original_color/tubingen_scream.png" height="185px">
<img src="https://github.com/DarekGit/neural_style/blob/main/examples/outputs/original_color/tubingen_composition_vii.png" height="185px">
</div>

## Setup:

Dependencies:
* [PyTorch](http://pytorch.org/)


Optional dependencies:
* For CUDA backend:
  * CUDA 7.5 or above
* For cuDNN backend:
  * cuDNN v6 or above
* For ROCm backend:
  * ROCm 2.1 or above
* For MKL backend:
  * MKL 2019 or above
* For OpenMP backend:
  * OpenMP 5.0 or above



## Usage
You can observe progress of transformation in Colab notebook as well results are recoded in Outputs directory together with configuration file.

<div align="center">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/configs/OUT_Figs_000006.jpg" width="1000px">
</div>
<div align="center">
 <img src="https://github.com/DarekGit/neural_style/blob/main/examples/configs/OUT_000006.jpg" height="400px">
</div>

<br><br>
To use multiple style images, pass a comma-separated list like this:

`params.style_image = "starry_night.jpg,the_scream.jpg"`.

Note that paths to images should not contain the `~` character to represent your home directory; you should instead use a relative
path or a full absolute path.

**Options**:
* `image_size`: Maximum side length (in pixels) of the generated image. Default is 512.
* `style_blend_weights`: The weight for blending the style of multiple style images, as a
  comma-separated list, such as `style_blend_weights 3,7`. By default all style images
  are equally weighted.
* `gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `gpu` to `c`.

**Optimization options**:
* `content_weight`: How much to weight the content reconstruction term. Default is 5e0.
* `style_weight`: How much to weight the style reconstruction term. Default is 1e2.
* `tv_weight`: Weight of total-variation (TV) regularization; this helps to smooth the image.
  Default is 1e-3. Set to 0 to disable TV regularization.
* `num_iterations`: Default is 1000.
* `init`: Method for generating the generated image; one of `random` or `image`.
  Default is `random` which uses a noise initialization as in the paper; `image`
  initializes with the content image.
* `init_image`: Replaces the initialization image with a user specified image.
* `optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `lbfgs`.
  L-BFGS tends to give better results, but uses more memory. Switching to ADAM will reduce memory usage;
  when using ADAM you will probably need to play with other parameters to get good results, especially
  the style weight, content weight, and learning rate.
* `learning_rate`: Learning rate to use with the ADAM optimizer. Default is 1e1.

**Output options**:
* `output_image`: Name of the output image. Default is `out.jpg`.
* `print_iter`: Print progress every `print_iter` iterations. Set to 0 to disable printing.
* `save_iter`: Save the image every `save_iter` iterations. Set to 0 to disable saving intermediate results.

**Layer options**:
* `content_layers`: Comma-separated list of layer names to use for content reconstruction.
  Default is `relu4_2`.
* `style_layers`: Comma-separated list of layer names to use for style reconstruction.
  Default is `relu1_1,relu2_1,relu3_1,relu4_1,relu5_1`.

**Other options**:
* `style_scale`: Scale at which to extract features from the style image. Default is 1.0.
* `original_colors`: If you set this to 1, then the output image will keep the colors of the content image.
* `model_file`: Path to the `.pth` file for the VGG Caffe model. Default is the original VGG-19 model; you can also try the original VGG-16 or NIN model.
* `pooling`: The type of pooling layers to use; one of `max` or `avg`. Default is `max`.
  The VGG-19 models uses max pooling layers, but the paper mentions that replacing these layers with average
  pooling layers can improve the results. 
* `seed`: An integer value that you can specify for repeatable results. By default this value is random for each run.
* `multidevice_strategy`: A comma-separated list of layer indices at which to split the network when using multiple devices. See [Multi-GPU scaling](https://github.com/ProGamerGov/neural-style-pt#multi-gpu-scaling) for more details.
* `backend`: `nn`, `cudnn`, `openmp`, or `mkl`. Default is `nn`. `mkl` requires Intel's MKL backend.
* `cudnn_autotune`: When using the cuDNN backend, pass this flag to use the built-in cuDNN autotuner to select
  the best convolution algorithms for your architecture. This will make the first iteration a bit slower and can
  take a bit more memory, but may significantly speed up the cuDNN backend.

## Frequently Asked Questions

**Problem:** The program runs out of memory and dies

**Solution:** Try reducing the image size: `image_size 256` (or lower). Note that different image sizes will likely
require non-default values for `style_weight` and `content_weight` for optimal results.
If you are running on a GPU, you can also try running with `backend cudnn` to reduce memory usage.

**Problem:** `backend cudnn` is slower than default NN backend

**Solution:** Add the flag `cudnn_autotune`; this will use the built-in cuDNN autotuner to select the best convolution algorithms.

**Problem:** Get the following error message:

`Missing key(s) in state_dict: "classifier.0.bias", "classifier.0.weight", "classifier.3.bias", "classifier.3.weight".
        Unexpected key(s) in state_dict: "classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias".`

**Solution:** Due to a mix up with layer locations, older models require a fix to be compatible with newer versions of PyTorch. The included [`donwload_models.py`](https://github.com/ProGamerGov/neural-style-pt/blob/master/models/download_models.py) script will automatically perform these fixes after downloading the models.



## Memory Usage
By default, `neural_style` uses the `cudnn` backend for convolutions and L-BFGS for optimization.

* **Use ADAM**: Set the params `optimizer adam` to use ADAM instead of L-BFGS. This should significantly
  reduce memory usage, but may require tuning of other parameters for good results; in particular you should
  play with the learning rate, content weight, and style weight.
  This should work in both CPU and GPU modes.
* **Reduce image size**: If the above tricks are not enough, you can reduce the size of the generated image;
  set the params `image_size 256` to generate an image at half the default size.

With the default settings, neural_style uses about 3.7 GB of GPU memory ; switching to ADAM and cuDNN reduces the GPU memory footprint to about 1GB.

## Speed
Speed can vary a lot depending on the backend and the optimizer.
Here are some times for running 500 iterations with `image_size=512` on a Tesla K80 with different settings:
* `backend nn  optimizer lbfgs`: 117 seconds
* `backend nn  optimizer adam`: 100 seconds
* `backend cudnn  optimizer lbfgs`: 124 seconds
* `backend cudnn  optimizer adam`: 107 seconds
* `backend cudnn  cudnn_autotune  optimizer lbfgs`: 109 seconds
* `backend cudnn  cudnn_autotune  optimizer adam`: 91 seconds


## Multi-GPU scaling
Not available in Colab.

## Implementation details
Images are initialized with white noise and optimized using L-BFGS.

We perform style reconstructions using the `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1` layers
and content reconstructions using the `conv4_2` layer. As in the paper, the five style reconstruction losses have
equal weights.

## Citation

If you find this code useful for your research, please cite:

```
@misc{DDzialkowski2020,
  author = {Dariusz Dzialkowsk},
  title = {neural_style},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DarekGit/neural_style}},
}
```
