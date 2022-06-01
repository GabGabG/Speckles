# Speckles
Code related to speckle analysis and simulations. For now, it is mostly in testing and development phase, so there is no guarantee everything works properly.

## Main code
The main code can be found in the `Speckles` directory, right [here](https://github.com/GabGabG/Speckles/tree/main/Speckles). Then, in that directory there are 2 submodules: [`SpeckleAnalysis`](https://github.com/GabGabG/Speckles/tree/main/Speckles/SpeckleAnalysis) and [`SpeckleSimulations`](https://github.com/GabGabG/Speckles/tree/main/Speckles/SpeckleSimulations). The analysis one is still in early development, while the simulations one is more advanced and almost fully ready to use. Some features are not quite fully implemented.

### `SpeckleAnalysis`
Speckle analysis ranges from finding the average speckle size to full blown contrast analysis to have high contrast and quality images. This submodules aims at providing both simple analysis (i.e. simple statistics about speckle patterns) and more complete and complex analysis (i.e. higher order statistics and contrast analysis). Since it is still in early development and I am the only developer for the code (as I am writing this), it is normal that only some of the basic stuff is present. But rest assured that it will be there in the near future, since it is an important part of my project.

Currently:
- Speckle size can be obtained from a single image or multiple images
- Local contrast can be obtained from a single image or multiple images

TODO:
- Simple stats (mean, variance, etc.)
- Fits (exponential, gamma, etc.) to _visually_ infer developedness (is that even a word?!)
- Correlation / decorrelation time of multiple images (really useful)
- Rework the code so it is pleasant to read **AND** use

**I am open to suggestions and help!**

### `SpeckleSimulations`
This is the most advanced submodule. There are not a lot of laser speckles simulation packages for _Python_. There is https://github.com/scottprahl/pyspeckle by Scott Prahl. It is somewhat powerful, yet there are not many options available. This is why this submodule was created (also because it helps to understand!). There are two categories of speckles:
1. Fully developed
2. Partially developed

Without going into the mathematical details (see _Speckle phenomena in optics: theory and applications_ by J. W. Goodman for that), both types are statistically distinct and thus different algorithms must be employed to simulate them. Then, once simulated, one can better understand the fundamental differences between the two categories. This is why it is an important aspect of our code to allow both.

It is also important to be able to generate speckles in a dynamic way. For example, we could be interested in the decorrelation of certain speckle patterns following a certain decorrelation function. This is why is it an important aspect of our code to allow both dynamic and static speckle simulations. In the end, we want to offer the most possibilities to users and we want them to be simple, yet elegant, to use.

Currently:
- Fully developed speckles can be statically simulated
- Partially developed speckles can be **somewhat** simulated when generating a sum of speckles
- Partially polarized speckles can be simulated
- Dynamic speckles can be simulated in different ways:
  1. Movement of the sample (or, it this case the pupil of the imaging setup simulation)
  2. Brownian motion of the sample
  3. Linear electric field decorrelation (results in a quadratic intensity decorrelation)

TODO:
- Speckle decorrelation with arbitrary tensor (for spatial *and* temporal correlation / decorrelation)
- Partially developed speckles with arbitrary phase distribution
- Rework the code so it is pleasant to read **AND** use

**I am open to suggestions and help!**

Note that the dynamic and static speckles simulations are based on:

Donald D. Duncan, Sean J. Kirkpatrick, "Algorithms for simulation of speckle (laser and otherwise)," Proc. SPIE 6855, Complex Dynamics and Fluctuations in Biomedical Photonics V, 685505 (6 February 2008); https://doi.org/10.1117/12.760518

Lipei Song, Zhen Zhou, Xueyan Wang, Xing Zhao, and Daniel S. Elson, "Simulation of speckle patterns with pre-defined correlation distributions," Biomed. Opt. Express 7, 798-809 (2016)

## Examples
There are two _Jupyter Notebooks_ present [here](https://github.com/GabGabG/Speckles/tree/main/examples). The first one to start with, [`staticSpeckleSimulations.ipynb`](https://github.com/GabGabG/Speckles/blob/main/examples/staticSpeckleSimulations.ipynb), is to introduce the code for static simulations . The second one,  [`dynamicSpeckleSimulations.ipynb`](https://github.com/GabGabG/Speckles/blob/main/examples/dynamicSpeckleSimulations.ipynb), is for dynamic simulations. Both detail the code, the functions and how to use them.
