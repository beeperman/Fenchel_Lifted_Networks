Author: Fangda Gu (gfd18@berkeley.edu)

Date: July 28th 2020

This is an implementation of batched lifted neural network in accordance to the paper *Fenchel lifted networks: A lagrange relaxation of neural network training*.

There are scripts in the working directory:

    layers.py
        In this file, the layer classes are implemented.
        There are FCLayer, FC2Layer(Fenchel), ConvLayer and FINLayer.
        FINLayer is used when adding additional variables in the final layer.
    model.py
        In this file, the batched LNN is implemented. Parameters and network structure can be specified.
    nn.py
        In this file, the Tensorflow helper functions that include computations are implemented.
    utils.py
        In this file, some other helper functions are defined.
    ridge.py
        In this file, the sklearn ridge regression is wrapped into a optimizer that can be used by LNN
    closed.py
        In this file, optimizers that contain only closed form computations are implemented.
        Only a closed form solution optimizer for a special case is implemented.
        (original LNN, regression problem, additional variables in the final layer)
Other scripts are testing files whose names suggest their use. They start with the name of dataset (mnist for now).
Some phrase definition is given below.

    lenet:  LeNet-5
    batch:  batched Fenchel LNN
Tested on Tensorflow 1.15

If you find the code useful, please consider citing the paper. 

    @inproceedings{gu2020fenchel,
      title={Fenchel lifted networks: A lagrange relaxation of neural network training},
      author={Gu, Fangda and Askari, Armin and El Ghaoui, Laurent},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      pages={3362--3371},
      year={2020}
    }
