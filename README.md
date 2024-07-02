# CUDA Tiny Model - A simple neural network for predicting next character.
A Tiny model for character generation using CUDA and CUDNN. This model is based on the Auto-regressive character model presented here https://github.com/karpathy/makemore (by Andrej Karapathy). 

## Project Description
The project creates a tiny neural network for auto-regressive character level generation.  This is entirely based on the youtube video (and associated resoruces) by Andrej Karapathy's makemore project. 

The model takes a string as input and predicts the next character in the string.  The model consists of an embedding layer --> linear layer --> tanh activation -> linear layer --> softmax activation. 

The model is trained using 'names' from an input text file. The current implementation **ONLY** trains the model. The output shows loss reducing with each training epoch. 

The output.text contains an output from one training run. 

Future Improvements:
* Add optimizers like Adam for better training efficiency and stability.
* Add model validation and evaluation steps.
* Improve model parameter initialization using (use Xavier etc).
* Add regularization and batch norm.
* Add inference functionalaity. 

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.
