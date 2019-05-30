# mpctools
A set of python tools for extending standard (and non-standard) libraries. These originated from my own needs and I 
decided to put them here in case they may be useful to other people.

## Features

The library currently contains the following two packages:
 1. `extensions`: A number of extensions to numpy, sklearn, pandas and matplotlib, as well as general-purpose utilities.
 2. `parallel`: A set of tools for wrapping pathos multiprocessing in a simple easy to use interface with multiple
     parallel workers. 
 
Eventually, I plan to add a neural toolbox.

More details for each library are provided as doxygen-style comments in the modules.

## Setting up

### Requirements

This Library has the following dependencies:
  * scikit-multilearn
  * scikit-learn
  * matplotlib
  * seaborn
  * pandas
  * pathos
  * numpy
  
In most cases, the above can be automatically installed through the library itself (i.e. pip will attempt to download 
them). If this causes issues, just install them manually.

### Installing

Installation is as easy as running the setup script and then installing using pip:
  ```bash
  python setup.py sdist --format=tar
  pip install dist/mpctools-0.2.4.tar --user
  ```
 The `--user` flag is optional and only necessary when one does not have full system permissions. Note that depending on
 which version the library is at, you may need to change the version number of the install command.
