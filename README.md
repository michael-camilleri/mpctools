# mpctools
A set of python tools for extending standard (and non-standard) libraries.

## Setting up

### Requirements

This Library has the following dependencies:
  * scikit-multilearn
  * scikit-learn
  * **cudatoolkit**
  * matplotlib
  * **pytorch**
  * pandas
  * **pathos**
  * numpy
  
Most of the above can be automatically installed through the library itself (i.e. pip will attempt to download them).
However, you may wish to install some explicitly: specifically, the libraries listed in bold are not included:
   1. `pathos` is not included as this is only required if using the `parallel` toolbox
   2. `pytorch` and `cudatoolkit` are not included as these are only required if using the `neural` toolbox

### Installing

Installation is as easy as running the setup script and then installing using pip:
  ```bash
  python setup.py sdist --format=tar
  pip install dist/mpctools-0.1.4.tar --user
  ```
 The `--user` flag is optional and only necessary when one does not have full system permissions. Note that depending on
 which version the library is at, you may need to change the version number of the install command.



