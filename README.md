# mpctools
A set of python tools for extending standard (and non-standard) libraries.

## Setting up

### Requirements

This Library has the following dependencies:
  * scikit-multilearn
  * **opencv-python**
  * scikit-learn
  * **cudatoolkit**
  * matplotlib
  * **pytorch**
  * pandas
  * **pathos**
  * numpy
  
Most of the above can be automatically installed through the library itself (i.e. pip will attempt to download them).
However, you may wish to install some explicitly: specifically, the libraries listed in bold are not included:
   1. `opencv` is not included in the list of requirements since one may wish to build it from source and allow for proprietery codecs.
   2. `pathos` is not included as this is only required if using the `parallel` toolbox
   2. `pytorch` and `cudatoolkit` are not included as these are only required if using the `neural` toolbox

### Installing

Installation is as easy as running the setup script:
  ```bash
  pip setup.py install --user
  ```
 The `--user` flag is optional and only necessary when one does not have full system permissions.



