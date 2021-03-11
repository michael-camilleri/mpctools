# mpctools
A set of python tools for extending standard (and non-standard) libraries. These originated from
 my own needs and those of my students, and I decided to put them here in case they may be useful
 to other people.

## Features

The library currently contains the following two packages:
 1. `extensions`: A number of extensions to numpy, sklearn, pandas and matplotlib, as well as general-purpose utilities.
 2. `parallel`: A set of tools for wrapping pathos multiprocessing in a simple easy to use interface with multiple
     parallel workers.

More details for each library are provided as doxygen-style comments in the modules.

## Setting up

### Requirements

This Library has the following dependencies:
  * opencv-python
  * scikit-learn
  * matplotlib
  * seaborn
  * pandas
  * pathos
  * scipy
  * numpy
  * numba
  
In most cases, the above can be automatically installed through the library itself (i.e. pip will 
attempt to download them). If this causes issues, or you wish to install specific versions (such
as building `opencv` from source), you can prevent dependency checking by passing the `--no-deps`
flag.

### Installing

The project is available on [PyPi](https://pypi.org/project/mpctools/), and hence the latest
 (stable) release can be installed simply:
  ```shell script
  pip install mpctools [--no-deps]
  ```
Note that the `--no-deps` flag is optional (as described [above](#requirements)).

Alternatively, you may choose to install directly from source. This has the added advantage that if 
you change any of the implementations, the changes will be reflected without having to rebuild. 
However, you will have to manually download the source (via git or just zipped and then extracted):
  ```shell script
  python setup.py build develop [--no-deps]
  ```

### Known Issues

 * **Python 3.7: parallel.IWorker** - There seems to be an incompatibility in pathos with python 3.7, which is causing
 it to  default to pickle rather than dill, and sometimes preventing abc-derived classes (namely the IWorker instance)
 from  being pickled. If this happens to you, just make your worker a standard class and copy the initialiser and 
 `update_progress` methods from IWorker. We are working on a solution to this.
 * **parallel Blocking** - If the program seems to hang for no reason, it could be that one of the child processes died
 maybe due  to a memory overlow... if this happens, try to limit the amount of memory usage by each IWorker.