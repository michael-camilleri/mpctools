# mpctools
A set of python tools for extending standard (and non-standard) libraries. These originated from
 my own needs and those of my students, and I decided to put them here in case they may be useful
 to other people.

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
  * scikit-learn
  * matplotlib
  * seaborn
  * pandas
  * pathos
  * scipy
  * numpy
  
In most cases, the above can be automatically installed through the library itself (i.e. pip will attempt to download 
them). If this causes issues, just install them manually.

There is an additional requirement for opencv: however, this is not included in the list of requirements for the reason that
some people may wish to build it from source. This is required for example if one wishes to use non open-source encoders: in this
case, I have provided a blog-post about how to do this on my 
[webpage](https://michaelpjcamilleri.wordpress.com/2019/03/21/installing-opencv-with-all-the-bells-and-whistles/).
Otherwise, you can either chose to ignore it if you are not going to use the CV extensions module (cvext),
or install the stock open-cv wrapper for python:
```bash
pip install opencv-python
```

### Installing

The project is available on [PyPi](https://pypi.org/project/mpctools/0.4.0/), and hence the
 latest (stable) release can be installed simply:
  ```shell script
  pip install mpctools
  ```
Alternatively, you may choose to install directly from source. This has the added advantage that if 
you change any of the implementations, the changes will be reflected without having to rebuild. 
However, you will have to manually download the source (via git or just zipped and then extracted):
  ```shell script
  python setup.py build develop --no-deps
  ```

### Known Issues

 * **Python 3.7: parallel.IWorker** - There seems to be an incompatibility in pathos with python 3.7, which is causing
 it to  default to pickle rather than dill, and sometimes preventing abc-derived classes (namely the IWorker instance)
 from  being pickled. If this happens to you, just make your worker a standard class and copy the initialiser and 
 `update_progress` methods from IWorker. We are working on a solution to this.
 * **parallel Blocking** - If the program seems to hang for no reason, it could be that one of the child processes died
 maybe due  to a memory overlow... if this happens, try to limit the amount of memory usage by each IWorker.