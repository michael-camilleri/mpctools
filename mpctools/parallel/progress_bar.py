"""
Implementation of a progress bar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see http://www.gnu.org/licenses/.

Author: Michael P. J. Camilleri
"""
from mpctools.extensions import utils
import time as tm
import sys


class ProgressBar:
    """
    Class for printing a progress bar, while keeping track of progress. The code is adapted from:
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
    """

    def __init__(self, total, width=100, sink=sys.stdout, prec=1, verbose=True):
        """
        Initialiser

        :param total:   Total Number of steps involved
        :param width:   Printed Character Width: Default 100
        :param sink:    The output sink: can be None to suppress output.
        :param prec:    The precision to display percentages at: Default 1 d.p.
        """
        # Initialise Members
        self.__total = round(float(total))
        self.__width = round(float(width))
        self.__sink = sink
        self.__prec = prec
        self.__strt = None
        self.__verb = verbose

        # State Control
        self.__count = int(0)
        self.__prefix = None

    @property
    def Sink(self):
        return self.__sink

    def reset(self, prefix="", suffix=""):
        """
        Convenience Function for starting the Progress bar:

        :param prefix: Prefix to set (if any)
        :param suffix: Suffix to write
        :return: Self, for chaining
        """
        self.__strt = tm.time()
        return self.update(value=0, prefix=prefix, suffix=suffix)

    def update(self, update=None, value=None, prefix=None, suffix=""):
        """
        Update the Progress. By default, this amounts to adding 1 to the count. However, set has precedence over
        updating.

        :param update:  If value is None, this specifies the step to add to count, or 1 if None: otherwise it is ignored
        :param value:   If not None, then set to the specified value (truncated to integer)
        :param prefix:  If not None, update the prefix value
        :param suffix:  Suffix to print (may be empty string)
        :return:        Self (for chaining)
        """
        # Check that there is somewhere to write.
        if self.__sink is not None:

            # Check Update logic
            if update is not None:
                self.__count += int(update)
            else:
                if value is None:
                    self.__count += 1
                else:
                    self.__count = int(value)

            # Check Prefix
            if prefix is not None:
                self.__prefix = prefix

            # Write Out
            _progress = self.__width * self.__count / self.__total
            if self.__verb:
                _elapsed = tm.time() - self.__strt
                _rate = self.__count / _elapsed if _elapsed > 0 else -1
                _rem = (
                    utils.show_time(int((self.__total - self.__count) / _rate))
                    if _rate > 0
                    else "--:--"
                )
                self.__sink.write(
                    "\r{0} |{1}{2}| {3:.{4}f}% {5:.{4}f} it/s ({6}) {7}       ".format(
                        self.__prefix,
                        "\u2588" * int(_progress),
                        "-" * (self.__width - int(_progress)),
                        _progress,
                        self.__prec,
                        _rate,
                        _rem,
                        suffix,
                    )
                )
            else:
                self.__sink.write(
                    "\r{0} |{1}{2}| {3:.{4}f}% {5}".format(
                        self.__prefix,
                        "\u2588" * int(_progress),
                        "-" * (self.__width - int(_progress)),
                        _progress,
                        self.__prec,
                        suffix,
                    )
                )
            if int(_progress) == int(self.__width):
                if self.__verb:
                    self.__sink.write(
                        f" [DONE ({utils.show_time(int(tm.time() - self.__strt))})]\n"
                    )
                else:
                    self.__sink.write(" [DONE]\n")
            self.__sink.flush()

        return self
