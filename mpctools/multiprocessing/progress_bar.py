import sys


class ProgressBar:
    """
    Class for printing a progress bar. The code is adapted from:
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    This is a convenient method for displaying (and keeping track) of progress.
    """

    def __init__(self, total, width=100, sink=sys.stdout, prec=1):
        """
        Initialiser

        :param total:   Total Number of steps involved
        :param width:   Printed Character Width: Default 100
        :param sink:    The output sink: can be None to suppress output.
        :param prec:    The precision to display percentages at: Default 1 d.p.
        """
        # Initaliseables
        self.__total = round(float(total))
        self.__width = round(float(width))
        self.__sink = sink
        self.__prec = prec

        # State Control
        self.__count = int(0)
        self.__prefix = None

    @property
    def Sink(self):
        return self.__sink

    def reset(self, prefix='', suffix=''):
        """
        Convenience Function for starting the Progress bar:

        :param prefix: Prefix to set (if any)
        :param suffix: Suffix to write
        :return: None
        """
        return self.update(value=0, prefix=prefix, suffix=suffix)

    def update(self, update=None, value=None, prefix=None, suffix=''):
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
        if self.__sink is None: return

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
        self.__sink.write('\r{0} |{1}{2}| {3:.{4}f} {5}'.format(self.__prefix,
                                                                '\u2588'*int(_progress),
                                                                '-'*(self.__total - int(_progress)),
                                                                _progress, self.__prec,
                                                                suffix))
        if int(_progress) == self.__total: self.__sink.write('\n')
        self.__sink.flush()

        return self
