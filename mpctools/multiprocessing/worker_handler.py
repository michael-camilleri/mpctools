from pathos.helpers import mp
import numpy as np
import threading
import weakref
import queue
import time
import abc
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
        return self.update(set=0, prefix=prefix, suffix=suffix)

    def update(self, update=None, set=None, prefix=None, suffix=''):
        """
        Update the Progress. By default, this amounts to adding 1 to the count. However, set has precedence over
        updating.

        :param update:  If set is None, this specifies the step to add to count: if None, add 1
        :param set:     If not None, then set to the specified value (truncated to integer)
        :param prefix:  If not None, update the prefix value
        :param suffix:  Suffix to print (may be empty string)
        :return:        None
        """
        # Check that there is somewhere to write.
        if self.__sink is None: return

        # Check Update logic
        if update is not None:
            self.__count += int(update)
        else:
            if set is None:
                self.__count += 1
            else:
                self.__count = int(set)

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

        return


class IWorker(metaclass=abc.ABCMeta):
    """
    This is the Worker Interface. All worker-threads should implement this abstract interface, specifically
    the parallel_compute method.

    IWorker instances should be stateless, with all functionality contained within the parallel_compute method. Each
    worker is managed by the WorkManager (below), which controls instantiation, execution and progress control. The
    workers communicate with the Manager through a queue (to which each IWorker has access).
    """

    def __init__(self, _id, _mgr):
        """
        Initialiser

        Note that the implementation Constructor should follow the same template and not take additional arguments:
        these must be passed to the parallel_compute method if required.

        :param _id:  A Unique Identifier: This must be greater than 0!
        :param _mgr: An instance of a Worker Handler for getting the queue from.

        """
        self.ID = _id
        self.__queue = weakref.ref(_mgr.Queue)   # Keep Weak-Reference to Handler Class' Queue
        _mgr._register_worker(id)

    def update_progress(self, progress):
        """
        This method should be called (regularly) to indicate progress updates by the worker-thread.

        Internally, it uses the manager queue to update it of progress. Each message consists of two parts:
            a) ID: this is automatically assigned (and managed) by the Manager, and is a number greater than 0 (which is
                   reserved for the Manager itself.
            b) Progress: A number between 0 and 100 indicating the current (relative) progress.

        Note, that due to some bug in the MultiProcessing Library, which seems to sometimes reset the connection,
         I had to wrap this in a try-catch block

        :param progress: A number between 0 and 100 indicating the *CURRENT* progress as a percentage
        :return: None
        """
        try:
            self.__queue().put((self.ID, progress), block=False)
        except TypeError:
            sys.stderr.write('Warn: TypeError Encountered')
            pass

    @abc.abstractmethod
    def parallel_compute(self, _common, _data):
        """
        Parallel Computation to be performed

        Must be implemented by respective classes.
        :param _common: Any common data, across all workers
        :param _data:   The Data to utilise for each individual computation
        :return:        Any results **Must be a Tuple, at least of size 1**
        """
        raise NotImplementedError()


class WorkerHandler(metaclass=abc.ABCMeta):
    """
    This Class Implements an interface for controlling multiple workers. It is itself based on the multiprocessing
    library from pathos, however, the class adds the following functionality:
        a) Capability of switching between multiprocessing and multi-threading seamlessly, with a consistent interface
        b) Specific Worker-Server Model, including management of resources and progress tracking.
        c) (Crude) Timing of individual tasks for performance tracking.

    Technicalities:
      * Communication is done via a queue interface.
      * When multi-threading is enabled (as opposed to multiprocessing) the threads are run one at a time (there is no
            inter-leaving.
    """
    HANDLER_ID = 0

    # ======================================== Internal Interfaces ======================================== #



    # ========================================= Abstract Interface ========================================= #

    @abc.abstractmethod
    def aggregate_results(self, results):
        """
        Must be implemented to aggregate results from the Worker Pool

        :param results: List of Results (one from each worker)
        :return:       Aggregated Result Data
        """
        raise NotImplementedError()

    # =========================================== Implementation =========================================== #

    @property
    def Queue(self):
        return self._queue

    def __init__(self, num_proc):
        """
        Initialiser

        :param num_proc:    Number of processes to employ (default to the number of cores less 2). If 0 or less, then
                            defaults to Multi-Threading instead of Multi-Processing: this can be especially useful for
                            debugging.
        """
        # Parameters
        self.NumProc = num_proc                 # Number of Processes to employ

        # Management
        self._queue = mp.Manager().Queue() if num_proc > 0 else queue.Queue()   # Queue
        self.__timers = {}                                                      # Timers
        self.__thread = None                                                    # Progress Thread Handler
        self.__worker_set = None                                                # List of Workers and progress
        self.__done = 0                             # How many are finished
        self.__tasks_done = 0                       # How many (workers) are finished
        self.__progress = None                      # Eventually will be the progress bar

    def _reset(self, num_workers):
        self.__thread = threading.Thread(target=self.__handler_thread)
        self.__worker_set = np.zeros(num_workers)  # Set of Workers and associated progress
        self.__tasks_done = 0                      # How many (workers) are finished

    def start_timer(self, name):
        """
        Starts a timer running on name: if exists, resets

        :param name: Name of the timer to assign
        :return:     None
        """
        self.__timers[name] = [time.time(), None]

    def stop_timer(self, name):
        """
        Stops a timer: If name does not exist, raises an exception

        :param name: Timer to stop
        :return:     Elapsed time
        """
        self.__timers[name][1] = time.time()
        return self.elapsed(name)

    def elapsed(self, name):
        """
        Returns the elapsed time (after having stopped)

        :param name: Name of timer: must exist
        :return: Elapsed time
        """
        return self.__timers[name][1] - self.__timers[name][0]

    def run_workers(self, _num_work, _type, _configs, _args, _sink=sys.stdout):
        """
        Starts the Pool of Workers and executes them.

        The method blocks until all workers have completed. However, it also starts a background update-thread which
        publishes information about progress.

        :param _num_work:   Number of workers to initialise
        :param _type:       The worker type to run
        :param _configs:    These are common across all workers: may be None
        :param _args:       These are arguments per-worker. Must be a list equal in length to _num_work or None
        :param sink:        Sink where to write progress to (may be None)
        :return:            Result of the Aggregator
        """
        # Reset Everything
        self._reset(_num_work)
        _args = _args if _args is not None else [None for _ in range(_num_work)]

        # Create List of Worker Objects, and initialise thread
        _workers = [_type(_i+1, self) for _i in range(_num_work)]
        self.__thread.start()

        # Start Pool and aggregate results
        if self.NumProc > 0:
            with mp.Pool(processes=self.NumProc) as pool:
                processes = [pool.apply_async(self.__computer, args=(_workers[_i], (_configs, _args[_i]))) for _i in range(_num_work)]
                aggregated = self.aggregate_results([result.get() for result in processes])
        else:
            r_q = queue.Queue()
            threads = [threading.Thread(target=self.__threader, args=(_workers[_i], (_configs, _args[_i]), r_q)) for _i in range(_num_work)]
            for thr in threads: thr.start(); thr.join()
            results = []
            while not r_q.empty():
                results.append(r_q.get())
                r_q.task_done()
            aggregated = self.aggregate_results(results)

        # Inform and join thread
        self.Queue.put([0, -1])
        self.__thread.join()

        # Prepare the Progress Bar if not None
        if _sink is not None:
            self.__progress = ProgressBar(100 * _num_work, sink=_sink)
        else:
            self.__progress = None

        # Return the aggregated information
        return aggregated

    def __computer(self, worker, arguments):
        return worker.parallel_compute(arguments[0], arguments[1])

    def __threader(self, worker, arguments, _queue):
        _queue.put(worker.parallel_compute(arguments[0], arguments[1]))

    def _register_worker(self, wrk_id):
        """
        Used by the Worker Class to register its' ID
        :param wrk_id:
        :return:
        """
        assert wrk_id > 0, 'Worker ID must be greater than 0'
        self.__worker_set[wrk_id-1] = 0.0

    def __handler_thread(self):
        # Flag for Stopping the function
        _continue = True

        # Print First one
        self.__progress.reset(prefix=('Working [{0} Proc]'.format(self.NumProc) if self.NumProc > 0 else
                                      'Working [Multi-Thread]'),
                              suffix=('Completed {0}/{1} Tasks'.format(self.__done, len(self.__worker_set))))

        # This provides an infinite loop until signalled to stop
        while _continue:
            # Handle Message
            _msg = self.Queue.get()
            if _msg[0] > self.HANDLER_ID:
                # Update Counts (and done count if this is the first time)
                if _msg[1] == 100.0 and self.__worker_set[_msg[0]-1] < 100: self.__done += 1
                self.__worker_set[_msg[0]-1] = _msg[1]
                # Print Progress
                self.__progress.update(set=self.__worker_set.sum(),
                                       suffix='Completed {0}/{1} Tasks'.format(self.__done, len(self.__worker_set)))
            else:
                _continue = False
                self.Sink.write('Done: Completed All Tasks\n' if self.__done == len(self.__worker_set) else
                                    '\nStopped after {0}/{1} Tasks\n'.format(self.__done, len(self.__worker_set)))

            # Indicate Task Done
            self.Queue.task_done()
