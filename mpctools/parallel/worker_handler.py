"""
A Set of interfaces for multi-processing in python

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

from mpctools.parallel.progress_bar import ProgressBar
from mpctools.extensions import utils
from pathos.helpers import mp
import numpy as np
import threading
import queue
import time
import abc
import sys


class IWorker(metaclass=abc.ABCMeta):
    """
    This is the Worker Interface. All worker-threads should implement this abstract interface,
    specifically the parallel_compute method.

    IWorker instances should be stateless, with all functionality contained within the
    parallel_compute method. Each worker is managed by the WorkManager (below), which controls
    instantiation, execution and progress control. The workers communicate with the Manager through
    a queue (to which each IWorker has access).
    """

    def __init__(self, _id, _mgr):
        """
        Initialiser

        Note that the implementation Constructor should follow the same template and not take
        additional arguments: these must be passed to the parallel_compute method if required.

        :param _id:  A Unique Identifier: This will be greater than 0! This is typically used not
                     only to identify the different workers, but could also be used to help ensure a
                     consistent but unique seed to each of the workers.
        :param _mgr: An instance of a Worker Handler for getting the queue from.

        """
        self.ID = _id
        self.__queue = _mgr.Queue  # Has to be a strong reference (weakref loses object)
        _mgr._register_worker(_id)

    def update_progress(self, progress):
        """
        This method should be called (regularly) to indicate progress updates by the worker-thread.

        Internally, it uses the manager queue to update it of progress. Each message consists of two
        parts:
            a) ID: this is automatically assigned (and managed) by the Manager, and is a number
                   greater than 0 (which is reserved for the Manager itself.
            b) Progress: A number between 0 and 100 indicating the current (relative) progress.

        Note, that due to some bug in the MultiProcessing Library, which seems to sometimes reset
        the connection, I had to wrap this in a try-catch block

        :param progress: A number between 0 and 100 indicating the *CURRENT* progress as a
               percentage
        :return: None
        """
        try:
            self.__queue.put((self.ID, progress), block=False)
        except TypeError:
            sys.stderr.write("Warn: TypeError Encountered")

    @abc.abstractmethod
    def parallel_compute(self, _common, _data):
        """
        Parallel Computation to be performed

        Must be implemented by respective classes.
        :param _common: Any extensions data, across all workers
        :param _data:   The Data to utilise for each individual computation
        :return:        Any results **Must be a Tuple, at least of size 1**
        """
        raise NotImplementedError()


class WorkerHandler(metaclass=abc.ABCMeta):
    """
    This Class Implements an interface for controlling multiple workers. It is itself based on the
    parallel library from pathos, however, the class adds the following functionality:
        a) Capability of switching between parallel and multi-threading seamlessly, with a
           consistent interface
        b) Specific Worker-Server Model, including management of resources and progress tracking.
        c) (Crude) Timing of individual tasks for performance tracking.

    Technicalities:
      * Communication is done via a queue interface.
      * When multi-threading is enabled (as opposed to parallel) the threads are run one at a time
        (there is no inter-leaving). This can provide a better level of debugging.
    """

    HANDLER_ID = 0

    # =================================== Abstract Interface =================================== #

    @abc.abstractmethod
    def _aggregate_results(self, results):
        """
        Must be implemented to aggregate results from the Worker Pool

        :param results: List of Results (one from each worker)
        :return:       Aggregated Result Data
        """
        raise NotImplementedError()

    # ===================================== Implementation ===================================== #

    @property
    def Queue(self):
        return self._queue

    def __init__(self, num_proc, sink=sys.stdout):
        """
        Initialiser

        :param num_proc: Number of processes to employ (default to the number of cores less 2). If 0
                         or less, then defaults to Multi-Threading instead of Multi-Processing: this
                         can be especially useful for debugging.
        :param sink:     Sink where to write progress to (may be None)
        """
        # Parameters
        self.NumProc = num_proc  # Number of Processes to employ

        # Management
        self._queue = mp.Manager().Queue() if num_proc > 0 else queue.Queue()  # Queue
        self.__timers = {}  # Timers
        self.__thread = None  # Progress Thread Handler
        self.__worker_set = None  # List of Workers and progress
        self.__done = 0  # How many are finished
        self.__tasks_done = 0  # How many (workers) are finished
        self.__progress = None  # Eventually will be the progress bar
        self.__sink = utils.NullableSink(sink)  # Sink to write to

    def _reset(self, num_workers):
        self.__thread = threading.Thread(target=self.__handler_thread)
        self.__worker_set = np.zeros(num_workers)  # Set of Workers (for progress)
        self.__tasks_done = 0  # How many (workers) are finished

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

    def run_workers(self, _num_work, _type, _configs, _args):
        """
        Starts the Pool of Workers and executes them.

        The method blocks until all workers have completed. However, it also starts a background
        update-thread which publishes information about progress.

        :param _num_work:   Number of workers to initialise
        :param _type:       The worker type to run
        :param _configs:    These are extensions across all workers: may be None
        :param _args:       These are arguments per-worker. Must be a list equal in length to
                            _num_work or None
        :return:            Result of the Aggregator
        """
        # Reset Everything
        self._reset(_num_work)
        _args = _args if _args is not None else [None for _ in range(_num_work)]

        # Prepare the Progress Bar: will automatically handle None
        self.__progress = ProgressBar(100 * _num_work, sink=self.__sink.Obj)

        # Create List of Worker Objects, and initialise thread
        _workers = [_type(_i + 1, self) for _i in range(_num_work)]
        self.__thread.start()

        # Start Pool and aggregate results
        if self.NumProc > 0:
            with mp.Pool(processes=self.NumProc) as pool:
                processes = [
                    pool.apply_async(
                        self.__computer, args=(_workers[_i], (_configs, _args[_i]))
                    )
                    for _i in range(_num_work)
                ]
                aggregated = self._aggregate_results(
                    [result.get() for result in processes]
                )
        else:
            r_q = queue.Queue()
            threads = [
                threading.Thread(
                    target=self.__threader,
                    args=(_workers[_i], (_configs, _args[_i]), r_q),
                )
                for _i in range(_num_work)
            ]
            for thr in threads:
                thr.start()
                thr.join()
            results = []
            while not r_q.empty():
                results.append(r_q.get())
                r_q.task_done()
            aggregated = self._aggregate_results(results)

        # Inform and join thread
        self.Queue.put([0, -1])
        self.__thread.join()

        # Delete the Workers explicitly just in case - this prevents the circular referencing from
        # remaining.
        _workers.clear()

        # Return the aggregated information
        return aggregated

    @staticmethod
    def __computer(worker, arguments):
        return worker.parallel_compute(arguments[0], arguments[1])

    @staticmethod
    def __threader(worker, arguments, _queue):
        _queue.put(worker.parallel_compute(arguments[0], arguments[1]))

    def _register_worker(self, wrk_id):
        """
        Used by the Worker Class to register its' ID
        :param wrk_id:
        :return:
        """
        assert wrk_id > 0, "Worker ID must be greater than 0"
        self.__worker_set[wrk_id - 1] = 0.0

    def _write(self, *args):
        self.__sink.write(*args)

    def _flush(self):
        self.__sink.flush()

    def _print(self, *args):
        self.__sink.write(*args)
        self.__sink.write("\n")
        self.__sink.flush()

    def __handler_thread(self):
        # Flag for Stopping the function
        _continue = True

        # Print First one
        self.__progress.reset(
            prefix=(
                "Working [{0} Proc]".format(self.NumProc)
                if self.NumProc > 0
                else "Working [Multi-Thread]"
            ),
            suffix=(
                "Completed {0}/{1} Tasks".format(self.__done, len(self.__worker_set))
            ),
        )

        # This provides an infinite loop until signalled to stop
        while _continue:
            # Handle Message
            _msg = self.Queue.get()
            if _msg[0] > self.HANDLER_ID:
                # Update Counts (and done count if this is the first time)
                if _msg[1] == 100.0 and self.__worker_set[_msg[0] - 1] < 100:
                    self.__done += 1
                self.__worker_set[_msg[0] - 1] = _msg[1]
                # Print Progress
                self.__progress.update(
                    value=self.__worker_set.sum(),
                    suffix="Completed {0}/{1} Tasks".format(
                        self.__done, len(self.__worker_set)
                    ),
                )
            else:
                _continue = False
                if self.__progress.Sink is not None:
                    self.__progress.Sink.write(
                        "Done: Completed All Tasks\n"
                        if self.__done == len(self.__worker_set)
                        else "\nStopped after {0}/{1} Tasks\n".format(
                            self.__done, len(self.__worker_set)
                        )
                    )

            # Indicate Task Done
            self.Queue.task_done()
