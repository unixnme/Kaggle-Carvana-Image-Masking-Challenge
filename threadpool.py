from threading import Thread
from Queue import Queue

class Worker(Thread):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.tasks_queue = queue
        self.daemon = True

    def run(self):
        while True:
            f, arg, callback = self.tasks_queue.get()
            result = f(arg)
            if callback is not None:
                callback(result)
            self.tasks_queue.task_done()

class ThreadPool(object):
    def __init__(self, num_threads):
        self.tasks_queue = Queue()
        for _ in range(num_threads):
            Worker(self.tasks_queue).start()

    def add_task(self, f, arg, callback=None):
        self.tasks_queue.put((f, arg, callback))

    def wait_complete(self):
        self.tasks_queue.join()

if __name__ == '__main__':
    from time import sleep

    pool = ThreadPool(4)
    def task(t):
        sleep(t)
        return t

    def callback(t):
        print t

    for i in reversed(range(10)):
        pool.add_task(task, i, callback)

    pool.wait_complete()