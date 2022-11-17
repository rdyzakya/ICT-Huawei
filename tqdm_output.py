from time import sleep

import contextlib
import sys

from tqdm import tqdm

class DummyFile(object):
  file = None
  def __init__(self, file):
    self.file = file

  def write(self, x):
    tqdm.write(x, file=self.file)

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(save_stdout)
    yield
    sys.stdout = save_stdout