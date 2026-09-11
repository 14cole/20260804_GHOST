"""Exercise timed traceback dumping during short-lived standard-library threads."""
import ctypes
import faulthandler
from pathlib import Path
import threading
import time
import sys

ctypes.windll.kernel32.SetErrorMode(3)
timed = '--timed' in sys.argv
with (Path(__file__).parent/('traceback-timed.log' if timed else 'traceback-control.log')).open('w') as log:
    faulthandler.enable(file=log)
    if timed:
        faulthandler.dump_traceback_later(.005, repeat=True, file=log)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        thread=threading.Thread(target=lambda:threading.Event().wait(.001))
        thread.start()
        thread.join()
    faulthandler.cancel_dump_traceback_later()
    faulthandler.disable()
print('Thread lifecycle stress completed; timed dumps:', timed)
