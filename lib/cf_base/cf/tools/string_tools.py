import re
from datetime import datetime

# unique string based on current time
def now_string():
   return datetime.utcnow().strftime("%Y.%m.%d_%H.%M.%S.%f")[:-3]

# print a string to the console into the same line
def print_same_line(v):
    print("\r"+str(v), end="")