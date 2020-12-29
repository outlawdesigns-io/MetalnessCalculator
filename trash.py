#import os.path
from os import path
user_input = '/tmp/romeo'
if path.exists(str(user_input)):
    print("The path exists")
else:
    print("No path")
