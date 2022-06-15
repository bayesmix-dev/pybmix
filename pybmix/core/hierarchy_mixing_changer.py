# Internal function to change the name of the hierarchy I want to use
import os


def change_hierarchy(new_name):
    os.environ['HIER_NAME'] = new_name
