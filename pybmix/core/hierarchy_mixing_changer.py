# Internal function to change the name of the hierarchy I want to use
import os


def change_hierarchy(new_name):
    os.environ['HIER_C_NAME'] = new_name


def change_non_conjugate_hierarchy(new_name):
    os.environ['HIER_NC_NAME'] = new_name
