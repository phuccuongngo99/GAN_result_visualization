#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:12:32 2018

@author: root
"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser('Run model in sequence')
parser.add_argument('-m','--master_path',help='The folder that contains many model')
parser.add_argument('-p','--py',help='the name python file that you want to run')
args = parser.parse_args()

master_path = args.master_path
py_file = args.py

#Execute in sequence yourfile.py in each folders under the master folder
def excute(master_path,py_file):
    for folder in os.listdir(master_path):
    	if folder.startswith("Model_"):
	        folder_path = os.path.join(master_path,folder)
	        code = subprocess.call(['python', py_file, '-c', 'config.json'], cwd=folder_path)
	        if code != 0:
	            print("Have problems with this ",folder)
            
excute(master_path,py_file)