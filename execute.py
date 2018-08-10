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
args = parser.parse_args()

master_path = args.master_path

#Execute in sequence yourfile.py in each folders under the master folder
def excute(master_path):
    for folder in os.listdir(master_path):
        if folder.startswith("Model_"):
            folder_path = os.path.join(master_path,folder)
            code = subprocess.call(['qsub','submit.pbs'], cwd=folder_path)
            if code != 0:
                print("Have problems with this ",folder)
            
excute(master_path)