#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:19:17 2018

@author: root
"""
import argparse
import json
from collections import OrderedDict

parser = argparse.ArgumentParser('Train your GAN')
parser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')
args = parser.parse_args()

config_path = args.conf

config = json.loads(open((config_path),'r').read())

### Set environment variables of hyperparameter you are going to use
# It should be the same as those specified
epoch = config['epoch']
batch_size = config['batch_size']

###
###The main code
###

### Write your output result in dictionary
# Include the name to gif file
result = [("mAP",0.8),("recall",0.6),("gif","result.gif")]

with open('result.json','w') as outfile:
    json.dump(OrderedDict(result), outfile, indent=4)