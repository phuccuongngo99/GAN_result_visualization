#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:49:13 2018

@author: root
"""
import json
import os
import shutil
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-c','--conf', help='your config.json hyperparameter file')
parser.add_argument('-p','--py', help='your python folder to run')
parser.add_argument('-m','--master_path', help='the master folder to store different models')
parser.add_argument('-s','--script', help='your main python script')

args = parser.parse_args()

json_file = args.conf
py_folder = args.py
master_path = args.master_path
py_file = args.script

cwd = os.getcwd()
#Make master_folder

if not os.path.exists(master_path):
    os.makedirs(master_path)

shutil.copy(json_file,master_path)
subprocess.call(['cp', '-r', py_folder, master_path]) # Linux

def pbs(folder_path,py_file):
    a = \
    '#!/bin/bash' + '\n' + \
    '#PBS -q gpu' + '\n' + \
    '#PBS -l select=1:ncpus=8:mem=8G:ngpus=1' + '\n' + \
    '#PBS -l walltime=01:00:00' + '\n' + \
    '#PBS -P Personal' + '\n' + \
    '#PBS -N Lennard_Jones.py' + '\n' + \
    'cd ' + '$PBS_O_WORKDIR' + '\n' + \
    'module load tensorflow/1.0+keras' + '\n' + \
    'python ' + str(py_file) +' -c config.json' + '\n' + \
    '# Output of R is generated to simple.Rout'
    
    with open(os.path.join(folder_path,'submit.pbs'),'w+') as file:
        file.write(a)
    file.close()

#Create all different combination of hyperparameters
#Return a list of dictionaries of hyperparameters
def permutation(config_list,para_name,para_list):
    new_config_list = []
    for config in config_list:
        for para in para_list:
            new_config = config.copy()
            new_config.update({para_name:para})
            new_config_list.append(new_config)
            
    return new_config_list

#Create different folders for different hyperparameters configurations
#Each folder will have config.json and yourfile.py
def create_folder(master_path,config_dict,py_folder,py_file,count):
    folder_name = 'Model_' + str(count) + '_'
    for para in config_dict:
        folder_name += para+'_'+str(config_dict[para])+'_'
    folder_name = folder_name[:-1]
    folder_path = os.path.join(master_path,folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path,'config.json'),'w') as outfile:
        json.dump(config_dict, outfile, indent=4)
    
    for python_file in os.listdir(py_folder):
        shutil.copy(os.path.join(py_folder,python_file),folder_path)
    
    pbs(folder_path,py_file)
    
#One main function
def overall(json_file,py_folder,master_path,py_file):
    config_list = [{}]
    file = open(json_file,'r')
    data = json.loads(file.read())
    for para_name in data:
        para_list = data[para_name]
        config_list = permutation(config_list,para_name,para_list)
    
    count = 0
    for config_dict in config_list:
        create_folder(master_path,config_dict,py_folder,py_file,count)
        count += 1
    return config_list

overall_out = overall(json_file,py_folder,master_path,py_file)