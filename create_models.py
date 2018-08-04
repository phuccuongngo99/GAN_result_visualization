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

parser = argparse.ArgumentParser()
parser.add_argument('-c','--conf', help='your config.json hyperparameter file')
parser.add_argument('-p','--py', help='your python file to by run')
parser.add_argument('-m','--master_path', help='the master folder to store different models')

args = parser.parse_args()

json_file = args.conf
py_file = args.py
master_path = args.master_path

#Make master_folder
if not os.path.exists(master_path):
    os.makedirs(master_path)

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
def create_folder(master_path,config_dict,py_file,count):
    folder_name = 'Model_' + str(count) + '_'
    for para in config_dict:
        folder_name += para+'_'+str(config_dict[para])+'_'
    folder_name = folder_name[:-1]
    folder_path = os.path.join(master_path,folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path,'config.json'),'w') as outfile:
        json.dump(config_dict, outfile, indent=4)
        
    shutil.copy(py_file,folder_path)   
    
#One main function
def overall(json_file,py_file,master_path):
    config_list = [{}]
    file = open(json_file,'r')
    data = json.loads(file.read())
    for para_name in data:
        para_list = data[para_name]
        config_list = permutation(config_list,para_name,para_list)
    
    count = 0
    for config_dict in config_list:
        create_folder(master_path,config_dict,py_file,count)
        count += 1
    return config_list

overall_out = overall(json_file,py_file,master_path)