#!/bin/sh

export DATASET_DIR="data/"
# Activate the relevant virtual environment:
cd ../test_script
python test_meta_feat.py --name_of_args_json_file  ../exp_config/rhg_simple_version.json