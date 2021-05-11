#!/bin/sh
cd ../test_script
export DATASET_DIR="data/"
# Activate the relevant virtual environment:

python test_meta_init.py --name_of_args_json_file ../exp_config/maml_simple_version.json