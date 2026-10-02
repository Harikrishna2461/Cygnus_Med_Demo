import yaml
import os

def load_config(filepath=''):
    if filepath == '':
        abs_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(abs_dir, 'config.yaml')
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config