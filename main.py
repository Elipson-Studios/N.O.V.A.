import sys
import tensorflow
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    prefix = config.get('prefix', '')

values = { 
    'example': {
        'description': 'Example command',
        'usage': f'{prefix}example <int>',
        'function': 'thing.example(int)'
    }
}

"""
Function Definition
"""
class thing:
    def example(int):
        print(f"You chose {int}")

# Make sure that you define at least two functions per class!
