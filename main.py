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
# Proper format (You can remove this so long as you remove it from values as you chose)

class thing:
    def example(int):
        print(f"You chose {int}")
    def example2(int):
        print(f"You chose {int} again")