import sys
import tensorflow

"""
Configuration
"""
prefix = '@' # Command prefix
values = { 
    'example': {
        'description': 'Example command',
        'usage': f'{prefix}example <int>',
        'function': 'thing.example(int)'
    }
}

"""
Define functionality of each value within the dictionary, use this format
"""
class thing:
    def example(int):
        print(f"You chose {int}")

# Make sure that you define at least two functions per class!

