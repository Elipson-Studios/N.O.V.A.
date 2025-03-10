# N.O.V.A. documentation

N.O.V.A. operates using a set of predefined commands. Each command is associated with a function and is stored as an entry in a dictionary. Commands look like this:

'!call[command(arg1, arg2)]'

The prefix and caller can always be changed based on the call in 'config.json'

## Ensuring scalability
We prefer avoiding any external modules where possible.