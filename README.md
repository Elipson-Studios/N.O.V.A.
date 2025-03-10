# N.O.V.A. Documentation

## Overview
N.O.V.A. operates using a set of predefined commands. Each command is associated with a function and is stored as an entry in a dictionary. This system allows for flexible command execution and easy scalability.

## Command Syntax
Commands follow a specific format:

```
!call[command(arg1, arg2)]
```

- `!call`: This is the default prefix and caller for commands.
- `command`: The function to be executed.
- `arg1, arg2`: Arguments passed to the function.

## Configuration
The prefix and caller for commands can be customized via the `config.json` file. Modify the relevant fields to change how commands are invoked.

Example `config.json` entry:

```json
{
  "prefix": "!call",
  "caller": "["
}
```

## Ensuring Scalability
To maintain scalability and simplicity:

1. **Minimal External Dependencies**: Avoid external modules whenever possible. This reduces potential conflicts and simplifies maintenance.
2. **Modular Functions**: Keep command functions modular and independent to ensure they can be easily maintained or expanded.
3. **Configurable Settings**: Use configuration files for customizable settings to avoid hardcoding values.
4. **Error Handling**: Implement consistent error handling for smoother debugging and more reliable command execution.

## Example Command Entry

```json
{
    "example": {
        "description": "Example command",
        "usage": "{prefix}example <int>",
        "function": "thing.example(int)"
    }
}
```

```python
def greet(name):
    return f"Hello, {name}!"
```

When invoked with the command:

```
!call[greet("Alice")]
```

The response would be:

```
Hello, Alice!
```

## Best Practices
- **Consistent Naming**: Use clear and consistent naming conventions for commands and functions.
- **Documentation**: Document new commands and configurations thoroughly to assist future development.
- **Testing**: Regularly test commands to ensure they function as expected.

## Conclusion
This approach provides a simple, scalable framework for command execution within N.O.V.A. Customization and expansion are straightforward due to the modular and configurable design.

