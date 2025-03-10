# Define the compiler
CC = gcc

# Define the source file and the output executable
SRC = ../*.c
TARGET = program

# Check the OS and set the executable extension
ifeq ($(OS),Windows_NT)
    EXE_EXT = .exe
else
    EXE_EXT =
endif

# Define the build rule
all: $(TARGET)$(EXE_EXT)

$(TARGET)$(EXE_EXT): $(SRC)
    $(CC) -o $(TARGET)$(EXE_EXT) $(SRC)

# Define the clean rule
clean:
    rm -f $(TARGET)$(EXE_EXT)