#include <Python.h> // Waiting patiently for github to fix this
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX 100

char root[MAX] = "N.O.V.A."; // This means that the folder name MUST be N.O.V.A. or it will not locate!
char main_script[MAX] = "main.py";
char modules[MAX] = "Depend/interpret/ed/modules";

int main(int argc, char *argv[]) {
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    Py_SetProgramName(program);  // optional but recommended
    Py_Initialize();

    // Run the main script
    char main_script_path[MAX * 2];
    snprintf(main_script_path, sizeof(main_script_path), "%s/%s", root, main_script);
    FILE* fp = fopen(main_script_path, "r");
    if (fp != NULL) {
        PyRun_SimpleFile(fp, main_script_path);
        fclose(fp);
    } else {
        fprintf(stderr, "Could not open main script: %s\n", main_script_path);
    }

    // Run the modules
    char modules_path[MAX * 2];
    snprintf(modules_path, sizeof(modules_path), "%s/%s", root, modules);
    FILE* fp_modules = fopen(modules_path, "r");
    if (fp_modules != NULL) {
        PyRun_SimpleFile(fp_modules, modules_path);
        fclose(fp_modules);
    } else {
        fprintf(stderr, "Could not open modules: %s\n", modules_path);
    }

    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}