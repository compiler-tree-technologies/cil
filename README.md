# Build steps
1. -DLLVM_ENABLE_PROJECTS="mlir;clang" in cmake
2. `ninja cml` should build the cml clang tool
3. `ninja check-clang-cml` to run the CIL tests.
4. Check `cml --help` tool for various options
