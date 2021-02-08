set(LLVM_TARGET_DEFINITIONS CILOps.td)
mlir_tablegen(CILOps.h.inc -gen-op-decls)
mlir_tablegen(CILOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCILOpsIncGen)
