add_clang_library(CILToLLVM
  CILToLLVMLowering.cpp
  CILToLLVMTypeConverter.cpp
)
target_link_libraries(CILToLLVM
    MLIRAffineOps
    MLIRLLVMIR
    MLIRStandardOps
    MLIRLoopOps
    MLIRCILOps
)

install(TARGETS CILToLLVM DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


