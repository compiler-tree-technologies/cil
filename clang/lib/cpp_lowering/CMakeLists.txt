add_clang_library(CPPLowering
  ClassLowering.cpp
  ConstructorLowering.cpp
  ClassInheritanceLowering.cpp
  ForLoopLowering.cpp
)

target_link_libraries(CPPLowering
    MLIRAffineOps
    MLIRLLVMIR
    MLIRStandardOps
    MLIRLoopOps
    MLIRCILOps
 )

install(TARGETS CPPLowering DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


