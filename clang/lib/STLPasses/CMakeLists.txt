add_clang_library(STLPasses
  VectorPushBackOptimizer.cpp	
  VectorReserveOptimizer.cpp	
)

target_link_libraries(STLPasses
    MLIRAffineOps
    MLIRLLVMIR
    MLIRStandardOps
    MLIRLoopOps
    MLIRCILOps
 )

install(TARGETS STLPasses DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


