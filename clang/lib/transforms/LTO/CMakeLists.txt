add_clang_library(CILLTOTransforms
  RemoveDeadField.cpp
  DCE.cpp
  AoSToSoA.cpp
)

target_link_libraries(CILLTOTransforms
 MLIRLLVMIR
 MLIRStandardOps
 )

install(TARGETS CILLTOTransforms DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
