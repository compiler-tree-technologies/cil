add_clang_library(MLIRCILOps
  CILOps.cpp
  DialectRegistration.cpp
  CILTypes.cpp
  CILBuilder.cpp
  )
add_dependencies(MLIRCILOps MLIRCILOpsIncGen  MLIRIR MLIRStandardOps)
target_link_libraries(MLIRCILOps MLIRIR MLIRStandardOps)

install (TARGETS MLIRCILOps
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)

