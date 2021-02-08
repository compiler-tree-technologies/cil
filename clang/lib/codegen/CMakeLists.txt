add_clang_library(CMLIRCodeGen
  CodeGen.cpp
  CGTypeConverter.cpp
  CGStmt.cpp
  CGExpr.cpp
  CGDecl.cpp
  CGUtil.cpp
  CGClass.cpp
)

set(LIBS
  MLIRStandardOps
  MLIRAffineOps
  MLIRLLVMIR
  MLIRLoopOps
)

#whole_archive_link(CMLIRCodeGen ${LIBS})

target_link_libraries(CMLIRCodeGen
    clangAST
    clangAnalysis
    clangEdit
    clangParse
    clangBasic
    clangDriver
    clangFrontend
    clangIndex
    clangLex
    clangSema
    clangSerialization
    clangTooling
    clangRewrite
    clangDriver
    MLIRAffineOps
    MLIRLLVMIR
    MLIRStandardOps
    MLIRLoopOps
    MLIRCILOps
    CILMangle
 )

install(TARGETS CMLIRCodeGen DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)


