
set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  TransformUtils
  Vectorize
  )

add_clang_tool(cml
    cml.cpp
)

set(LIBS
  MLIRStandardOps
  MLIRAffineOps
  MLIRLLVMIR
  MLIRLoopOps
)

clang_target_link_libraries(cml
  PRIVATE
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangSerialization
    MLIRTranslation
    LLVMSupport
    MLIRAnalysis
    MLIRLLVMIR
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRTransforms
    MLIRTargetLLVMIR
    MLIRIR
    CMLIRCodeGen
    CILToLLVM
    MLIRCILOps
    CPPLowering
    CILLTOTransforms
    STLPasses
    MLIRStandardToLLVM
    ${LIBS}
  )


whole_archive_link(cml ${LIBS})

