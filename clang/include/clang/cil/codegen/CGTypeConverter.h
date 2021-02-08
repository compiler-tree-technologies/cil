
#ifndef MLIR_CIL_CODEGEN_TYPES_H
#define MLIR_CIL_CODEGEN_TYPES_H

#include "clang/AST/AST.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"
#include "clang/cil/mangler/CILMangle.h"

namespace clang {
namespace mlir_codegen {

class CodeGenTypes {
public:
  CodeGenTypes(mlir::MLIRContext &mlirContext, CIL::CILBuilder &builder,
               clang::ASTContext &context);

  mlir::FunctionType convertFunctionType(const clang::FunctionType *FT);
  mlir::FunctionType convertFunctionProtoType(const FunctionProtoType *FT);
  mlir::Type convertClangType(const clang::Type *type);
  mlir::Type convertClangType(const clang::QualType type);
  mlir::Type convertBuiltinType(const clang::BuiltinType *BT);

private:
  mlir::MLIRContext &mlirContext;
  CIL::CILBuilder &builder;
  clang::ASTContext &context;
  CIL::CILMangle mangler;

  llvm::DenseMap<const clang::Type *, CIL::StructType> RecordDeclTypes;
};
} // namespace mlir_codegen
} // namespace clang

#endif