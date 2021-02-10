// Copyright (c) 2019, Compiler Tree Technologies Pvt Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


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