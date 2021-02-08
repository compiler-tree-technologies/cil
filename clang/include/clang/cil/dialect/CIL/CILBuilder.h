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
//===- CILOps.h CIL specific Ops ----------------------------------===//

#ifndef MLIR_DIALECT_CIL_OP_BUILDER_H
#define MLIR_DIALECT_CIL_OP_BUILDER_H

#include "clang/cil/dialect/CIL/CILOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"

namespace CIL {
class CILBuilder : public mlir::OpBuilder {
public:
  CILBuilder(MLIRContext *context) : OpBuilder(context) {}

  CILBuilder(mlir::Operation *op) : OpBuilder(op) {}

  bool isCILRefType(mlir::Type type);

  // Get the actual type underneath pointer/array types.
  // It might be scalar/ struct/ class.
  mlir::Type getUnderlyingType(mlir::Type type);

  CIL::PointerType getCILPointerType(mlir::Type eleTy, bool isRef = false);

  CIL::IntegerTy getCILIntType();

  CIL::IntegerTy getCILLongIntType();

  CIL::CharTy getCILUCharType();

  CIL::IntegerTy getCILULongIntType();

  CIL::IntegerTy getCILBoolType();

  CIL::IntegerTy getCILChar8Type();

  CIL::FloatingTy getCILFloatType();

  CIL::FloatingTy getCILDoubleType();

  CIL::ClassType getCILClassType(std::string name);

  mlir::Value getCILIntegralConstant(mlir::Location loc, mlir::Type type,
                                     int64_t value);

  mlir::Value getCILIntegralConstant(mlir::Location loc, int64_t value,
                                     unsigned width);

  mlir::Value getCILBoolConstant(mlir::Location loc, bool value);

  mlir::Value getCILLongConstant(mlir::Location loc, int64_t value);

  mlir::Value getCILIntConstant(mlir::Location loc, int64_t value);

  mlir::Value getCILFloatingConstant(mlir::Location loc, mlir::Type type,
                                     double value);

  unsigned getCILIntOrFloatBitWidth(mlir::Type type);
};
} // namespace CIL

#endif
