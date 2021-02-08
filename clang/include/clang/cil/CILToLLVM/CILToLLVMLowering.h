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
//===----------- CILToLLVMLowering.h - lower CILOps to LLVM
// conversion-------===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CILOPS_LOWERING_H
#define MLIR_TRANSFORMS_CILOPS_LOWERING_H

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#include "clang/cil/dialect/CIL/CILOps.h"

using namespace std;
using namespace mlir;

namespace CIL {
namespace lowering {

class CILTypeConverter : public LLVMTypeConverter {
private:
  LLVM::LLVMType convertArrayType(CIL::ArrayType type);
  LLVM::LLVMType convertStructType(CIL::StructType type);
  LLVM::LLVMType convertIntegerTy(CIL::IntegerTy type);
  LLVM::LLVMType convertFloatingTy(CIL::FloatingTy type);
  LLVM::LLVMType convertCharType(CIL::CharTy type);
  LLVM::LLVMType convertFunctionType(mlir::FunctionType type);
  LLVM::LLVMType convertPointerType(CIL::PointerType type);
  LLVM::LLVMType convertVoidType(CIL::VoidType type);
  // LLVM::LLVMType convertUnsignedIntegerTy(CIL::UnsignedIntegerTy type);

public:
  using LLVMTypeConverter::convertType;
  CILTypeConverter(MLIRContext *ctx);
};

void populateCILFuncOpLoweringPatterns(OwningRewritePatternList &patterns,
                                       CILTypeConverter *typeConverter,
                                       MLIRContext *context);

void populateComplexOpLoweringPatterns(OwningRewritePatternList &patterns,
                                       CILTypeConverter *typeConverter,
                                       MLIRContext *context);
} // namespace lowering
} // namespace CIL
#endif
