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
//===- CILToLLVMLowering.cpp - Loop.for to affine.for conversion
//-----------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of mos to CIL dialect operations to LLVM
// dialect
//
// NOTE: This code contains lot of repeated code. Needs lot of cleanup/
// refactoring.
// TODO: Split patterns into multiple files liek ArrayOps patterns, etc..
//===----------------------------------------------------------------------===//

#include "clang/cil/CILToLLVM/CILToLLVMLowering.h"
#include "clang/cil/dialect/CIL/CILOps.h"

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

#define PASS_NAME "CILToLLVMLowering"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace mlir;
using namespace CIL;
using namespace CIL::lowering;

// TODO Should this be part of some class ?
llvm::DenseMap<mlir::Type, LLVM::LLVMType> StructDeclTypes;

CILTypeConverter::CILTypeConverter(MLIRContext *ctx) : LLVMTypeConverter(ctx) {
  addConversion([&](CIL::IntegerTy type) { return convertIntegerTy(type); });
  addConversion([&](CIL::FloatingTy type) { return convertFloatingTy(type); });
  addConversion(
      [&](CIL::PointerType type) { return convertPointerType(type); });
  addConversion([&](CIL::CharTy type) { return convertCharType(type); });
  addConversion(
      [&](mlir::FunctionType type) { return convertFunctionType(type); });
  addConversion([&](CIL::ArrayType type) { return convertArrayType(type); });
  addConversion([&](CIL::StructType type) { return convertStructType(type); });
  addConversion([&](CIL::VoidType type) { return convertVoidType(type); });
  // addConversion([&](CIL::UnsignedIntegerTy type) {
  //   return convertUnsignedIntegerTy(type);
  // });
}

LLVM::LLVMType CILTypeConverter::convertCharType(CIL::CharTy type) {
  return LLVM::LLVMType::getInt8Ty(llvmDialect);
}

LLVM::LLVMType CILTypeConverter::convertFunctionType(mlir::FunctionType type) {
  SmallVector<LLVM::LLVMType, 2> inputs;
  bool hasVarArg = false;
  for (auto type : type.getInputs()) {
    if (type.isa<CIL::VarArgTy>()) {
      hasVarArg = true;
      continue;
    }
    inputs.push_back(convertType(type).cast<LLVM::LLVMType>());
  }
  LLVM::LLVMType result = LLVM::LLVMType::getVoidTy(llvmDialect);
  if (type.getNumResults() > 0) {
    assert(type.getNumResults() < 2);
    result = convertType(type.getResult(0)).cast<LLVM::LLVMType>();
  }
  return LLVM::LLVMType::getFunctionTy(result, inputs, hasVarArg);
}

LLVM::LLVMType CILTypeConverter::convertArrayType(CIL::ArrayType type) {
  auto llTy = convertType(type.getEleTy()).cast<LLVM::LLVMType>();
  assert(type.hasStaticShape());
  return LLVM::LLVMType::getArrayTy(llTy, type.getShape().front());
}

LLVM::LLVMType CILTypeConverter::convertVoidType(CIL::VoidType type) {
  return LLVM::LLVMType::getVoidTy(llvmDialect);
}

LLVM::LLVMType CILTypeConverter::convertIntegerTy(CIL::IntegerTy type) {
  switch (type.getIntegerKind()) {
  case BoolKind:
    return LLVM::LLVMType::getInt1Ty(llvmDialect);
  case Char8Kind:
    return LLVM::LLVMType::getInt8Ty(llvmDialect);
  case ShortKind:
    return LLVM::LLVMType::getInt16Ty(llvmDialect);
  case IntKind:
    return LLVM::LLVMType::getInt32Ty(llvmDialect);
  case LongKind:
    return LLVM::LLVMType::getInt64Ty(llvmDialect);
  case LongLongKind:
    return LLVM::LLVMType::getInt64Ty(llvmDialect);
  }
  return {};
}

LLVM::LLVMType CILTypeConverter::convertFloatingTy(CIL::FloatingTy type) {
  switch (type.getFloatingKind()) {
  case Float:
    return LLVM::LLVMType::getFloatTy(llvmDialect);
  case Double:
    return LLVM::LLVMType::getDoubleTy(llvmDialect);
  case LongDouble:
    return LLVM::LLVMType::getFP128Ty(llvmDialect);
  default:
    llvm_unreachable("Unhandled float type");
  }
}

LLVM::LLVMType CILTypeConverter::convertPointerType(CIL::PointerType type) {
  auto llTy = convertType(type.getEleTy()).cast<LLVM::LLVMType>();
  return llTy.getPointerTo();
}

LLVM::LLVMType CILTypeConverter::convertStructType(CIL::StructType type) {
  auto eleTypes = type.getElementTypes();
  if (StructDeclTypes.find(type) == StructDeclTypes.end()) {
    StructDeclTypes[type] = LLVM::LLVMType::getOpaqueStructTy(llvmDialect);
  } else {
    return StructDeclTypes[type];
  }

  // TODO  We should not be using llvm type here. We are using because
  //       there is no interface in LLVMType for forward decl.
  llvm::SmallVector<llvm::Type *, 2> actualTypes;
  for (auto eleTy : eleTypes) {
    actualTypes.push_back(
        convertType(eleTy).cast<LLVM::LLVMType>().getUnderlyingType());
  }

  auto currType = StructDeclTypes[type];
  auto structTy =
      llvm::dyn_cast<llvm::StructType>(currType.getUnderlyingType());
  assert(structTy);

  if (type.getName() != "") {
    structTy->setName(type.getName());
  }

  // if opaque type
  if (actualTypes.empty())
    return currType;

  structTy->setBody(actualTypes);
  return currType;
}

// LLVM::LLVMType
// CILTypeConverter::convertUnsignedIntegerTy(CIL::UnsignedIntegerTy type) {
//   switch (type.getIntegerKind()) {
//   case Char8Kind:
//     return LLVM::LLVMType::getInt8Ty(llvmDialect);
//   case ShortKind:
//     return LLVM::LLVMType::getInt16Ty(llvmDialect);
//   case IntKind:
//     return LLVM::LLVMType::getInt32Ty(llvmDialect);
//   case LongKind:
//     return LLVM::LLVMType::getInt64Ty(llvmDialect);
//   case LongLongKind:
//     return LLVM::LLVMType::getInt64Ty(llvmDialect);
//   }
//   return {};
// }
