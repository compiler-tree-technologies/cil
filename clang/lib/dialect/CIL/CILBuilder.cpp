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
//===- CILOps.cpp - CIL Operations ---------------------------------===//

#include "clang/cil/dialect/CIL/CILBuilder.h"

using namespace mlir;
using namespace CIL;

bool CILBuilder::isCILRefType(mlir::Type type) {
  if (auto ptrTy = type.dyn_cast<CIL::PointerType>()) {
    return ptrTy.isReference();
  }
  return false;
}

CIL::FloatingTy CILBuilder::getCILFloatType() {
  return CIL::FloatingTy::get(CIL::Float, context);
}

CIL::FloatingTy CILBuilder::getCILDoubleType() {
  return CIL::FloatingTy::get(CIL::Double, context);
}

CIL::PointerType CILBuilder::getCILPointerType(mlir::Type eleTy, bool isRef) {
  return CIL::PointerType::get(eleTy, isRef);
}

CIL::IntegerTy CILBuilder::getCILChar8Type() {
  return CIL::IntegerTy::get(CIL::Char8Kind, context);
}

CIL::IntegerTy CILBuilder::getCILIntType() {
  return CIL::IntegerTy::get(CIL::IntKind, context);
}

CIL::IntegerTy CILBuilder::getCILLongIntType() {
  return CIL::IntegerTy::get(CIL::LongKind, context);
}

CIL::IntegerTy CILBuilder::getCILULongIntType() {
  CILQualifiers qual;
  qual.setUnsigned();
  return CIL::IntegerTy::get(CIL::LongKind, qual, context);
}

CIL::CharTy CILBuilder::getCILUCharType() {
  CILQualifiers qual;
  qual.setUnsigned();
  return CIL::CharTy::get(qual, context);
}

CIL::IntegerTy CILBuilder::getCILBoolType() {
  return CIL::IntegerTy::get(CIL::BoolKind, context);
}

CIL::ClassType CILBuilder::getCILClassType(std::string name) {
  return CIL::ClassType::get(name, context);
}

mlir::Type CILBuilder::getUnderlyingType(mlir::Type type) {
  if (auto ptrTy = type.dyn_cast<CIL::PointerType>()) {
    return getUnderlyingType(ptrTy.getEleTy());
  }
  if (auto ptrTy = type.dyn_cast<CIL::ArrayType>()) {
    return getUnderlyingType(ptrTy.getEleTy());
  }
  return type;
}

mlir::Value CILBuilder::getCILIntConstant(mlir::Location loc, int64_t value) {
  return getCILIntegralConstant(loc, getCILIntType(), value);
}

mlir::Value CILBuilder::getCILLongConstant(mlir::Location loc, int64_t value) {
  return getCILIntegralConstant(loc, getCILLongIntType(), value);
}

mlir::Value CILBuilder::getCILBoolConstant(mlir::Location loc, bool value) {
  return getCILIntegralConstant(loc, getCILBoolType(), value);
}

mlir::Value CILBuilder::getCILIntegralConstant(mlir::Location loc,
                                               int64_t value, unsigned width) {
  mlir::Type intTy;
  switch (width) {
  case 1:
    intTy = getCILBoolType();
    break;
  case 8:
    intTy = getCILChar8Type();
    break;
  case 32:
    intTy = getCILIntType();
    break;
  case 64:
    intTy = getCILLongIntType();
    break;
  default:
    llvm_unreachable("Unhandled");
  }

  return getCILIntegralConstant(loc, intTy, value);
}

unsigned CILBuilder::getCILIntOrFloatBitWidth(mlir::Type type) {
  if (auto intTy = type.dyn_cast_or_null<CIL::IntegerTy>()) {
    switch (intTy.getIntegerKind()) {
    case CIL::BoolKind:
      return 1;
    case CIL::Char8Kind:
      return 8;
    case CIL::ShortKind:
      return 16;
    case CIL::IntKind:
      return 32;
    case CIL::LongKind:
      return 64;
    case CIL::LongLongKind:
      return 128;
    default:
      llvm_unreachable("Unknown int kind");
    }
  }

  if (auto floatTy = type.dyn_cast_or_null<CIL::FloatingTy>()) {
    switch (floatTy.getFloatingKind()) {
    case CIL::Float:
      return 32;
    case CIL::Double:
      return 64;
    default:
      llvm_unreachable("Unknown float kind");
      break;
    }
  }

  llvm_unreachable("Not CIL Int or float type");
}

mlir::Value CILBuilder::getCILIntegralConstant(mlir::Location loc,
                                               mlir::Type type, int64_t value) {
  auto intTy = type.dyn_cast_or_null<CIL::IntegerTy>();
  assert(intTy && "Expecting IntegerTy");
  Attribute valueAttr;
  switch (intTy.getIntegerKind()) {
  case CIL::BoolKind:
    valueAttr = getIntegerAttr(getI1Type(), value);
    break;
  case CIL::Char8Kind:
    valueAttr = getI8IntegerAttr(value);
    break;
  case CIL::ShortKind:
    valueAttr = getI16IntegerAttr(value);
    break;
  case CIL::IntKind:
    valueAttr = getI32IntegerAttr(value);
    break;
  case CIL::LongKind:
    valueAttr = getI64IntegerAttr(value);
    break;
  // TODO: need a new attribute?
  case CIL::LongLongKind:
    valueAttr = getI64IntegerAttr(value);
    break;
  default:
    llvm_unreachable("Unhandled");
  }
  return create<CIL::CILConstantOp>(loc, type, valueAttr);
}

mlir::Value CILBuilder::getCILFloatingConstant(mlir::Location loc,
                                               mlir::Type type, double value) {

  auto floatTy = type.dyn_cast_or_null<CIL::FloatingTy>();
  assert(floatTy && "Expecting float type");
  FloatAttr valueAttr;
  switch (floatTy.getFloatingKind()) {
  case CIL::Float:
    valueAttr = getF32FloatAttr(value);
    break;
  case CIL::Double:
    valueAttr = getF64FloatAttr(value);
    break;
  default:
    llvm_unreachable("Unhandeled");
    break;
  }

  return create<CIL::CILConstantOp>(loc, type, valueAttr);
}
