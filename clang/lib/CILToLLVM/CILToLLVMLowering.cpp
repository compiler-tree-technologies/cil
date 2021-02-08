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

struct CILConstantOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILConstantOpLowering(MLIRContext *_context,
                                 CILTypeConverter *converter)
      : ConversionPattern(CILConstantOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILConstantOp> transformed(operands);
    auto constOp = cast<CIL::CILConstantOp>(op);
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, converter->convertType(op->getResult(0).getType()),
        constOp.value());
    return matchSuccess();
  }
};

struct CILGlobalOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit CILGlobalOpLowering(MLIRContext *_context)
      : ConversionPattern(CIL::GlobalOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto globalOp = cast<CIL::GlobalOp>(op);
    CILTypeConverter typeConverter(context);
    auto eleTy = globalOp.getType();

    auto valAttr =
        globalOp.getAttr("external").dyn_cast_or_null<mlir::BoolAttr>();

    auto linkage = (valAttr && valAttr.getValue()) ? LLVM::Linkage::External
                                                   : LLVM::Linkage::Internal;
    mlir::LLVM::LLVMType llTy =
        typeConverter.convertType(eleTy).cast<mlir::LLVM::LLVMType>();
    mlir::LLVM::GlobalOp llGlobal = rewriter.create<mlir::LLVM::GlobalOp>(
        globalOp.getLoc(), llTy, false, linkage, globalOp.sym_name().str(),
        globalOp.getValueOrNull());
    assert(llGlobal);
    auto &cilGlobalRegion = globalOp.getInitializerRegion();
    auto &llGlobalRegion = llGlobal.getInitializerRegion();

    rewriter.inlineRegionBefore(cilGlobalRegion, llGlobalRegion,
                                llGlobalRegion.end());

    TypeConverter::SignatureConversion result(1);
    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&llGlobalRegion, result);
    rewriter.eraseOp(globalOp);
    return matchSuccess();
  }
};

struct CILAddrOfOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  mlir::ModuleOp &module;

public:
  explicit CILAddrOfOpLowering(mlir::ModuleOp &module, MLIRContext *_context)
      : ConversionPattern(CIL::GlobalAddressOfOp::getOperationName(), 1,
                          _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        module(module) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto addrOfOp = cast<CIL::GlobalAddressOfOp>(op);
    CILTypeConverter typeConverter(context);

    auto global = module.lookupSymbol<LLVM::GlobalOp>(addrOfOp.global_name());
    if (!global) {
      return matchFailure();
    }
    mlir::LLVM::LLVMType llTy = typeConverter.convertType(global.getType())
                                    .cast<mlir::LLVM::LLVMType>();
    auto llOp = rewriter.create<mlir::LLVM::AddressOfOp>(
        addrOfOp.getLoc(), llTy.getPointerTo(), addrOfOp.global_name());
    rewriter.replaceOp(addrOfOp, {llOp});
    return matchSuccess();
  }
};

struct CILSelectOpLowering : public ConversionPattern {
  MLIRContext *context;

public:
  explicit CILSelectOpLowering(MLIRContext *_context)
      : ConversionPattern(CIL::CILSelectOp::getOperationName(), 1, _context),
        context(_context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    CILSelectOpOperandAdaptor transformed(operands);
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        op, transformed.cond(), transformed.trueVal(), transformed.falseVal());
    return matchSuccess();
  }
};

struct CILIndexCastOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *typeConverter;

public:
  explicit CILIndexCastOpLowering(MLIRContext *_context,
                                  CILTypeConverter *converter)
      : ConversionPattern(CIL::CILIndexCastOp::getOperationName(), 1, _context),
        context(_context), typeConverter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    CILIndexCastOpOperandAdaptor transformed(operands);
    auto indexCastOp = cast<CILIndexCastOp>(op);

    auto targetType =
        this->typeConverter->convertType(indexCastOp.getResult().getType())
            .cast<LLVM::LLVMType>();
    auto sourceType = transformed.inval().getType().cast<LLVM::LLVMType>();
    unsigned targetBits = targetType.getUnderlyingType()->getIntegerBitWidth();
    unsigned sourceBits = sourceType.getUnderlyingType()->getIntegerBitWidth();

    if (targetBits == sourceBits)
      rewriter.replaceOp(op, transformed.inval());
    else if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, targetType,
                                                 transformed.inval());
    else
      rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, targetType,
                                                transformed.inval());
    return matchSuccess();
  }
};

struct CILBoolCastLowering : public ConversionPattern {
  MLIRContext *context;

public:
  explicit CILBoolCastLowering(MLIRContext *_context)
      : ConversionPattern(CIL::CILBoolCastOp::getOperationName(), 1, _context),
        context(_context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return matchSuccess();
  }
};

struct CILIntCastLowering : public ConversionPattern {
  MLIRContext *context;

public:
  explicit CILIntCastLowering(MLIRContext *_context)
      : ConversionPattern(CIL::IntCastOp::getOperationName(), 1, _context),
        context(_context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return matchSuccess();
  }
};

struct CILCastToStdBoolLowering : public ConversionPattern {
  MLIRContext *context;

public:
  explicit CILCastToStdBoolLowering(MLIRContext *_context)
      : ConversionPattern(CIL::CastToStdBoolOp::getOperationName(), 1,
                          _context),
        context(_context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return matchSuccess();
  }
};

struct CILLNotOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit CILLNotOpLowering(MLIRContext *_context)
      : ConversionPattern(CIL::LNotOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    OperandAdaptor<CIL::LNotOp> transformed(operands);
    auto boolTy = LLVM::LLVMType::getInt1Ty(llvmDialect);
    auto mlBoolTy = rewriter.getIntegerType(1);
    auto trueAttr = rewriter.getIntegerAttr(mlBoolTy, 1);
    auto falseAttr = rewriter.getIntegerAttr(mlBoolTy, 0);
    auto trueVal =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), boolTy, trueAttr)
            .getResult();
    auto falseVal =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), boolTy, falseAttr)
            .getResult();
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, transformed.arg(), falseVal,
                                                trueVal);
    return matchSuccess();
  }
};

struct CILCastToPointerLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  mlir::ModuleOp &module;

public:
  explicit CILCastToPointerLowering(mlir::ModuleOp &module,
                                    MLIRContext *_context)
      : ConversionPattern(CIL::CastToPointerOp::getOperationName(), 1,
                          _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        module(module) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CILTypeConverter typeConverter(context);
    rewriter.replaceOp(op, operands[0]);
    return matchSuccess();
  }
};

template <class CILCastOp, class LLVMCastOp>
struct CILCastOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit CILCastOpLowering(MLIRContext *_context)
      : ConversionPattern(CILCastOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CILTypeConverter typeConverter(context);
    mlir::LLVM::LLVMType toTy =
        typeConverter.convertType(*op->getResultTypes().begin())
            .cast<mlir::LLVM::LLVMType>();

    rewriter.replaceOpWithNewOp<LLVMCastOp>(op, toTy, operands[0]);
    return matchSuccess();
  }
};

template <class MLIROp, class LLVMOp>
struct CILExtOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit CILExtOpLowering(MLIRContext *_context)
      : ConversionPattern(MLIROp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CILTypeConverter typeConverter(context);
    mlir::LLVM::LLVMType llTy =
        typeConverter.convertType(*op->getResultTypes().begin())
            .cast<mlir::LLVM::LLVMType>();
    rewriter.replaceOpWithNewOp<LLVMOp>(op, llTy, operands[0]);
    return matchSuccess();
  }
};

struct CILPointerBitcastLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  mlir::ModuleOp &module;

public:
  explicit CILPointerBitcastLowering(mlir::ModuleOp &module,
                                     MLIRContext *_context)
      : ConversionPattern(CIL::PointerBitCastOp::getOperationName(), 1,
                          _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        module(module) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CILTypeConverter typeConverter(context);
    mlir::LLVM::LLVMType llTy =
        typeConverter.convertType(*op->getResultTypes().begin())
            .cast<mlir::LLVM::LLVMType>();
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llTy, operands[0]);
    return matchSuccess();
  }
};

struct CILAllocaOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILAllocaOpLowering(MLIRContext *_context,
                               CILTypeConverter *converter)
      : ConversionPattern(CIL::AllocaOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::AllocaOp> transformed(operands);
    auto llType = converter->convertType(*op->getResultTypes().begin());
    auto Int32 =
        LLVM::LLVMType::getInt32Ty(&llType.cast<LLVM::LLVMType>().getDialect());
    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 1);
    auto One = rewriter.create<LLVM::ConstantOp>(op->getLoc(), Int32, attr);
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, llType, One, 0);
    return matchSuccess();
  }
};

struct CILLoadOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILLoadOpLowering(MLIRContext *_context, CILTypeConverter *converter)
      : ConversionPattern(CILLoadOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILLoadOp> transformed(operands);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, transformed.pointer());
    return matchSuccess();
  }
};

struct CILNullPointerOp : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILNullPointerOp(MLIRContext *_context, CILTypeConverter *converter)
      : ConversionPattern(NullPointerOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    CILTypeConverter typeConverter(context);
    mlir::LLVM::LLVMType resultTy =
        typeConverter.convertType(*op->getResultTypes().begin())
            .cast<mlir::LLVM::LLVMType>();
    rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, resultTy);
    return matchSuccess();
  }
};

struct CILReturnOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILReturnOpLowering(MLIRContext *_context,
                               CILTypeConverter *converter)
      : ConversionPattern(CILReturnOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

struct CILPointerAddLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILPointerAddLowering(MLIRContext *_context,
                                 CILTypeConverter *converter)
      : ConversionPattern(CIL::CILPointerAddOp::getOperationName(), 1,
                          _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILPointerAddOp> transformed(operands);
    auto llType = this->converter->convertType(*op->getResultTypes().begin())
                      .cast<LLVM::LLVMType>();
    auto Int32 =
        LLVM::LLVMType::getInt32Ty(&llType.cast<LLVM::LLVMType>().getDialect());
    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
    auto Zero = rewriter.create<LLVM::ConstantOp>(op->getLoc(), Int32, attr)
                    .getResult();
    assert(Zero);
    SmallVector<mlir::Value, 2> ops{transformed.rhs()};
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, llType, transformed.lhs(),
                                             ops);
    return matchSuccess();
  }
};

template <class TerminatorLikeOp>
struct CILTerminatorLikeOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;

public:
  explicit CILTerminatorLikeOpLowering(MLIRContext *_context)
      : ConversionPattern(TerminatorLikeOp::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value>> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<ValueRange, 2> operandRanges(operands.begin(), operands.end());
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, properOperands, destinations, operandRanges, op->getAttrs());
    return this->matchSuccess();
  }
};

struct CILArrayIndexOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILArrayIndexOpLowering(MLIRContext *_context,
                                   CILTypeConverter *converter)
      : ConversionPattern(CILArrayIndexOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILArrayIndexOp> transformed(operands);
    auto llType = this->converter->convertType(*op->getResultTypes().begin())
                      .cast<LLVM::LLVMType>();
    auto Int32 =
        LLVM::LLVMType::getInt32Ty(&llType.cast<LLVM::LLVMType>().getDialect());
    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
    auto Zero = rewriter.create<LLVM::ConstantOp>(op->getLoc(), Int32, attr)
                    .getResult();
    SmallVector<mlir::Value, 2> ops{Zero, transformed.idx()};
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, llType, transformed.array(),
                                             ops);
    return matchSuccess();
  }
};

struct CILGEPOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILGEPOpLowering(MLIRContext *_context, CILTypeConverter *converter)
      : ConversionPattern(CILGEPOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    OperandAdaptor<CIL::CILGEPOp> transformed(operands);
    auto llType = this->converter->convertType(*op->getResultTypes().begin())
                      .cast<LLVM::LLVMType>();
    SmallVector<mlir::Value, 2> ops;

    for (auto idx : transformed.indices()) {
      ops.push_back(idx);
    }

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, llType, transformed.pointer(),
                                             ops);
    return matchSuccess();
  }
};

struct CILUndefOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILUndefOpLowering(MLIRContext *_context,
                              CILTypeConverter *converter)
      : ConversionPattern(CIL::UndefOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto llType = converter->convertType(*op->getResultTypes().begin());
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, llType);
    return matchSuccess();
  }
};

struct CILSizeOfOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILSizeOfOpLowering(MLIRContext *_context,
                               CILTypeConverter *converter)
      : ConversionPattern(SizeOfOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::SizeOfOp> transformed(operands);

    auto llType = this->converter->convertType(*op->getResultTypes().begin())
                      .cast<LLVM::LLVMType>();

    auto ptrType = this->converter->convertType(transformed.ptr().getType())
                       .cast<LLVM::LLVMType>();
    auto I64 =
        LLVM::LLVMType::getInt64Ty(&llType.cast<LLVM::LLVMType>().getDialect());

    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1);
    auto one =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), I64, attr).getResult();

    ArrayRef<mlir::Value> args{transformed.ptr(), one};
    auto gep = rewriter.create<LLVM::GEPOp>(op->getLoc(), ptrType, args);
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, llType, gep);
    return matchSuccess();
  }
};

struct CILStructElementOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILStructElementOpLowering(MLIRContext *_context,
                                      CILTypeConverter *converter)
      : ConversionPattern(StructElementOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::StructElementOp> transformed(operands);
    auto llType = this->converter->convertType(*op->getResultTypes().begin())
                      .cast<LLVM::LLVMType>();
    auto Int32 =
        LLVM::LLVMType::getInt32Ty(&llType.cast<LLVM::LLVMType>().getDialect());
    auto attr = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
    auto Zero = rewriter.create<LLVM::ConstantOp>(op->getLoc(), Int32, attr)
                    .getResult();
    SmallVector<mlir::Value, 2> ops{Zero, transformed.index()};
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, llType, transformed.ptr(),
                                             ops);
    return matchSuccess();
  }
};

struct CILStoreOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILStoreOpLowering(MLIRContext *_context,
                              CILTypeConverter *converter)
      : ConversionPattern(CILStoreOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILStoreOp> transformed(operands);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.valueToStore(),
                                               transformed.pointer());
    return matchSuccess();
  }
};

struct CILCmpPointerEqLowering : public ConversionPattern {
  LLVM::LLVMDialect *llvmDialect;
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILCmpPointerEqLowering(MLIRContext *_context,
                                   CILTypeConverter *converter)
      : ConversionPattern(CIL::CmpPointerEqualOp::getOperationName(), 1,
                          _context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CmpPointerEqualOp> transformed(operands);
    auto I64 = LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto lhs =
        rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), I64, transformed.lhs());
    auto rhs =
        rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), I64, transformed.rhs());
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, lhs,
                                              rhs);
    return matchSuccess();
  }
};

struct CILCMPIOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILCMPIOpLowering(MLIRContext *_context, CILTypeConverter *converter)
      : ConversionPattern(CIL::CILCmpIOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILCmpIOp> transformed(operands);
    auto cmpOp = cast<CIL::CILCmpIOp>(op);
    auto pred = cmpOp.getPredicate();
    LLVM::ICmpPredicate llvmPred;
    // TODO : move to a function.
    switch (pred) {
    case CmpIPredicate::eq:
      llvmPred = LLVM::ICmpPredicate::eq;
      break;
    case CmpIPredicate::ne:
      llvmPred = LLVM::ICmpPredicate::ne;
      break;
    case CmpIPredicate::slt:
      llvmPred = LLVM::ICmpPredicate::slt;
      break;
    case CmpIPredicate::sle:
      llvmPred = LLVM::ICmpPredicate::sle;
      break;
    case CmpIPredicate::sgt:
      llvmPred = LLVM::ICmpPredicate::sgt;
      break;
    case CmpIPredicate::sge:
      llvmPred = LLVM::ICmpPredicate::sge;
      break;
    default:
      llvm_unreachable("Unhandled MLIR to LLVM CmpIPred translation");
    };
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, llvmPred, transformed.lhs(),
                                              transformed.rhs());
    return matchSuccess();
  }
};

struct CILCMPFOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILCMPFOpLowering(MLIRContext *_context, CILTypeConverter *converter)
      : ConversionPattern(CIL::CILCmpFOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::CILCmpFOp> transformed(operands);
    auto cmpOp = cast<CIL::CILCmpFOp>(op);
    auto pred = cmpOp.getPredicate();
    LLVM::FCmpPredicate llvmPred;
    // TODO : move to a function.
    switch (pred) {
    case CmpFPredicate::OEQ:
      llvmPred = LLVM::FCmpPredicate::oeq;
      break;
    case CmpFPredicate::ONE:
      llvmPred = LLVM::FCmpPredicate::one;
      break;
    case CmpFPredicate::OLT:
      llvmPred = LLVM::FCmpPredicate::olt;
      break;
    case CmpFPredicate::OLE:
      llvmPred = LLVM::FCmpPredicate::ole;
      break;
    case CmpFPredicate::OGT:
      llvmPred = LLVM::FCmpPredicate::ogt;
      break;
    case CmpFPredicate::OGE:
      llvmPred = LLVM::FCmpPredicate::oge;
      break;
    default:
      llvm_unreachable("Unhandled MLIR to LLVM CmpIPred translation");
    };
    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(op, llvmPred, transformed.lhs(),
                                              transformed.rhs());
    return matchSuccess();
  }
};

struct CILConditionalOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILConditionalOpLowering(MLIRContext *_context,
                                    CILTypeConverter *converter)
      : ConversionPattern(CIL::ConditionalOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::ConditionalOp> transformed(operands);
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        op, transformed.cond(), transformed.trueVal(), transformed.falseVal());
    return matchSuccess();
  }
};

struct CILGlobalCtorOpLowering : public ConversionPattern {
  LLVM::LLVMDialect *llvmDialect;
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILGlobalCtorOpLowering(MLIRContext *_context,
                                   CILTypeConverter *converter)
      : ConversionPattern(CIL::GlobalCtorOp::getOperationName(), 1, _context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::GlobalCtorOp> transformed(operands);
    auto globalCtorOp = llvm::cast<GlobalCtorOp>(op);
    auto i32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto i64Ty = LLVM::LLVMType::getInt64Ty(llvmDialect);
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        LLVM::LLVMType::getVoidTy(llvmDialect), false);
    auto funcPointerTy = funcTy.getPointerTo();
    auto voidPtr = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto structTy = LLVM::LLVMType::getStructTy(
        llvmDialect, {i32Ty, funcPointerTy, voidPtr});
    auto arrTy = LLVM::LLVMType::getArrayTy(structTy, 1);

    Attribute val;
    auto newOp = rewriter.create<LLVM::GlobalOp>(op->getLoc(), arrTy, false,
                                                 LLVM::Linkage::Appending,
                                                 "llvm.global_ctors", val);

    rewriter.createBlock(&newOp.getInitializerRegion());
    auto block = newOp.getInitializerBlock();

    rewriter.setInsertionPointToStart(block);

    // Build value.
    auto i32Val = rewriter.getI32IntegerAttr(65535);
    auto nullPointer = rewriter.getI64IntegerAttr(0);
    auto i32 = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32Ty, i32Val);
    auto zeroVal =
        rewriter.create<LLVM::ConstantOp>(op->getLoc(), i64Ty, nullPointer);
    auto nullVal = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), voidPtr,
                                                     zeroVal.getResult());

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto func = module.lookupSymbol<mlir::FuncOp>(
        globalCtorOp.constructor().getRootReference());
    assert(func);
    auto funcPtr = rewriter.create<CIL::CILConstantOp>(
        op->getLoc(), func.getType(), globalCtorOp.constructor());

    auto tempStruct = rewriter.create<LLVM::UndefOp>(op->getLoc(), arrTy);
    auto insertValOp = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), arrTy, tempStruct, i32, rewriter.getI64ArrayAttr({0, 0}));
    insertValOp = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), arrTy, insertValOp, funcPtr,
        rewriter.getI64ArrayAttr({0, 1}));
    insertValOp = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), arrTy, insertValOp, nullVal,
        rewriter.getI64ArrayAttr({0, 2}));
    rewriter.create<LLVM::ReturnOp>(op->getLoc(), insertValOp.getResult());
    rewriter.setInsertionPoint(op);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

template <class CILBinOp, class LLVMBinOp>
struct CILBinaryOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILBinaryOpLowering(MLIRContext *_context,
                               CILTypeConverter *converter)
      : ConversionPattern(CILBinOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CILBinOp> transformed(operands);
    rewriter.replaceOpWithNewOp<LLVMBinOp>(op, transformed.lhs(),
                                           transformed.rhs());
    return matchSuccess();
  }
};

// A CallOp automatically promotes MemRefType to a sequence of alloca /
// store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CILCallOpLowering : public ConversionPattern {
  MLIRContext *context;
  LLVM::LLVMDialect *llvmDialect;
  CILTypeConverter *converter;

public:
  explicit CILCallOpLowering(MLIRContext *_context, CILTypeConverter *converter)
      : ConversionPattern(CallOpType::getOperationName(), 1, _context),
        context(_context),
        llvmDialect(_context->getRegisteredDialect<LLVM::LLVMDialect>()),
        converter(converter) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CallOpType> transformed(operands);
    auto callOp = cast<CallOpType>(op);

    // Pack the result types into a struct.
    Type packedResult;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());
    if (numResults != 0) {
      if (!(packedResult = this->converter->packFunctionResults(resultTypes))) {
        return this->matchFailure();
      }
    }

    SmallVector<Value, 4> opOperands(op->getOperands());
    auto promoted = this->converter->promoteMemRefDescriptors(
        op->getLoc(), opOperands, operands, rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), packedResult,
                                               promoted, op->getAttrs());

    // If < 2 results, packing did not do anything and we can just return.
    if (numResults < 2) {
      SmallVector<Value, 4> results(newOp.getResults());
      rewriter.replaceOp(op, results);
      return this->matchSuccess();
    }

    SmallVector<Value, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->converter->convertType(op->getResult(i).getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          rewriter.getIndexArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);

    return this->matchSuccess();
  }
};

struct CILInsertValueOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILInsertValueOpLowering(MLIRContext *_context,
                                    CILTypeConverter *converter)
      : ConversionPattern(CIL::InsertValueOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::InsertValueOp> transformed(operands);

    auto insertOp = cast<CIL::InsertValueOp>(op);
    auto llType = this->converter->convertType(*op->getResultTypes().begin());
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, llType, transformed.container(), transformed.value(),
        insertOp.position());
    return matchSuccess();
  }
};

struct CILExtractValueOpLowering : public ConversionPattern {
  MLIRContext *context;
  CILTypeConverter *converter;

public:
  explicit CILExtractValueOpLowering(MLIRContext *_context,
                                     CILTypeConverter *converter)
      : ConversionPattern(CIL::ExtractValueOp::getOperationName(), 1, _context),
        context(_context), converter(converter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<mlir::Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CIL::ExtractValueOp> transformed(operands);
    auto extractOp = cast<CIL::ExtractValueOp>(op);

    auto llType = this->converter->convertType(*op->getResultTypes().begin());
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, llType, transformed.container(), extractOp.position());
    return matchSuccess();
  }
};

struct CILToLLVMLowering : public ModulePass<CILToLLVMLowering> {
  virtual void runOnModule() {

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    OwningRewritePatternList patterns;
    CILTypeConverter typeConverter(&getContext());

    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    auto module = getModule();

    auto _context = &getContext();

    patterns.insert<CILConstantOpLowering>(_context, &typeConverter);
    patterns.insert<CILLoadOpLowering>(_context, &typeConverter);
    patterns.insert<CILStoreOpLowering>(_context, &typeConverter);
    patterns.insert<CILAllocaOpLowering>(_context, &typeConverter);

    patterns.insert<CILGlobalOpLowering>(_context);
    patterns.insert<CILAddrOfOpLowering>(module, _context);
    patterns.insert<CILCastToPointerLowering>(module, _context);
    patterns.insert<CILPointerBitcastLowering>(module, _context);
    patterns.insert<CILCallOpLowering<CIL::CILCallOp>>(_context,
                                                       &typeConverter);
    patterns.insert<CILCallOpLowering<CIL::CILCallIndirectOp>>(_context,
                                                               &typeConverter);
    patterns.insert<CILCastToStdBoolLowering>(_context);
    patterns.insert<CILIntCastLowering>(_context);
    patterns.insert<CILBoolCastLowering>(_context);
    patterns.insert<CILIndexCastOpLowering>(_context, &typeConverter);
    patterns.insert<CILSelectOpLowering>(_context);

    // NULL op lowering
    patterns.insert<CILNullPointerOp>(_context, &typeConverter);

    patterns.insert<CILArrayIndexOpLowering>(_context, &typeConverter);
    patterns.insert<CILGEPOpLowering>(_context, &typeConverter);
    patterns.insert<CILPointerAddLowering>(_context, &typeConverter);
    patterns.insert<CILUndefOpLowering>(_context, &typeConverter);
    patterns.insert<CILInsertValueOpLowering>(_context, &typeConverter);
    patterns.insert<CILExtractValueOpLowering>(_context, &typeConverter);

    patterns.insert<CILStructElementOpLowering>(_context, &typeConverter);

    patterns.insert<CILSizeOfOpLowering>(_context, &typeConverter);

    patterns.insert<CILCMPIOpLowering>(_context, &typeConverter);
    patterns.insert<CILCMPFOpLowering>(_context, &typeConverter);
    patterns.insert<CILCmpPointerEqLowering>(_context, &typeConverter);

    patterns.insert<CILExtOpLowering<CIL::ZeroExtendOp, LLVM::ZExtOp>>(
        _context);
    patterns.insert<CILExtOpLowering<CIL::SignExtendOp, LLVM::SExtOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::CILFPTruncOp, LLVM::FPTruncOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::CILFPExtOp, LLVM::FPExtOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::TruncateOp, LLVM::TruncOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::CILSIToFPOp, LLVM::SIToFPOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::CILFPToSIOp, LLVM::FPToSIOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::CILIntToPtrOp, LLVM::IntToPtrOp>>(
        _context);
    patterns.insert<CILCastOpLowering<CIL::CILPtrToIntOp, LLVM::PtrToIntOp>>(
        _context);

    // Control flow operations
    patterns.insert<CILTerminatorLikeOpLowering<CIL::CILIfOp>>(_context);
    patterns.insert<CILTerminatorLikeOpLowering<CIL::CILWhileOp>>(_context);
    patterns.insert<CILTerminatorLikeOpLowering<CIL::CILDoWhileOp>>(_context);
    patterns.insert<CILTerminatorLikeOpLowering<CIL::CILForOp>>(_context);
    patterns.insert<CILConditionalOpLowering>(_context, &typeConverter);
    patterns.insert<CILReturnOpLowering>(_context, &typeConverter);

    patterns.insert<CILBinaryOpLowering<CIL::CILAddIOp, LLVM::AddOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILSubIOp, LLVM::SubOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILMulIOp, LLVM::MulOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILDivIOp, LLVM::SDivOp>>(
        _context, &typeConverter);
    // TODO: Handle for unsigned binary integers.
    patterns.insert<CILBinaryOpLowering<CIL::CILModIOp, LLVM::SRemOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILModUIOp, LLVM::URemOp>>(
        _context, &typeConverter);

    patterns.insert<CILBinaryOpLowering<CIL::CILAddFOp, LLVM::FAddOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILSubFOp, LLVM::FSubOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILMulFOp, LLVM::FMulOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILDivFOp, LLVM::FDivOp>>(
        _context, &typeConverter);

    // Bitwise ops
    patterns.insert<CILBinaryOpLowering<CIL::CILShlOp, LLVM::ShlOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILAShrOp, LLVM::AShrOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILAndOp, LLVM::AndOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILOrOp, LLVM::OrOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILXOrOp, LLVM::XOrOp>>(
        _context, &typeConverter);

    // Logical operations
    patterns.insert<CILBinaryOpLowering<CIL::CILLogicalAndOp, LLVM::AndOp>>(
        _context, &typeConverter);
    patterns.insert<CILBinaryOpLowering<CIL::CILLogicalOrOp, LLVM::OrOp>>(
        _context, &typeConverter);

    patterns.insert<CILLNotOpLowering>(_context);

    patterns.insert<CILGlobalCtorOpLowering>(_context, &typeConverter);

    mlir::OpBuilder builder(_context);
    if (failed(applyFullConversion(module, target, patterns, &typeConverter)))
      signalPassFailure();
  }
};

namespace CIL {
namespace lowering {
std::unique_ptr<mlir::Pass> createCILToLLVMLoweringPass() {
  return std::make_unique<CILToLLVMLowering>();
}
} // namespace lowering
} // namespace CIL
