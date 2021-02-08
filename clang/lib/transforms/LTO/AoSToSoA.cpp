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

//===------------------------AoSToSoA.cpp---------------------------------===//
//
// This file implements conversion of array of structs to struct of arrays
//
//===---------------------------------------------------------------------===//

#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"
#include "clang/cil/dialect/CIL/CILTypes.h"
#include "clang/cil/pass/Pass.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include <map>

class TransformInfo {
  std::string allocaFnName;

  CIL::StructType type;

  mlir::Value allocSize;

  mlir::Operation *allocUse;

  CIL::GlobalOp globalOp;

  CIL::StructType structArrType;

public:
  explicit TransformInfo(std::string name, CIL::StructType type,
                         mlir::Value size, mlir::Operation *use)
      : allocaFnName(name), type(type), allocSize(size), allocUse(use) {}

  std::string getAllocFnName() { return allocaFnName; }

  std::string getStructName() { return type.getName().str(); }

  CIL::StructType getStructType() { return type; }

  mlir::Value getAllocSize() { return allocSize; }

  mlir::Operation *getCallOp() { return allocUse; }

  CIL::GlobalOp getGlobal() { return globalOp; }

  void setGlobal(CIL::GlobalOp op) { globalOp = op; }

  CIL::StructType getArrayType() { return structArrType; }

  void setArrayType(CIL::StructType type) { structArrType = type; };

  void print() {
    llvm::errs() << "Alloc fn  : " << allocaFnName << "\n";
    llvm::errs() << "struct    : " << type.getName() << "\n";
    llvm::errs() << "Size      : " << allocSize << "\n";
    llvm::errs() << "Call      : " << *allocUse << "\n";
    llvm::errs() << "\n";
  }
};

struct AoSToSoA : public ModulePass<AoSToSoA> {

  virtual void runOnModule();

private:
  void printAnalysis();

  void populateTransformInfo(llvm::ArrayRef<Operation *> uses);

  void transform();

  void rewriteUses(TransformInfo *);

  void createAndAllocateGlobal(CIL::StructType, TransformInfo *info);

  SmallVector<TransformInfo *, 2> transformList;

  ModuleOp TheModule;
};

static CIL::StructType getStructTypeFromUse(mlir::Value value) {
  if (!value.hasOneUse())
    return {};

  auto bitcast = dyn_cast<CIL::PointerBitCastOp>(*value.user_begin());
  if (!bitcast)
    return {};

  auto type =
      bitcast.getResult().getType().dyn_cast_or_null<CIL::PointerType>();
  assert(type);
  return type.getEleTy().dyn_cast_or_null<CIL::StructType>();
}

void AoSToSoA::printAnalysis() {
  llvm::errs() << "------AoSToSoA Analysis-------\n";
  for (auto info : transformList) {
    info->print();
  }
}

static mlir::Type getNewTypeFor(mlir::Type eleType, CIL::StructType oldType,
                                CIL::StructType newType) {
  if (eleType == oldType)
    return CIL::PointerType::get(newType);

  if (auto ptrTy = eleType.dyn_cast_or_null<CIL::PointerType>())
    return CIL::PointerType::get(
        getNewTypeFor(ptrTy.getEleTy(), oldType, newType));
  return CIL::PointerType::get(eleType);
}

static int getConstantIndex(mlir::Value val) {
  auto constOp = dyn_cast<CIL::CILConstantOp>(val.getDefiningOp());
  assert(constOp);

  auto intAttr = constOp.value().cast<IntegerAttr>();
  return intAttr.getInt();
}

static void populateStoreUse(mlir::Operation *op, CIL::StructType type,
                             llvm::SmallVector<CIL::CILStoreOp, 2> &StoreOps) {
  if (auto store = dyn_cast<CIL::CILStoreOp>(op)) {
    auto ptrTy = store.getPointer().getType().cast<CIL::PointerType>();
    if (ptrTy.getEleTy() == type) {
      StoreOps.push_back(store);
      return;
    }

    if (ptrTy.getEleTy() == CIL::PointerType::get(type)) {
      StoreOps.push_back(store);
      return;
    }
  }

  if (op->getNumResults() == 1) {
    for (auto &use : op->getResult(0).getUses()) {
      populateStoreUse(use.getOwner(), type, StoreOps);
    }
  }
  return;
}

void AoSToSoA::rewriteUses(TransformInfo *info) {

  SmallVector<CIL::StructElementOp, 2> OpsToRewrite;
  SmallVector<CIL::CILStoreOp, 2> StoreOpsToRewrite;

  auto structType = info->getStructType();

  // Find StructElementOp uses of struct. Capture the secondary uses
  // which are stores, then bitcast them from node* to node_arr*
  TheModule.walk([&](CIL::StructElementOp op) {
    auto ptrType = op.ptr().getType().dyn_cast_or_null<CIL::PointerType>();
    assert(ptrType);
    CIL::StructType type =
        ptrType.getEleTy().dyn_cast_or_null<CIL::StructType>();
    assert(type);
    if (type != structType)
      return;

    for (auto &use : op.getResult().getUses()) {
      auto useOp = use.getOwner();
      populateStoreUse(useOp, info->getStructType(), StoreOpsToRewrite);
    }
    OpsToRewrite.push_back(op);
  });

  CIL::CILBuilder builder(TheModule.getContext());
  for (auto op : OpsToRewrite) {
    auto ptrType = op.ptr().getType().dyn_cast_or_null<CIL::PointerType>();
    assert(ptrType);
    CIL::StructType type =
        ptrType.getEleTy().dyn_cast_or_null<CIL::StructType>();
    assert(type);
    auto global = info->getGlobal();

    auto loc = op.getLoc();
    builder.setInsertionPoint(op);

    auto index = getConstantIndex(op.index());
    auto globalAddress = builder.create<CIL::GlobalAddressOfOp>(loc, global);
    auto fieldTy = builder.getCILPointerType(
        info->getArrayType().getStructElementType(index));

    auto newStructOp = builder.create<CIL::StructElementOp>(
        loc, fieldTy, globalAddress, op.index());
    auto newLoad = builder.create<CIL::CILLoadOp>(loc, newStructOp);

    auto I64 = builder.getCILLongIntType();

    mlir::Value currOffset =
        builder.create<CIL::CILPtrToIntOp>(loc, I64, op.ptr());

    fieldTy =
        builder.getCILPointerType(info->getArrayType().getStructElementType(0));
    auto structOp = builder.create<CIL::StructElementOp>(
        loc, fieldTy, globalAddress,
        builder.getCILIntegralConstant(loc, 0, /* width */ 32));
    mlir::Value basePtr = builder.create<CIL::CILLoadOp>(loc, structOp);
    basePtr = builder.create<CIL::CILPtrToIntOp>(loc, I64, basePtr);
    mlir::Value offset =
        builder.create<CIL::CILSubIOp>(loc, currOffset, basePtr);

    auto nullPtrOp = builder.create<CIL::NullPointerOp>(
        loc, builder.getCILPointerType(type));
    auto sizeOfOp = builder.create<CIL::SizeOfOp>(loc, I64, nullPtrOp);

    auto arrayIndex =
        builder.create<CIL::CILDivIOp>(loc, I64, offset, sizeOfOp);
    auto newPtrIndex =
        builder.create<CIL::CILPointerAddOp>(loc, newLoad, arrayIndex);

    op.getResult().replaceAllUsesWith(newPtrIndex.getResult());
  }

  for (auto op : StoreOpsToRewrite) {
    auto value = op.valueToStore();
    auto valueTy = op.pointer().getType().cast<CIL::PointerType>().getEleTy();
    builder.setInsertionPoint(op);
    auto cast =
        builder.create<CIL::PointerBitCastOp>(value.getLoc(), valueTy, value);

    // TODO : Not using replace uses or create a new store since
    //        it doesn't go well with destructors. Might be a bug?
    //        Hard coding 0 is not safe!
    op.setOperand(0, cast);
  }
}

void AoSToSoA::transform() {
  TransformInfo *info = nullptr;

  for (auto tinfo : transformList) {
    if (tinfo->getStructName() == "node") {
      if (info && info->getStructName() == "node") {
        llvm::errs() << "Multiple allocation for node found!!\n";
        llvm_unreachable("Do not transform");
      }
      info = tinfo;
    }
  }

  assert(info);

  auto structType = info->getStructType();
  llvm::SmallVector<mlir::Type, 2> types;
  auto newName = structType.getName().str() + "_arr";
  auto newType = CIL::StructType::get(structType.getContext(), types, newName);
  for (auto eleTy : structType.getElementTypes()) {
    types.push_back(getNewTypeFor(eleTy, structType, newType));
  }

  newType.setBody(types);
  newType.finalize();

  info->setArrayType(newType);
  createAndAllocateGlobal(newType, info);

  rewriteUses(info);
}

void AoSToSoA::createAndAllocateGlobal(CIL::StructType type,
                                       TransformInfo *info) {
  std::string name = type.getName().str() + "_var";

  CIL::CILBuilder builder(type.getContext());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(TheModule.getBody());

  Attribute val;
  auto global = builder.create<CIL::GlobalOp>(loc, type, false, name, val);
  info->setGlobal(global);

  builder.setInsertionPointAfter(info->getCallOp());

  auto oldCall = cast<CIL::CILCallOp>(info->getCallOp());
  loc = oldCall.getLoc();
  auto argType = oldCall.getCalleeType().getInput(1);
  auto types = type.getElementTypes();
  auto arraySize = info->getAllocSize();

  auto globalAddress = builder.create<CIL::GlobalAddressOfOp>(loc, global);

  mlir::Value currBase;
  for (unsigned int i = 0; i < types.size(); ++i) {
    auto index = builder.getCILIntegralConstant(loc, i, /*width*/ 32);

    auto fieldTy = builder.getCILPointerType(types[i]);

    auto structOp = builder.create<CIL::StructElementOp>(loc, fieldTy,
                                                         globalAddress, index);
    if (i == 0) {
      auto ptr = builder.create<CIL::PointerBitCastOp>(loc, types[i],
                                                       oldCall.getResult(0));
      builder.create<CIL::CILStoreOp>(loc, ptr, structOp);
      currBase = oldCall.getResult(0);
      continue;
    }

    auto nullPtrOp = builder.create<CIL::NullPointerOp>(loc, types[i - 1]);
    auto sizeOfOp = builder.create<CIL::SizeOfOp>(loc, argType, nullPtrOp);
    auto offset = builder.create<CIL::CILMulIOp>(loc, sizeOfOp, arraySize);
    auto offsetPtr =
        builder.create<CIL::CILPointerAddOp>(loc, currBase, offset);

    currBase = offsetPtr;

    auto ptr = builder.create<CIL::PointerBitCastOp>(loc, types[i], offsetPtr);
    builder.create<CIL::CILStoreOp>(loc, ptr, structOp);
  }
}

void AoSToSoA::populateTransformInfo(llvm::ArrayRef<Operation *> uses) {
  std::string fnName = "calloc";
  for (auto user : uses) {
    auto callOp = dyn_cast<CIL::CILCallOp>(user);
    assert(callOp);
    auto structType = getStructTypeFromUse(callOp.getResult(0));
    if (!structType)
      continue;

    SmallVector<Value, 2> args(callOp.getArgOperands());
    auto size = args[0];
    transformList.push_back(new TransformInfo(fnName, structType, size, user));
  }
}

void AoSToSoA::runOnModule() {
  TheModule = getModule();

  auto callocFunc = TheModule.lookupSymbol<mlir::FuncOp>("calloc");
  if (!callocFunc)
    return;

  mlir::Region *region = &TheModule.getBodyRegion();
  Optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses("calloc", region);

  if (!uses)
    return;

  llvm::SmallVector<Operation *, 2> callocUses;
  for (auto use : *uses) {
    callocUses.push_back(use.getUser());
  }

  populateTransformInfo(callocUses);

  printAnalysis();

  transform();

  if (failed(mlir::verify(TheModule))) {
    llvm::errs() << "Module verification failed\n";
  }
}

namespace CIL {
std::unique_ptr<mlir::Pass> createAoSToSoAPass() {
  return std::make_unique<AoSToSoA>();
}
} // namespace CIL

static PassRegistration<AoSToSoA>
    pass("aos-to-soa", "Convert array of structures to structure of arrays");
