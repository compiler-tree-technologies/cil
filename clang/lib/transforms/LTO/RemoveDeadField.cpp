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

//===----------------------RemoveDeadFields.cpp---------------------------===//
//
// This file implements dead fields removal pass
//
//===---------------------------------------------------------------------===//

#include "clang/cil/dialect/CIL/CILOps.h"
#include "clang/cil/dialect/CIL/CILTypes.h"
#include "clang/cil/pass/Pass.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include <map>

enum AccessType { NOACCESS = 0x1, READ = 0x2, WRITE = 0x4, UNKNOWN = 0x8 };

class StructInfo {

private:
  // Number of fields
  unsigned int numFields;

  // bool recursiveStruct;

  llvm::SmallPtrSet<mlir::Type, 2> parentTypes;

  llvm::SmallPtrSet<mlir::Value, 2> uses;

  llvm::SmallVector<int, 2> accessTypes;

  CIL::StructType type;

public:
  explicit StructInfo(CIL::StructType strType) : type(strType) {
    numFields = type.getNumElementTypes();
    accessTypes.resize(numFields);
    int accessType = (0x0 | AccessType::NOACCESS);
    for (unsigned int i = 0; i < numFields; ++i) {
      accessTypes[i] = accessType;
    }
  }

  void getUses(llvm::SmallPtrSet<mlir::Value, 2> &useSet) {
    useSet.clear();
    useSet.insert(uses.begin(), uses.end());
  }

  void insertUse(mlir::Value ptr) { uses.insert(ptr); }

  void insertParentType(CIL::StructType strTy) { parentTypes.insert(strTy); }

  unsigned int getNumFields() { return numFields; }

  bool isDeadField(unsigned int idx) {
    return (accessTypes[idx] & AccessType::WRITE ||
            accessTypes[idx] & AccessType::NOACCESS) &&
           !(accessTypes[idx] & AccessType::UNKNOWN ||
             accessTypes[idx] & AccessType::READ);
  }

  bool hasDeadFields() {
    for (unsigned int i = 0; i < numFields; i++) {
      if (isDeadField(i)) {
        return true;
      }
    }
    return false;
  }

  void populateDeadFieldIndices(SmallVectorImpl<unsigned int> &deadFields) {
    for (unsigned int i = 0; i < numFields; i++) {
      if (isDeadField(i)) {
        deadFields.push_back(i);
      }
    }
  }

  void setAccessType(unsigned int idx, AccessType accesType) {
    accessTypes[idx] |= accesType;
  }

  llvm::StringRef getName() { return type.getName(); }

  std::string getAccessTypeAsString(int accesType) {
    if (accesType & AccessType::UNKNOWN)
      return "UNKNOWN";

    std::string accesString = "";
    if (accesType & READ)
      accesString += "READ";

    if (accesType & WRITE)
      accesString += "WRITE";

    if (accesString != "")
      return accesString;

    if (accesType & AccessType::NOACCESS)
      return "NOACCESS";

    return accesString;
  }

  void print(int id = 0) {
    llvm::errs().indent(id) << "\nstruct." << getName() << " {\n";
    for (unsigned i = 0; i < numFields; i++) {
      llvm::errs().indent(id + 2)
          << i << " : " << getAccessTypeAsString(accessTypes[i]) << "\n";
    }
    llvm::errs().indent(id) << "}\n\n";

    llvm::SmallVector<unsigned int, 2> deadIndices;
    populateDeadFieldIndices(deadIndices);
    if (deadIndices.size() > 0) {
      llvm::errs().indent(id) << "Dead indices : ";
      for (auto idx : deadIndices) {
        llvm::errs() << idx << " ";
      }
      llvm::errs() << "\n";
    }

    if (parentTypes.size() > 0) {
      llvm::errs().indent(id) << "Parent struct : ";
      for (auto parentType : parentTypes) {
        llvm::errs() << "struct."
                     << parentType.cast<CIL::StructType>().getName() << " ";
      }
      llvm::errs() << "\n";
    }
    llvm::errs().indent(id) << "Number of uses " << uses.size() << "\n";
  }
};

class StructAnalysisInfo {
  llvm::DenseMap<mlir::Type, StructInfo *> container;

public:
  StructInfo *getOrCreateStructInfo(CIL::StructType type) {
    if (container.find(type) != container.end())
      return container[type];

    container[type] = new StructInfo(type);

    for (auto eleType : type.getElementTypes()) {
      if (auto ptrTy = eleType.dyn_cast_or_null<CIL::PointerType>())
        eleType = ptrTy.getEleTy();

      if (eleType == type)
        continue;

      if (auto strTy = eleType.dyn_cast_or_null<CIL::StructType>()) {
        auto childInfo = getOrCreateStructInfo(strTy);
        childInfo->insertParentType(type);
      }
    }
    return container[type];
  }

  StructInfo *getInfo(CIL::StructType type) {
    if (container.find(type) != container.end())
      return container[type];
    llvm_unreachable("info not created");
  }

  void print() {
    llvm::errs() << "\n ====== Struct StructAnalysisInfo ====== \n";

    for (std::pair<mlir::Type, StructInfo *> it : container) {
      it.second->print(2);
    }

    llvm::errs() << "\n ======================================= \n";
  }

  void
  populateDeadFieldStructs(SmallVectorImpl<CIL::StructType> &deadFieldStructs) {
    for (std::pair<mlir::Type, StructInfo *> it : container) {
      if (it.second->hasDeadFields()) {
        deadFieldStructs.push_back(it.first.cast<CIL::StructType>());
      }
    }
  }
};

static mlir::Type getNewTypeFor(mlir::Type eleType, CIL::StructType oldType,
                                CIL::StructType newType) {
  if (eleType == oldType)
    return newType;

  if (auto ptrTy = eleType.dyn_cast_or_null<CIL::PointerType>())
    return CIL::PointerType::get(
        getNewTypeFor(ptrTy.getEleTy(), oldType, newType));
  return eleType;
}

// Not used currently
struct TransformInfo {
  std::map<int, int> IndexMap;
  CIL::StructType oldType;
  CIL::StructType newType;
};

// Not used currently.
struct AllocaRewriter : public OpRewritePattern<CIL::AllocaOp> {
  TransformInfo *info;

public:
  mlir::Type getUnderlyingType(mlir::Type type) const {
    if (auto ptrTy = type.dyn_cast_or_null<CIL::PointerType>()) {
      return getUnderlyingType(ptrTy.getEleTy());
    }
    return type;
  }

  int getConstantIndex(mlir::Value val, mlir::Type &attrType) const {
    auto constOp = dyn_cast<CIL::CILConstantOp>(val.getDefiningOp());
    assert(constOp);

    auto intAttr = constOp.value().cast<IntegerAttr>();
    attrType = intAttr.getType();
    return intAttr.getInt();
  }

  AllocaRewriter(mlir::MLIRContext *context, TransformInfo *_info)
      : OpRewritePattern<CIL::AllocaOp>(context), info(_info) {}

  PatternMatchResult matchAndRewrite(CIL::AllocaOp op,
                                     PatternRewriter &rewriter) const override {
    auto actualTy = getUnderlyingType(op.getAllocatedType());
    if (actualTy != info->oldType)
      return matchFailure();

    for (auto &use : op.getResult().getUses()) {
      auto useOp = use.getOwner();
      if (auto structOp = dyn_cast<CIL::StructElementOp>(useOp)) {
        mlir::Type attrType;
        auto index = getConstantIndex(structOp.index(), attrType);
        auto newIndex = info->IndexMap[index];
        assert(newIndex >= 0 && "Use of dead fields found!!");
        if (newIndex == index)
          continue;
        auto oldIndexOp = structOp.index().getDefiningOp();
        rewriter.replaceOpWithNewOp<CIL::CILConstantOp>(
            oldIndexOp, structOp.index().getType(),
            rewriter.getIntegerAttr(attrType, newIndex));
      }
    }

    auto newType =
        getNewTypeFor(op.getAllocatedType(), info->oldType, info->newType);

    // TODO : FIX name
    rewriter.replaceOpWithNewOp<CIL::AllocaOp>(op, "", newType);
    return matchSuccess();
  }
};

struct DeadFieldRemover : public ModulePass<DeadFieldRemover> {

  virtual void runOnModule();

  void analyzeStructAccess(CIL::StructElementOp op);

  void transform(mlir::ModuleOp op);

private:
  StructAnalysisInfo analyzer;
};

void DeadFieldRemover::analyzeStructAccess(CIL::StructElementOp op) {
  auto ptrType = op.ptr().getType().dyn_cast_or_null<CIL::PointerType>();
  assert(ptrType);

  CIL::StructType type = ptrType.getEleTy().dyn_cast_or_null<CIL::StructType>();
  assert(type);

  auto constOp = dyn_cast<CIL::CILConstantOp>(op.index().getDefiningOp());
  assert(constOp);

  auto intAttr = constOp.value().cast<IntegerAttr>();
  auto index = intAttr.getInt();

  StructInfo *info = analyzer.getOrCreateStructInfo(type);
  for (auto &use : op.getResult().getUses()) {
    auto useOp = use.getOwner();
    if (isa<CIL::CILLoadOp>(useOp)) {
      // ignore dummy loads;
      if (useOp->getResult(0).use_empty())
        continue;

      info->setAccessType(index, AccessType::READ);
      continue;
    }
    if (auto storeOp = dyn_cast<CIL::CILStoreOp>(useOp)) {
      if (storeOp.pointer() == op.getResult())
        info->setAccessType(index, AccessType::WRITE);
      else
        info->setAccessType(index, AccessType::UNKNOWN);
      continue;
    }

    // TODO : Being safe for now, we can further analyze and check for
    //        ultimate load/store uses
    info->setAccessType(index, AccessType::UNKNOWN);
  }

  info->insertUse(op);
}

static int getConstantIndex(mlir::Value val, mlir::Type &attrType) {
  auto constOp = dyn_cast<CIL::CILConstantOp>(val.getDefiningOp());
  assert(constOp);

  auto intAttr = constOp.value().cast<IntegerAttr>();
  attrType = intAttr.getType();
  return intAttr.getInt();
}

void DeadFieldRemover::transform(mlir::ModuleOp module) {
  llvm::SmallVector<CIL::StructType, 2> structList;
  analyzer.populateDeadFieldStructs(structList);

  for (auto structType : structList) {
    auto info = analyzer.getInfo(structType);

    llvm::SmallVector<unsigned int, 2> deadFields;
    info->populateDeadFieldIndices(deadFields);

    llvm::SmallVector<mlir::Type, 2> types;

    int offset = 0;
    std::map<int, int> IndexMap;

    // Pushing -1 to mark end of list;
    deadFields.push_back(-1);
    llvm::errs() << "Remap of Struct " << structType.getName() << "\n";
    auto numFields = info->getNumFields();
    for (unsigned int i = 0; i < numFields; i++) {
      if (i == deadFields[offset]) {
        IndexMap[i] = -1;
        offset++;
        llvm::errs().indent(2) << "Removing index " << i << "\n";
        continue;
      }
      IndexMap[i] = i - offset;
      llvm::errs().indent(2) << "Remap " << i << " " << IndexMap[i] << "\n";
      types.push_back(structType.getStructElementType(i));
    }

    structType.setBody(types);
    structType.finalize();

    llvm::SmallPtrSet<mlir::Value, 2> uses;
    info->getUses(uses);
    llvm::SmallVector<mlir::Operation *, 2> opsToRemove;
    mlir::OpBuilder builder(structType.getContext());
    for (auto use : uses) {
      auto structOp = dyn_cast<CIL::StructElementOp>(use.getDefiningOp());
      assert(use);
      mlir::Type attrType;
      auto index = getConstantIndex(structOp.index(), attrType);
      auto newIndex = IndexMap[index];

      // Write only use
      if (newIndex == -1) {
        for (auto &use : structOp.getResult().getUses()) {
          auto useOp = use.getOwner();
          if (isa<CIL::CILLoadOp>(useOp)) {
            // ignore dummy loads;
            assert(useOp->getResult(0).use_empty());
            opsToRemove.push_back(useOp);
            continue;
          }
          auto storeOp = dyn_cast<CIL::CILStoreOp>(useOp);
          assert(storeOp);
          opsToRemove.push_back(useOp);
        }
        opsToRemove.push_back(structOp);
      }

      if (newIndex == index)
        continue;
      auto oldIndexOp = structOp.index().getDefiningOp();
      builder.setInsertionPoint(oldIndexOp);
      auto newIndexOp = builder.create<CIL::CILConstantOp>(
          oldIndexOp->getLoc(), structOp.index().getType(),
          builder.getIntegerAttr(attrType, newIndex));
      oldIndexOp->replaceAllUsesWith(newIndexOp);
    }

    for (auto op : opsToRemove)
      op->erase();
  }
}

void DeadFieldRemover::runOnModule() {
  auto module = getModule();

  module.walk([this](CIL::StructElementOp op) { analyzeStructAccess(op); });

  analyzer.print();

  transform(module);
}

namespace CIL {
std::unique_ptr<mlir::Pass> createDeadFieldRemoverPass() {
  return std::make_unique<DeadFieldRemover>();
}
} // namespace CIL

static PassRegistration<DeadFieldRemover>
    pass("remove-dead-fields", "Pass to remove dead fields from a struct");
