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
//===----------------------------------------------------------------------===//
// Pass to convert the constructor call as normal class member function call.
//===----------------------------------------------------------------------===//

#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#define PASS_NAME "ConstructorLowering"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

static constexpr const char *globalConstructorFnName = "__cxx_global_var_init";

using namespace mlir;

struct ConstructorLoweringPass : public ModulePass<ConstructorLoweringPass> {
private:
public:
  virtual void runOnModule();
};

// Generates the loop depth same as array dimensions.
// Returns the array access op to the contained type.
static mlir::Value prepareLoopForArray(mlir::Value baseval,
                                       CIL::ArrayType arrType,
                                       CIL::CILBuilder &builder) {
  auto shape = arrType.getShape();
  assert(shape.size() == 1);
  assert(arrType.hasStaticShape());
  auto loc = baseval.getLoc();
  auto lb = builder.getCILLongConstant(loc, 0);
  auto ub = builder.getCILLongConstant(loc, shape[0]);
  auto one = builder.getCILLongConstant(loc, 1);

  auto forLoopOp = builder.create<CIL::ForLoopOp>(loc, lb, ub, one,
                                                  builder.getCILLongIntType());
  builder.setInsertionPointToStart(forLoopOp.getBody());
  baseval =
      builder.create<CIL::CILArrayIndexOp>(loc, baseval, forLoopOp.getIndVar());

  auto newArrType = arrType.getEleTy().dyn_cast<CIL::ArrayType>();
  if (!newArrType)
    return baseval;
  return prepareLoopForArray(baseval, newArrType, builder);
}

static bool handleClassAlloca(CIL::AllocaOp op) {
  auto constructRef = op.constructSym();
  if (!constructRef.hasValue()) {
    return true;
  }
  auto symRef = constructRef.getValue();
  auto args = op.constructArgs();
  llvm::SmallVector<mlir::Value, 2> argsList(args.begin(), args.end());

  auto allocatedType = op.getAllocatedType();
  CIL::CILBuilder builder(op.getContext());
  builder.setInsertionPointAfter(op);
  auto newAlloca =
      builder.create<CIL::AllocaOp>(op.getLoc(), op.getName(), allocatedType);
  auto baseValue = newAlloca.getResult();

  if (auto arrType = allocatedType.dyn_cast<CIL::ArrayType>()) {
    baseValue = prepareLoopForArray(baseValue, arrType, builder);
  }
  builder.create<CIL::MemberCallOp>(op.getLoc(), baseValue, symRef, llvm::None,
                                    argsList);
  op.replaceAllUsesWith(newAlloca.getResult());
  op.erase();
  return true;
}

static mlir::FuncOp getGlobalConstructorFunc(ModuleOp module) {
  auto func = module.lookupSymbol<mlir::FuncOp>(globalConstructorFnName);
  if (func) {
    return func;
  }
  CIL::CILBuilder builder(module.getContext());
  auto funcType = builder.getFunctionType({}, {});
  func = mlir::FuncOp::create(builder.getUnknownLoc(), globalConstructorFnName,
                              funcType);
  module.push_back(func);
  auto block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

  // Create global ctor operation
  builder.setInsertionPointToStart(module.getBody());
  builder.create<CIL::GlobalCtorOp>(func.getLoc(),
                                    builder.getSymbolRefAttr(func));
  return func;
}

static void handleGlobalVariables(CIL::GlobalOp op) {
  if (!op.constructSymAttr()) {
    return;
  }
  auto symRef = op.constructSymAttr();
  auto module = op.getParentOfType<mlir::ModuleOp>();
  assert(module);
  auto func = getGlobalConstructorFunc(module);
  assert(func);
  CIL::CILBuilder builder(op.getContext());
  builder.setInsertionPoint(func.getBlocks().front().getTerminator());
  auto addrOfOp = builder.create<CIL::GlobalAddressOfOp>(op.getLoc(), op);
  builder.create<CIL::MemberCallOp>(
      op.getLoc(), addrOfOp,
      builder.getSymbolRefAttr(symRef.getRootReference()), llvm::None,
      llvm::None);
  Attribute val;
  builder.setInsertionPoint(op);
  builder.create<CIL::GlobalOp>(op.getLoc(), op.getType(), false, op.getName(),
                                val);
  op.erase();
}

void ConstructorLoweringPass::runOnModule() {
  auto module = getModule();

  module.walk([&](mlir::Operation *op) {
    if (auto alloca = dyn_cast<CIL::AllocaOp>(op)) {
      handleClassAlloca(alloca);
      return;
    }
    if (auto global = dyn_cast<CIL::GlobalOp>(op)) {
      handleGlobalVariables(global);
      return;
    }
  });
}
namespace CIL {
std::unique_ptr<mlir::Pass> createConstructorLoweringPass() {
  return std::make_unique<ConstructorLoweringPass>();
}
} // namespace CIL