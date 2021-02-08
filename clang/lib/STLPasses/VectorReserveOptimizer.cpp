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

#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <iostream>
#include <vector>
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace mlir;
using namespace CIL;

struct VectorReserveOptimizerPass
    : public ModulePass<VectorReserveOptimizerPass> {
public:
  virtual void runOnModule();
};

static bool isInsideLoop(CIL::MemberCallOp op) {
  auto block = op.getOperation()->getBlock();

  for (auto predBlock : block->getPredecessors()) {
    for (auto &op : predBlock->getOperations()) {
      if (auto forOp = dyn_cast<CIL::CILForOp>(op)) {
        return forOp.getTrueDest() == block;
      }
    }
  }
  return false;
}

void VectorReserveOptimizerPass::runOnModule() {
  auto module = getModule();
  CILBuilder builder(module.getContext());
  SmallVector<CIL::MemberCallOp, 2> pushBackCallList;
  module.walk([&](CIL::MemberCallOp memberCall) {
    if (memberCall.getCallee().getRootReference() ==
            "_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backERKi" ||
        memberCall.getCallee().getRootReference() ==
            "_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backEOi") {
      if (isInsideLoop(memberCall)) {
        pushBackCallList.push_back(memberCall);
      }
    }
  });

  if (pushBackCallList.empty())
    return;

  // Hardcoded!!
  auto memCall = pushBackCallList[0];
  auto loc = memCall.getLoc();
  SmallVector<mlir::Value, 2> args(memCall.getArgOperands());
  auto vectorBase = args[0];

  mlir::Block *predBlock;
  for (auto pred : memCall.getOperation()->getBlock()->getPredecessors()) {
    predBlock = pred;
    break;
  }

  builder.setInsertionPointToStart(predBlock);
  auto vectorClassOp = module.lookupSymbol<CIL::ClassOp>("std::__1::vector");
  auto VectorReserveFuncOp = vectorClassOp.lookupSymbol<mlir::FuncOp>(
      "_ZNSt3__16vectorIiNS_9allocatorIiEEE7reserveEm");

  auto symRef = builder.getSymbolRefAttr(VectorReserveFuncOp.getName(),
                                         builder.getSymbolRefAttr("vector"));
  // hardcoded
  auto I32 = builder.getCILIntType();
  auto tripCount = builder.getCILIntegralConstant(loc, I32, 10);

  SmallVector<mlir::Value, 2> argsList{tripCount};
  builder.create<CIL::MemberCallOp>(loc, vectorBase, symRef,
                                    VectorReserveFuncOp.getCallableResults(),
                                    argsList);
}

namespace CIL {
std::unique_ptr<mlir::Pass> createVectorReserveOptimizerPass() {
  return std::make_unique<VectorReserveOptimizerPass>();
}
} // namespace CIL
