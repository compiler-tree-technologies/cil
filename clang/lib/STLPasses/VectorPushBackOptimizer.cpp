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
#include <vector>
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace mlir;
using namespace CIL;

struct VectorPushBackOptimizerPass
    : public ModulePass<VectorPushBackOptimizerPass> {
public:
  virtual void runOnModule();
};

void VectorPushBackOptimizerPass::runOnModule() {
  auto module = getModule();

  SmallVector<CIL::MemberCallOp, 4> pushBackCallList;
  SmallVector<CIL::CILConstantOp, 4> pushBackArgs;
  std::vector<mlir::Attribute> attrElements;
  module.walk([&](CIL::MemberCallOp memberCall) {
    if (memberCall.getCallee().getRootReference() ==
        "_ZNSt3__16vectorIiNS_9allocatorIiEEE9push_backEOi") {
      pushBackCallList.push_back(memberCall);
    }
  });

  if (pushBackCallList.empty() || pushBackCallList.size() < 3) {
    return;
  }

  mlir::Value vectorBase;
  for (auto call : pushBackCallList) {
    SmallVector<mlir::Value, 1> args(call.getArgOperands());
    assert(args.size() == 2 &&
           "vector::push_back() with two args not possible!!");
    vectorBase = args[0];
    if (auto arg = dyn_cast<CIL::CILConstantOp>(args[1].getDefiningOp())) {
      pushBackArgs.push_back(arg);
      attrElements.push_back(arg.value());
    }
  }

  // hardcoded!
  auto op = pushBackCallList[pushBackCallList.size() - 1];
  CILBuilder builder(op);
  auto StdInitTy = builder.getCILClassType("std::initializer_list");

  auto loc = pushBackCallList[0].getLoc();
  auto arrAttr = builder.getArrayAttr(attrElements);
  static unsigned long long strTemp1 = 0;
  std::string name = "__cxx_vector_tmp" + std::to_string(strTemp1++);
  auto init = builder.create<CIL::GlobalOp>(loc, StdInitTy, false, name, arrAttr);

  auto itrTy = builder.getCILClassType("std::__1::__wrap_iter");
  auto itrAlloca = builder.create<CIL::AllocaOp>(loc, "", itrTy);
  auto loadItr = builder.create<CIL::CILLoadOp>(loc, itrAlloca);

  auto vectorClassOp = module.lookupSymbol<CIL::ClassOp>("std::__1::vector");
  auto VectorInsertFuncOp = vectorClassOp.lookupSymbol<mlir::FuncOp>(
      "_ZNSt3__16vectorIiNS_9allocatorIiEEE6insertENS_11__wrap_"
      "iterIPKiEESt16initializer_listIiE");

  auto symRef = builder.getSymbolRefAttr(VectorInsertFuncOp.getName(),
                                         builder.getSymbolRefAttr("vector"));
  SmallVector<mlir::Value, 2> argsList{loadItr, init};
  builder.create<CIL::MemberCallOp>(loc, vectorBase, symRef,
                                    VectorInsertFuncOp.getCallableResults(),
                                    argsList);

  for (auto pushBackCall : pushBackCallList) {
    pushBackCall.erase();
  }

  for (auto arg : pushBackArgs) {
    arg.erase();
  }
}

namespace CIL {
std::unique_ptr<mlir::Pass> createVectorPushBackOptimizerPass() {
  return std::make_unique<VectorPushBackOptimizerPass>();
}
} // namespace CIL
