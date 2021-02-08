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

//===-----------------------------DCE.cpp---------------------------------===//
//
// This file implements simple dead code elemination
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

struct DeadCodeEliminator : public ModulePass<DeadCodeEliminator> {

  virtual void runOnModule();
};

void DeadCodeEliminator::runOnModule() {
  auto module = getModule();

  llvm::SmallVector<std::string, 2> functions;
  module.walk(
      [&](mlir::FuncOp op) { functions.push_back(op.getName().str()); });

  llvm::SmallVector<std::string, 2> deadFunctions;
  mlir::Region *region = &module.getBodyRegion();
  for (auto func : functions) {
    if (func == "main")
      continue;

    Optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(func, region);
    if (!uses) {
      deadFunctions.push_back(func);
      continue;
    }

    unsigned numUses = llvm::size(*uses);
    if (numUses == 0)
      deadFunctions.push_back(func);
  }

  for (auto deadFuncName : deadFunctions) {
    llvm::errs() << "Removing dead function: " << deadFuncName << "\n";
    auto func = module.lookupSymbol<mlir::FuncOp>(deadFuncName);
    assert(func);
    func.erase();
  }
}

namespace CIL {
std::unique_ptr<mlir::Pass> createDCEPass() {
  return std::make_unique<DeadCodeEliminator>();
}
} // namespace CIL

static PassRegistration<DeadCodeEliminator> pass("dce",
                                                 "Eliminate dead functions");
