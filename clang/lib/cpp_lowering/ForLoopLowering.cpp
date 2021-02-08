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
//===- LoopStructureLoweringPass.cpp - convert Loops to CFG  --------------===//
//
//===----------------------------------------------------------------------===//
//
// Lowers all loop like operations to CFG based loops.
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "clang/cil/dialect/CIL/CILOps.h"

using namespace std;

namespace mlir {

struct ForLoopTermLoweringPattern
    : public OpRewritePattern<CIL::ForLoopTerminatorOp> {
public:
  using OpRewritePattern<CIL::ForLoopTerminatorOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(CIL::ForLoopTerminatorOp termOp,
                                     PatternRewriter &rewriter) const override {
    rewriter.eraseOp(termOp);
    return matchSuccess();
  }
}; // namespace mlir

struct ForLoopOpLoweringPattern : public OpRewritePattern<CIL::ForLoopOp> {
public:
  using OpRewritePattern<CIL::ForLoopOp>::OpRewritePattern;

  /// Performs the rewrite.
  PatternMatchResult matchAndRewrite(CIL::ForLoopOp doOp,
                                     PatternRewriter &rewriter) const override;
}; // namespace mlir

PatternMatchResult
ForLoopOpLoweringPattern::matchAndRewrite(CIL::ForLoopOp doOp,
                                          PatternRewriter &rewriter) const {

  auto loc = doOp.getLoc();

  auto lb = (doOp.getLowerBound());
  auto ub = (doOp.getUpperBound());
  auto step = (doOp.getStep());
  auto hasExpr = lb && ub && step;
  if (hasExpr && (!lb || !ub || !step))
    return matchFailure();

  // prepare LLVM like loop structure.
  auto *preheader = rewriter.getInsertionBlock();
  auto initPosition = rewriter.getInsertionPoint();
  auto *exitBlock = rewriter.splitBlock(preheader, initPosition);
  auto *header = &doOp.region().front();
  auto *firstBodyBlock = rewriter.splitBlock(header, header->begin());
  auto *lastBlock = &doOp.region().back();
  auto *latch =
      rewriter.splitBlock(lastBlock, lastBlock->getTerminator()->getIterator());
  rewriter.setInsertionPointToEnd(lastBlock);
  rewriter.create<BranchOp>(loc, latch);

  rewriter.inlineRegionBefore(doOp.region(), exitBlock);
  auto iv = header->getArgument(0);

  // Build preheader.
  rewriter.setInsertionPointToEnd(preheader);
  rewriter.create<BranchOp>(loc, header, ArrayRef<mlir::Value>({lb}));

  // Build Latch condition.
  rewriter.setInsertionPointToEnd(latch);
  auto stepValue = rewriter.create<CIL::CILAddIOp>(loc, iv, step).getResult();
  rewriter.create<BranchOp>(loc, header, ArrayRef<mlir::Value>({stepValue}));

  // Build header.
  rewriter.setInsertionPointToEnd(header);
  auto comparison =
      rewriter.create<CIL::CILCmpIOp>(loc, CmpIPredicate::slt, iv, ub);
  rewriter.create<CIL::CILForOp>(loc, comparison, firstBodyBlock, llvm::None,
                                 exitBlock, llvm::None);

  rewriter.eraseOp(doOp);
  return matchSuccess();
}

struct LoopStructureLoweringPass
    : public OperationPass<LoopStructureLoweringPass, mlir::FuncOp> {
  virtual void runOnOperation() {
    auto M = getOperation();

    OwningRewritePatternList patterns;

    patterns.insert<ForLoopOpLoweringPattern>(&getContext());
    patterns.insert<ForLoopTermLoweringPattern>(&getContext());

    applyPatternsGreedily(M, patterns);
  }
};
} // namespace mlir

namespace CIL {
/// Create a LoopTransform pass.
std::unique_ptr<Pass> createLoopStructureLoweringPass() {
  return std::make_unique<mlir::LoopStructureLoweringPass>();
}
} // namespace CIL
