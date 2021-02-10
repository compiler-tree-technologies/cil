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

#include "clang/cil/codegen/CodeGen.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

using namespace clang;
using namespace clang::mlir_codegen;

static bool isLegalToEmitLoopOp(ForStmt *stmt);

bool MLIRCodeGen::emitIfStmt(IfStmt *stmt) {
  auto cond = stmt->getCond();
  auto thenStmt = stmt->getThen();
  auto elseStmt = stmt->getElse();

  auto currRegion = builder.getBlock()->getParent();
  auto savept = builder.saveInsertionPoint();

  // FIXME: The placement of the basic blocks is not proper. Need to be fixed.
  auto thenBlock = builder.createBlock(currRegion, currRegion->end());
  auto elseBlock = builder.createBlock(currRegion, currRegion->end());
  builder.restoreInsertionPoint(savept);
  SmallVector<mlir::Value, 2> none;
  // auto boolTy = builder.getCILBoolType();
  // condExpr = emitCast(condExpr, boolTy);
  emitCondExprIf(cond, thenBlock, elseBlock);

  // builder.create<CIL::CILIfOp>(loc, condExpr, thenBlock, none, elseBlock,
  // none);

  mlir::Block *exitBlock = nullptr;
  if (!elseStmt)
    exitBlock = elseBlock;
  else {
    exitBlock = builder.createBlock(currRegion, currRegion->end());
  }
  builder.setInsertionPointToStart(thenBlock);
  emitStmt(thenStmt);
  builder.create<mlir::BranchOp>(getLoc(stmt->getEndLoc()), exitBlock);
  if (elseStmt) {
    builder.setInsertionPointToStart(elseBlock);
    emitStmt(elseStmt);
    builder.create<mlir::BranchOp>(getLoc(stmt->getEndLoc()), exitBlock);
  }
  builder.setInsertionPointToStart(exitBlock);
  return true;
}

bool MLIRCodeGen::emitWhileStmt(WhileStmt *stmt) {
  auto cond = stmt->getCond();
  auto loc = getLoc(stmt->getWhileLoc());
  auto body = stmt->getBody();

  auto currRegion = builder.getBlock()->getParent();
  // FIXME: The placement of the basic blocks is not proper.
  // Need to be fixed.
  auto savept = builder.saveInsertionPoint();
  auto headerBlock = builder.createBlock(currRegion, currRegion->end());
  auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
  auto exitBlock = builder.createBlock(currRegion, currRegion->end());
  builder.restoreInsertionPoint(savept);
  builder.create<mlir::BranchOp>(loc, headerBlock);

  builder.setInsertionPointToEnd(headerBlock);
  /*
  auto condExpr = emitExpression(cond);
  condExpr = getBoolTypeFor(condExpr);
  if (auto zext = dyn_cast<CIL::ZeroExtendOp>(condExpr.getDefiningOp())) {
    condExpr = zext.getValue();
    // FIXME: Bug. Make sure the zext isn't generated in the first place.
    zext.erase();
  }

  SmallVector<mlir::Value, 2> none;
  builder.create<CIL::CILWhileOp>(loc, condExpr, bodyBlock, none, exitBlock,
                                  none);
  */
  emitCondExprWhile(cond, bodyBlock, exitBlock);

  builder.setInsertionPointToStart(bodyBlock);
  BreakContinueStack.push_back(std::make_pair(exitBlock, headerBlock));
  emitStmt(body);
  BreakContinueStack.pop_back();
  builder.create<mlir::BranchOp>(getLoc(stmt->getEndLoc()), headerBlock);
  builder.setInsertionPointToStart(exitBlock);
  return true;
}

bool MLIRCodeGen::emitForStmtBB(ForStmt *stmt) {
  auto declStmt = stmt->getConditionVariable();
  auto init = stmt->getInit();
  auto cond = stmt->getCond();
  auto loc = getLoc(stmt->getForLoc());
  auto body = stmt->getBody();
  auto incr = stmt->getInc();

  auto currRegion = builder.getBlock()->getParent();
  // FIXME: The placement of the basic blocks is not proper.
  // Need to be fixed.
  auto savept = builder.saveInsertionPoint();
  auto preHeaderBlock = builder.createBlock(currRegion, currRegion->end());
  auto headerBlock = builder.createBlock(currRegion, currRegion->end());
  auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
  auto latchBlock = builder.createBlock(currRegion, currRegion->end());
  auto exitBlock = builder.createBlock(currRegion, currRegion->end());
  builder.restoreInsertionPoint(savept);

  builder.create<mlir::BranchOp>(loc, preHeaderBlock);
  builder.setInsertionPointToStart(preHeaderBlock);
  if (declStmt) {
    emitDecl(declStmt);
  }
  if (init) {
    emitStmt(init);
  }

  builder.create<mlir::BranchOp>(loc, headerBlock);
  builder.setInsertionPointToEnd(headerBlock);
  if (cond) {
    emitCondExprFor(cond, bodyBlock, exitBlock);
  } else {
    builder.create<mlir::BranchOp>(loc, bodyBlock);
  }

  builder.setInsertionPointToStart(bodyBlock);
  BreakContinueStack.push_back(std::make_pair(exitBlock, latchBlock));
  emitStmt(body);
  BreakContinueStack.pop_back();
  builder.create<mlir::BranchOp>(getLoc(stmt->getEndLoc()), latchBlock);
  builder.setInsertionPointToStart(latchBlock);
  if (incr) {
    emitExpression(incr);
  }
  builder.create<mlir::BranchOp>(getLoc(stmt->getEndLoc()), headerBlock);
  builder.setInsertionPointToStart(exitBlock);
  return true;
}

static void replaceAllUsesInRegionWith(Value orig, Value replacement,
                                       Region &region) {
  for (auto &use : llvm::make_early_inc_range(orig.getUses())) {
    if (region.isAncestor(use.getOwner()->getParentRegion())) {
      if (auto loadOp = dyn_cast<CIL::CILLoadOp>(use.getOwner())) {
        loadOp.replaceAllUsesWith(replacement);
        loadOp.erase();
      } else {
        use.set(replacement);
      }
    }
  }
}

static bool isIllegalStmtInLoop(Stmt *stmt) {
  switch (stmt->getStmtClass()) {
  case Stmt::ReturnStmtClass:
  case Stmt::IfStmtClass:
  case Stmt::WhileStmtClass:
  case Stmt::DoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::SwitchStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::LabelStmtClass:
    return true;
  case Stmt::ForStmtClass:
    return !isLegalToEmitLoopOp(static_cast<clang::ForStmt *>(stmt));
  default:
    return false;
  }
}

static bool containsIllegalStmt(Stmt *stmt) {
  if (stmt->getStmtClass() != Stmt::CompoundStmtClass) {
    return isIllegalStmtInLoop(stmt);
  }

  auto cmpStmt = static_cast<clang::CompoundStmt *>(stmt);
  for (auto subStmt : cmpStmt->body()) {
    if (isIllegalStmtInLoop(subStmt))
      return true;
  }
  return false;
}

static bool isSimpleVarDecl(DeclStmt *stmt) {
  if (!stmt->isSingleDecl())
    return false;

  auto decl = stmt->getSingleDecl();
  if (decl->getKind() != clang::Decl::Var)
    return false;

  return true;
}

static bool isLegalToEmitLoopOp(ForStmt *stmt) {
  auto incr = stmt->getInc();
  auto init = stmt->getInit();
  if (!incr)
    return false;

  if (!init)
    return false;

  if (incr->getStmtClass() != Stmt::UnaryOperatorClass)
    return false;

  if (init->getStmtClass() == Stmt::DeclStmtClass) {
    auto declStmt = static_cast<clang::DeclStmt *>(init);
    if (!isSimpleVarDecl(declStmt))
      return false;
  } else {
    if (init->getStmtClass() != Stmt::BinaryOperatorClass)
      return false;
    auto binOp = static_cast<clang::BinaryOperator *>(init);
    if (binOp->getOpcode() != BO_Assign)
      return false;
  }

  return !containsIllegalStmt(stmt->getBody());
}

bool MLIRCodeGen::emitForStmt(ForStmt *stmt) {
  if (!isLegalToEmitLoopOp(stmt)) {
    return emitForStmtBB(stmt);
  }

  auto declStmt = stmt->getConditionVariable();
  auto init = stmt->getInit();
  auto cond = stmt->getCond();
  auto loc = getLoc(stmt->getForLoc());
  auto body = stmt->getBody();
  auto incr = stmt->getInc();

  if (declStmt) {
    emitDecl(declStmt);
  }

  /*auto binInitOp = dyn_cast<BinaryOperator>(init);
  assert(binInitOp);

  auto dummyIV = emitExpression(binInitOp->getLHS());*/

  // auto I32Ty = builder.getIntegerType(32);

  mlir::Value lbound, dummyIV;

  computeLBoundAndIV(loc, init, lbound, dummyIV);

  /*auto lboundBinOp = dyn_cast<clang::BinaryOperator>(init);
  auto lbound = emitExpression(lboundBinOp->getRHS());*/

  auto uboundBinOp = dyn_cast<clang::BinaryOperator>(cond);
  assert(uboundBinOp);
  auto ubound = emitExpression(uboundBinOp->getRHS());

  auto step = getIncr(loc, incr, ubound.getType());

  auto forLoopOp = builder.create<CIL::ForLoopOp>(loc, lbound, ubound, step,
                                                  lbound.getType());

  builder.setInsertionPointToStart(forLoopOp.getBody());
  emitStmt(body);

  auto IV = forLoopOp.getIndVar();
  replaceAllUsesInRegionWith(dummyIV, IV, forLoopOp.region());

  builder.setInsertionPointAfter(forLoopOp);

  return true;
}

bool MLIRCodeGen::emitDoWhileStmt(DoStmt *stmt) {
  auto cond = stmt->getCond();
  auto loc = getLoc(stmt->getWhileLoc());
  auto body = stmt->getBody();

  auto currRegion = builder.getBlock()->getParent();
  // FIXME: The placement of the basic blocks is not proper.
  // Need to be fixed.
  auto savept = builder.saveInsertionPoint();
  auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
  auto condBlock = builder.createBlock(currRegion, currRegion->end());
  auto exitBlock = builder.createBlock(currRegion, currRegion->end());
  builder.restoreInsertionPoint(savept);
  builder.create<mlir::BranchOp>(loc, bodyBlock);

  builder.setInsertionPointToStart(bodyBlock);
  BreakContinueStack.push_back(std::make_pair(exitBlock, condBlock));
  emitStmt(body);
  BreakContinueStack.pop_back();
  builder.create<mlir::BranchOp>(getLoc(stmt->getEndLoc()), condBlock);

  builder.setInsertionPointToEnd(condBlock);
  auto condExpr = emitExpression(cond);
  if (auto zext = dyn_cast<CIL::ZeroExtendOp>(condExpr.getDefiningOp())) {
    condExpr = zext.getValue();
    // FIXME: Bug. Make sure the zext isn't generated in the first place.
    zext.erase();
  }

  SmallVector<mlir::Value, 2> none;
  builder.create<CIL::CILDoWhileOp>(loc, condExpr, bodyBlock, none, exitBlock,
                                    none);
  builder.setInsertionPointToStart(exitBlock);
  return true;
}

mlir::Value MLIRCodeGen::emitCompoundAssignment(CompoundAssignOperator *E) {
  QualType Ty = E->getType();
  if (const AtomicType *AT = Ty->getAs<AtomicType>())
    Ty = AT->getValueType();
  if (Ty->isAnyComplexType())
    llvm_unreachable("Unhandled\n");

  auto lhsPtr = emitExpression(E->getLHS());
  auto loc = getLoc(E->getSourceRange(), true);
  auto lhs = builder.create<CIL::CILLoadOp>(lhsPtr.getLoc(), lhsPtr);
  auto rhs = emitExpression(E->getRHS());
  mlir::Value result;

  // TODO: Refactor this with emitBinaryOperator
  switch (E->getOpcode()) {
  case BinaryOperatorKind::BO_AddAssign: {
    if (E->getType()->isIntegerType()) {
      result = builder.create<CIL::CILAddIOp>(loc, lhs, rhs);
    } else if (E->getType()->isPointerType()) {
      result = builder.create<CIL::CILPointerAddOp>(loc, lhs, rhs);
    } else if (E->getType()->isFloatingType()) {
      result = builder.create<CIL::CILAddFOp>(loc, lhs, rhs);
    } else {
      llvm_unreachable("Unknown type in CompoundAssignOperator");
    }
    break;
  }
  case BinaryOperatorKind::BO_SubAssign: {
    if (E->getType()->isPointerType()) {
      auto zero = builder.getCILIntegralConstant(loc, rhs.getType(), 0);
      rhs = builder.create<CIL::CILSubIOp>(loc, zero, rhs);
      result = builder.create<CIL::CILPointerAddOp>(loc, lhs, rhs);
    } else if (E->getType()->isIntegerType()) {
      result = builder.create<CIL::CILSubIOp>(loc, lhs, rhs);
    } else if (E->getType()->isFloatingType()) {
      result = builder.create<CIL::CILSubFOp>(loc, lhs, rhs);
    } else {
      llvm_unreachable("Unknown type in CompoundAssignOperator");
    }
    break;
  }
  case BinaryOperatorKind::BO_MulAssign: {
    if (E->getType()->isIntegerType()) {
      result = builder.create<CIL::CILMulIOp>(loc, lhs, rhs);
    } else if (E->getType()->isFloatingType()) {
      result = builder.create<CIL::CILMulFOp>(loc, lhs, rhs);
    } else {
      llvm_unreachable("Unknown type in CompoundAssignOperator");
    }
    break;
  }
  case BinaryOperatorKind::BO_RemAssign: {
    if (E->getType()->isIntegerType()) {
      assert(!E->getType()->isUnsignedIntegerType());
      result = builder.create<CIL::CILModIOp>(loc, lhs, rhs);
    } else if (E->getType()->isFloatingType()) {
      result = builder.create<CIL::CILModFOp>(loc, lhs, rhs);
    } else {
      llvm_unreachable("Unknown type in CompoundAssignOperator");
    }
    break;
  }
  case BinaryOperatorKind::BO_DivAssign: {
    if (E->getType()->isIntegerType()) {
      result = builder.create<CIL::CILDivIOp>(loc, lhs, rhs);
    } else if (E->getType()->isFloatingType()) {
      result = builder.create<CIL::CILDivFOp>(loc, lhs, rhs);
    } else {
      llvm_unreachable("Unknown type in CompoundAssignOperator");
    }
    break;
  }
  case BinaryOperatorKind::BO_OrAssign: {
    if (E->getType()->isIntegerType()) {
      result = builder.create<CIL::CILOrOp>(loc, lhs, rhs);
    } else {
      llvm_unreachable("Unknown type in CompoundAssignOperator");
    }
    break;
  }
  default:
    llvm_unreachable("Handle CompoundAssignOperator");
  }
  builder.create<CIL::CILStoreOp>(loc, result, lhsPtr);
  return builder.create<CIL::CILLoadOp>(lhsPtr.getLoc(), lhsPtr);
}

// TODO: lowering this to completely CFG base implementation for now.
bool MLIRCodeGen::emitSwitchStmt(SwitchStmt *stmt) {
  auto var = emitExpression(stmt->getCond());
  auto body = stmt->getBody();
  auto loc = getLoc(stmt->getEndLoc());
  mlir::Block *defaultBlock = nullptr;
  auto compoundStmt = dyn_cast<CompoundStmt>(body);
  assert(compoundStmt);
  auto exitBlock = getNewBlock();
  auto saveptr = builder.saveInsertionPoint();
  SmallVector<Stmt *, 2> stmts(compoundStmt->body());
  SmallVector<mlir::Block *, 2> blocks;
  for (auto I = 0; I < stmts.size(); ++I) {
    blocks.push_back(getNewBlock(exitBlock));
  }
  BreakContinueStack.push_back(std::make_pair(exitBlock, nullptr));
  typedef SmallVector<Expr *, 2> ExprList;
  SmallVector<std::pair<ExprList, mlir::Block *>, 2> caseBlocks;
  int defaultFallThru = -1;

  for (auto I = 0; I < stmts.size(); ++I) {
    auto stmt = stmts[I];
    builder.setInsertionPointToStart(blocks[I]);

    if (auto breakStmt = isa<BreakStmt>(stmt)) {
      builder.create<mlir::BranchOp>(loc, exitBlock);
      continue;
    }
    auto nextBlock = exitBlock;
    if (I < stmts.size() - 1) {
      nextBlock = blocks[I + 1];
    }
    if (auto caseStmt = dyn_cast<CaseStmt>(stmt)) {
      ExprList exprList;
      exprList.push_back(caseStmt->getLHS());
      CaseStmt *CurCase = caseStmt;
      CaseStmt *NextCase = dyn_cast<CaseStmt>(caseStmt->getSubStmt());

      while (NextCase && NextCase->getRHS() == nullptr) {
        CurCase = NextCase;
        exprList.push_back(CurCase->getLHS());
        NextCase = dyn_cast<CaseStmt>(CurCase->getSubStmt());
      }

      Stmt *caseBody;
      if (auto defaultStmt = dyn_cast<DefaultStmt>(CurCase->getSubStmt())) {
        assert(defaultFallThru == -1 && "Multiple defaultFallThru");
        defaultFallThru = caseBlocks.size();
        caseBody = defaultStmt->getSubStmt();
      } else
        caseBody = CurCase->getSubStmt();
      emitStmt(caseBody);
      caseBlocks.push_back(std::make_pair(exprList, blocks[I]));
      builder.create<mlir::BranchOp>(getLoc(caseStmt->getSourceRange()),
                                     nextBlock);
      continue;
    }
    if (auto defaultStmt = dyn_cast<DefaultStmt>(stmt)) {
      assert(!defaultBlock);
      auto defBody = defaultStmt->getSubStmt();
      emitStmt(defBody);
      defaultBlock = blocks[I];
      builder.create<mlir::BranchOp>(getLoc(defaultStmt->getSourceRange()),
                                     nextBlock);
      continue;
    }
    llvm_unreachable("unknown stmt in switch statement");
  }
  BreakContinueStack.pop_back();
  builder.restoreInsertionPoint(saveptr);
  auto insertBefore = caseBlocks.front().second;
  for (unsigned II = 0; II < caseBlocks.size(); ++II) {
    auto condBlockPair = caseBlocks[II];
    auto exprList = condBlockPair.first;
    assert(exprList.size() > 0);
    auto block = condBlockPair.second;
    assert(block);
    auto condExpr = emitExpression(exprList[0]);
    auto loc = getLoc(exprList[0]->getBeginLoc());
    condExpr =
        builder.create<CIL::CILCmpIOp>(loc, CmpIPredicate::eq, condExpr, var);
    for (unsigned k = 1; k < exprList.size(); ++k) {
      auto currExprVal = emitExpression(exprList[k]);
      auto currCond = builder.create<CIL::CILCmpIOp>(
          currExprVal.getLoc(), CmpIPredicate::eq, currExprVal, var);
      condExpr = builder.create<CIL::CILLogicalOrOp>(currExprVal.getLoc(),
                                                     condExpr, currCond);
    }

    if (defaultFallThru == II) {
      auto trueVal = builder.getCILBoolConstant(loc, 1);
      condExpr = builder.create<CIL::CILLogicalOrOp>(loc, condExpr, trueVal);
    }

    mlir::Block *nextBlock = nullptr;
    if (II == caseBlocks.size() - 1) {
      if (defaultBlock) {
        nextBlock = defaultBlock;
      } else {
        nextBlock = exitBlock;
      }
    } else {
      nextBlock = getNewBlock(insertBefore);
    }
    SmallVector<mlir::Value, 2> noValues;
    builder.create<CIL::CILIfOp>(loc, condExpr, block, noValues, nextBlock,
                                 noValues);
    builder.setInsertionPointToStart(nextBlock);
  }
  builder.setInsertionPointToEnd(exitBlock);
  return true;
}

bool MLIRCodeGen::emitStmt(Stmt *stmt) {
  switch (stmt->getStmtClass()) {
  case Stmt::CompoundStmtClass: {
    auto cmpStmt = static_cast<clang::CompoundStmt *>(stmt);
    for (auto stmt : cmpStmt->body()) {
      if (!emitStmt(stmt)) {
        return false;
      }
    }
    return true;
  }
  case Stmt::ParenExprClass: {
    emitExpression(static_cast<ParenExpr *>(stmt));
    return true;
  }
  case Stmt::DeclStmtClass: {
    auto declStmt = static_cast<clang::DeclStmt *>(stmt);
    return emitDeclStmt(declStmt);
  }
  case Stmt::BinaryOperatorClass: {
    auto binOperator = static_cast<clang::BinaryOperator *>(stmt);
    emitBinaryOperator(binOperator);
    return true;
  }
  case Stmt::UnaryOperatorClass: {
    auto unaryOp = static_cast<clang::UnaryOperator *>(stmt);
    emitUnaryOperator(unaryOp);
    return true;
  }
  case Stmt::ReturnStmtClass: {
    auto returnStmt = static_cast<clang::ReturnStmt *>(stmt);
    assert(returnStmt);
    auto loc = getLoc(stmt->getSourceRange());
    if (returnStmt->getRetValue()) {
      auto returnVal = emitExpression(returnStmt->getRetValue());
      if (!returnVal) {
        builder.create<mlir::ReturnOp>(loc);
      } else {

        auto currFunc =
            returnVal.getParentRegion()->getParentOfType<mlir::FuncOp>();
        assert(currFunc);
        if (returnVal.getType().isa<CIL::PointerType>() &&
            currFunc.getType().getResult(0) != returnVal.getType()) {
          returnVal =
              emitPointerRefCast(returnVal, currFunc.getType().getResult(0));
        }
        returnVal = emitCast(returnVal, currFunc.getType().getResult(0));
        builder.create<mlir::ReturnOp>(loc, returnVal);
      }
    } else {
      builder.create<mlir::ReturnOp>(loc);
    }
    // TODO: always create a new block.
    // FIXME: do not create new block if there are no further statements.
    builder.setInsertionPointToStart(getNewBlock());
    return true;
  } break;
  case Stmt::CallExprClass: {
    emitCallExpr(static_cast<clang::CallExpr *>(stmt));
    return true;
  }
  case Stmt::CXXMemberCallExprClass: {
    emitCXXMemberCall(static_cast<clang::CXXMemberCallExpr *>(stmt));
    return true;
  }
  case Stmt::CXXOperatorCallExprClass: {
    auto operatorCall = static_cast<clang::CXXOperatorCallExpr *>(stmt);
    emitCallExpr(operatorCall);
    return true;
  }
  case Stmt::MemberExprClass: {
    emitMemberExpr(static_cast<clang::MemberExpr *>(stmt));
    return true;
  }
  case Stmt::IfStmtClass: {
    return emitIfStmt(static_cast<clang::IfStmt *>(stmt));
  }
  case Stmt::WhileStmtClass: {
    return emitWhileStmt(static_cast<clang::WhileStmt *>(stmt));
  }
  case Stmt::DoStmtClass: {
    return emitDoWhileStmt(static_cast<clang::DoStmt *>(stmt));
  }
  case Stmt::ForStmtClass: {
    return emitForStmt(static_cast<clang::ForStmt *>(stmt));
  }
  case Stmt::BreakStmtClass: {
    auto brStmt = static_cast<clang::BreakStmt *>(stmt);
    builder.create<mlir::BranchOp>(getLoc(brStmt->getSourceRange()),
                                   BreakContinueStack.back().first);
    auto region = builder.getBlock()->getParent();
    auto block = builder.createBlock(region, region->end());
    builder.setInsertionPointToStart(block);
    return true;
  }
  case Stmt::ContinueStmtClass: {
    auto brStmt = static_cast<clang::BreakStmt *>(stmt);
    builder.create<mlir::BranchOp>(getLoc(brStmt->getSourceRange()),
                                   BreakContinueStack.back().second);
    auto region = builder.getBlock()->getParent();
    auto block = builder.createBlock(region, region->end());
    builder.setInsertionPointToStart(block);
    return true;
  }
  case Stmt::SwitchStmtClass: {
    return emitSwitchStmt(static_cast<SwitchStmt *>(stmt));
  }
  case Stmt::NullStmtClass: {
    return true;
  }
  case Stmt::CompoundAssignOperatorClass: {
    return emitCompoundAssignment(static_cast<CompoundAssignOperator *>(stmt));
  }
  case Stmt::CXXConstructExprClass: {
    auto constructExpr = static_cast<clang::CXXConstructExpr *>(stmt);
    auto CD = constructExpr->getConstructor();
    if (CD->isImplicit()) {
      break;
    }
    constructExpr->dump();
    CD->dump();
    llvm_unreachable("Construct expression should not reach in emitStmt!");
  }
  case Stmt::ExprWithCleanupsClass: {
    auto castExpr = static_cast<clang::ExprWithCleanups *>(stmt);
    emitExpression(castExpr->getSubExpr());
    return true;
  }
  case Stmt::CXXDeleteExprClass: {
    auto delExpr = static_cast<clang::CXXDeleteExpr *>(stmt);
    auto loc = getLoc(delExpr->getBeginLoc());
    auto newFn = delExpr->getOperatorDelete();
    auto func = emitFunction(newFn);
    auto expr = emitExpression(delExpr->getArgument());
    auto toType = func.getType().getInput(0);
    auto bitcast = builder.create<CIL::PointerBitCastOp>(loc, toType, expr);
    builder.create<CIL::CILCallOp>(loc, func, bitcast.getResult());
    return true;
  }
  case Stmt::CXXNewExprClass: {
    emitCXXNewExpr(static_cast<clang::CXXNewExpr *>(stmt));
    return true;
  }
  case Stmt::CXXTryStmtClass: {
    auto tryStmt = static_cast<clang::CXXTryStmt *>(stmt);
    emitStmt(tryStmt->getTryBlock());
    return true;
  }
  // TODO: Should be handled later.
  case Stmt::CXXThrowExprClass: {
    return true;
  }
  case Stmt::GotoStmtClass: {
    emitGotoStmt(static_cast<clang::GotoStmt *>(stmt));
    return true;
  }
  case Stmt::LabelStmtClass: {
    emitLabelStmt(static_cast<clang::LabelStmt *>(stmt));
    return true;
  }
  default:
    auto loc = getLoc(stmt->getSourceRange());
    loc.dump();
    stmt->dump();
    llvm_unreachable("unahandled statement");
  };
  return true;
}

mlir::Value MLIRCodeGen::emitPointerRefCast(mlir::Value from, mlir::Type to) {
  assert(to.isa<CIL::PointerType>());
  assert(from.getType().isa<CIL::PointerType>());
  if (from.getType() == to) {
    return from;
  }
  return builder.create<CIL::PointerBitCastOp>(from.getLoc(), to, from);
}
bool MLIRCodeGen::emitGotoStmt(GotoStmt *stmt) {
  auto labelDecl = stmt->getLabel();
  auto destBB = getGotoDestBB(labelDecl);
  auto loc = getLoc(stmt->getSourceRange());
  builder.create<mlir::BranchOp>(loc, destBB);
  auto nextBB = getNewBlock();
  builder.setInsertionPointToStart(nextBB);
  return true;
}

bool MLIRCodeGen::emitLabelStmt(LabelStmt *stmt) {
  auto loc = getLoc(stmt->getSourceRange());
  auto BB = getGotoDestBB(stmt->getDecl());
  builder.create<mlir::BranchOp>(loc, BB);
  builder.setInsertionPointToStart(BB);
  emitStmt(stmt->getSubStmt());
  return true;
}
