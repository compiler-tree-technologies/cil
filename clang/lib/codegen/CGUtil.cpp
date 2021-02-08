#include "clang/cil/codegen/CodeGen.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

using namespace clang;
using namespace clang::mlir_codegen;

mlir::Attribute MLIRCodeGen::emitValueAttrForGlobal(mlir::Location loc,
                                                    clang::Expr *initExpr,
                                                    mlir::Type type) {
  if (!isa_and_nonnull<InitListExpr>(initExpr))
    return {};

  Attribute valueAttr;
  auto initListExpr = cast<InitListExpr>(initExpr);
  if (initListExpr->getType()->isArrayType() &&
      initListExpr->getType()
          ->getArrayElementTypeNoTypeQual()
          ->isIntegerType()) {
    SmallVector<int, 2> values;
    for (auto II = 0; II < initListExpr->getNumInits(); ++II) {
      auto initVal = cast<IntegerLiteral>(initListExpr->getInit(II));
      values.push_back(initVal->getValue().getSExtValue());
    }
    valueAttr = builder.getI32ArrayAttr(values);
  } else if (initListExpr->getType()->isArrayType() &&
             initListExpr->getType()
                 ->getArrayElementTypeNoTypeQual()
                 ->isConstantArrayType()) {
    SmallVector<Attribute, 2> values;
    unsigned dimSize;
    for (auto II = 0; II < initListExpr->getNumInits(); ++II) {
      auto initVal = cast<StringLiteral>(initListExpr->getInit(II));
      auto arrTy = dyn_cast<clang::ConstantArrayType>(initVal->getType());
      assert(arrTy);
      dimSize = arrTy->getSize().getSExtValue();
      std::string constString = initVal->getString().str();
      auto stringSize = constString.size();
      for (unsigned k = 0; k < dimSize - stringSize; ++k)
        constString += '\0';
      values.push_back(builder.getStringAttr(constString));
    }
    auto arrayTy = type.dyn_cast_or_null<CIL::ArrayType>();
    assert(arrayTy);
    auto shape = arrayTy.getShape();
    assert(shape.size() > 0);
    auto numElements = shape[0];
    auto remainingInits = numElements - values.size();
    for (unsigned k = 0; k < remainingInits; ++k) {
      values.push_back(builder.getStringAttr(std::string(dimSize, '\0')));
    }
    assert(numElements == values.size());
    valueAttr = builder.getArrayAttr(values);
  } else {
    initExpr->dump();
    llvm_unreachable("unknown global constant array type");
  }
  return valueAttr;
}

void MLIRCodeGen::setGlobalInitializerRegion(CIL::GlobalOp op,
                                             clang::Expr *initExpr) {
  auto savept = builder.saveInsertionPoint();

  auto block = op.getInitializerBlock();
  if (!block) {
    block = builder.createBlock(&op.getInitializerRegion());
  }
  builder.setInsertionPointToStart(block);
  auto expr = emitExpression(initExpr);
  builder.create<CIL::CILReturnOp>(expr.getLoc(), expr);
  builder.restoreInsertionPoint(savept);
}

mlir::Value MLIRCodeGen::emitGlobalVar(mlir::Location loc, clang::VarDecl *decl,
                                       mlir::Type type) {

  CIL::GlobalOp globalOp;
  std::string mangledName;
  if (decl->getName() == "stdout" || decl->getName() == "stderr")
    mangledName = decl->getName().str();
  else
    mangledName = mangler.mangleGlobalName(decl);
  globalOp = TheModule->lookupSymbol<CIL::GlobalOp>(mangledName);
  if (globalOp) {
    assert(type == globalOp.getType() && "Duplicate global op symbol?");
    return builder.create<CIL::GlobalAddressOfOp>(loc, globalOp);
  }

  auto initExpr = decl->getInit();
  Attribute valueAttr;

  // Dump the class record declaration if not already done.
  if (type.isa<CIL::ClassType>()) {
    auto recordDecl =
        cast<clang::RecordDecl>(decl->getType()->getAsRecordDecl());
    emitClassDecl(cast<CXXRecordDecl>(recordDecl));
  }
  auto constructExpr = dyn_cast_or_null<CXXConstructExpr>(initExpr);
  if (constructExpr) {
    globalOp = emitGlobalClassVar(loc, decl, type, constructExpr);
  } else {
    valueAttr = emitValueAttrForGlobal(loc, decl->getInit(), type);
    auto savePoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(TheModule->getBody());
    globalOp =
        builder.create<CIL::GlobalOp>(loc, type, false, mangledName, valueAttr);
    if (initExpr && !valueAttr) {
      setGlobalInitializerRegion(globalOp, initExpr);
    }
    builder.restoreInsertionPoint(savePoint);
  }

  if (decl->hasExternalStorage())
    globalOp.setAttr("external", builder.getBoolAttr(true));

  return builder.create<CIL::GlobalAddressOfOp>(loc, globalOp);
}

// Returns block at the end of the current builder region by default
mlir::Block *MLIRCodeGen::getNewBlock(mlir::Block *insertBefore) {
  auto savePoint = builder.saveInsertionPoint();
  mlir::Block *newBlock = nullptr;
  if (insertBefore) {
    newBlock = builder.createBlock(insertBefore);
  } else {
    auto region = builder.getBlock()->getParent();
    newBlock = builder.createBlock(region, region->end());
  }
  builder.restoreInsertionPoint(savePoint);
  return newBlock;
}

mlir::Block *MLIRCodeGen::getGotoDestBB(LabelDecl *label) {
  if (LabelMap.find(label) != LabelMap.end())
    return LabelMap[label];

  auto newBlock = getNewBlock();
  LabelMap[label] = newBlock;
  return newBlock;
}

// TODO: Add support to decrements
mlir::Value MLIRCodeGen::getIncr(mlir::Location loc, Stmt *IncrStmt,
                                 mlir::Type I32Ty) {
  if (auto unaryOp = dyn_cast<UnaryOperator>(IncrStmt)) {
    switch (unaryOp->getOpcode()) {
    case UnaryOperatorKind::UO_PostInc:
    case UnaryOperatorKind::UO_PreInc: {
      auto valueAttr = builder.getI32IntegerAttr(1);
      return builder.create<CIL::CILConstantOp>(loc, I32Ty, valueAttr);
    }
    case UnaryOperatorKind::UO_PostDec:
    case UnaryOperatorKind::UO_PreDec: {
      auto valueAttr = builder.getI32IntegerAttr(-1);
      return builder.create<CIL::CILConstantOp>(loc, I32Ty, valueAttr);
    }
    default: {
      llvm_unreachable("unknown unaryop in IV increments");
    }
    }
  }

  if (auto binOp = dyn_cast<BinaryOperator>(IncrStmt)) {
    assert(binOp->getOpcode() == BinaryOperatorKind::BO_Assign &&
           "has to be assignment");
    auto incr = dyn_cast<BinaryOperator>(binOp->getRHS());
    assert(incr && "RHS has to be some kind of binary op");
    auto incrOp = incr->getOpcode();
    assert(incrOp == BinaryOperatorKind::BO_Add && "decrements unhandled now");

    return emitExpression(incr->getRHS());
  }

  return {};
}

void MLIRCodeGen::computeLBoundAndIV(mlir::Location loc, Stmt *init,
                                     mlir::Value &lbound, mlir::Value &IV) {
  if (auto binInitOp = dyn_cast<BinaryOperator>(init)) {
    IV = emitExpression(binInitOp->getLHS());
    lbound = emitExpression(binInitOp->getRHS());
  }

  if (auto declStmt = dyn_cast<clang::DeclStmt>(init)) {
    for (auto decl : declStmt->decls()) {
      if (auto varDecl = dyn_cast<clang::VarDecl>(decl)) {
        auto ty = cgTypes.convertClangType(varDecl->getType());
        auto alloc = builder.create<CIL::AllocaOp>(loc, varDecl->getName(), ty);
        assert(varDecl->getInit() && "varDecl without init in for loop");
        lbound = emitExpression(varDecl->getInit());

        auto namedDecl = varDecl->getUnderlyingDecl();
        allocaMap[namedDecl] = alloc.getResult();
        IV = alloc;
      }
    }
  }
}
