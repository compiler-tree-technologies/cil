#include "clang/cil/codegen/CodeGen.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

#include <iostream>

using namespace clang;
using namespace clang::mlir_codegen;

mlir::Value MLIRCodeGen::emitFieldAccessOp(clang::FieldDecl *field,
                                           clang::SourceLocation clangLoc,
                                           clang::Expr *baseExpr) {

  assert(field);
  auto recordDecl = field->getParent();
  assert(isa<CXXRecordDecl>(recordDecl));
  // Emit the c++ class if not already done.
  emitClassDecl(cast<CXXRecordDecl>(recordDecl));
  auto loc = getLoc(clangLoc);

  mlir::Value baseValue;
  if (baseExpr) {

    baseValue = emitExpression(baseExpr);
  } else {
    auto name = mangler.mangleClassName(cast<CXXRecordDecl>(recordDecl));
    auto classType = builder.getCILPointerType(builder.getCILClassType(name));
    baseValue = builder.create<CIL::ThisOp>(loc, classType);
  }
  assert(baseValue);
  auto basePtrType = baseValue.getType().cast<CIL::PointerType>();
  auto resType = builder.getCILPointerType(
      cgTypes.convertClangType(field->getType()), basePtrType.isReference());

  auto className = builder.getUnderlyingType(baseValue.getType())
                       .cast<CIL::ClassType>()
                       .getName();
  // auto classSym = TheModule->lookupSymbol(className);
  auto symRefAttr = builder.getSymbolRefAttr(
      field->getName(), builder.getSymbolRefAttr(className));
  auto structEle =
      builder.create<CIL::FieldAccessOp>(loc, resType, baseValue, symRefAttr);
  return structEle;
}

bool MLIRCodeGen::emitClassDecl(CXXRecordDecl *classDecl) {

  auto className = mangler.mangleClassName(classDecl);
  auto loc = getLoc(classDecl->getBeginLoc());
  if (auto classOp = TheModule->lookupSymbol<CIL::ClassOp>(className)) {
    return true;
  }

  if (!classDecl->hasDefinition()) {
    auto savept = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(TheModule->getBody());
    builder.create<CIL::ClassOp>(loc, className);
    builder.restoreInsertionPoint(savept);
    return true;
  }

  // First emit all the base classes.
  SmallVector<CXXRecordDecl *, 2> classesToDecl;

  SmallVector<mlir::Attribute, 2> baseList;
  for (auto baseClass : classDecl->bases()) {
    auto baseType = baseClass.getType();
    auto mlirType = cgTypes.convertClangType(baseType);
    baseList.push_back(mlir::TypeAttr::get(mlirType));
    auto baseClassDecl = baseType->getAsCXXRecordDecl();
    classesToDecl.push_back(baseClassDecl);
  }
  auto baseClasses = builder.getArrayAttr(baseList);

  auto savept = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(TheModule->getBody());
  auto mlirClass = builder.create<CIL::ClassOp>(loc, className, baseClasses);
  assert(mlirClass);

  builder.setInsertionPointToStart(&mlirClass.body().front());

  // Declare class member variables.
  for (auto field : classDecl->fields()) {
    auto type = cgTypes.convertClangType(field->getType());
    auto underlyingType = field->getType()->getUnqualifiedDesugaredType();
    while (underlyingType != underlyingType->getPointeeOrArrayElementType())
      underlyingType = underlyingType->getPointeeOrArrayElementType();
    auto newDecl = underlyingType->getAsCXXRecordDecl();
    if (newDecl) {
      classesToDecl.push_back(newDecl);
    }
    auto loc = getLoc(field->getSourceRange());
    auto name = field->getName();
    builder.create<CIL::FieldDeclOp>(loc, type, name);
  }

  // Declare clas member functions.
  for (auto methodDecl : classDecl->methods()) {
    if (methodDecl->isImplicit()) {
      continue;
    }
    DeferredDecls.insert(methodDecl);
  }
  builder.restoreInsertionPoint(savept);

  // Remove from deferred declarations
  DeferredDecls.erase(classDecl);

  // Now emit all the classes to declare.
  for (auto decl : classesToDecl) {
    emitClassDecl(decl);
  }
  return true;
}

CIL::GlobalOp MLIRCodeGen::emitGlobalClassVar(mlir::Location loc,
                                              clang::VarDecl *decl,
                                              mlir::Type type,
                                              CXXConstructExpr *constructExpr) {
  assert(constructExpr);
  // Construct expressions needs to know the base expression.
  auto CD = constructExpr->getConstructor();
  emitCXXMethodDecl(CD);

  assert(constructExpr->getNumArgs() == 0);

  auto classDecl = cast<CXXRecordDecl>(CD->getParent());
  emitClassDecl(classDecl);

  auto mangledName = mangler.mangleFunctionName(CD);
  auto symRefAttr = builder.getSymbolRefAttr(
      mangledName, builder.getSymbolRefAttr(classDecl->getName()));

  auto savePoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(TheModule->getBody());
  auto mangledGlobalName = mangler.mangleGlobalName(decl);
  auto global = builder.create<CIL::GlobalOp>(loc, type, false,
                                              mangledGlobalName, symRefAttr);
  builder.restoreInsertionPoint(savePoint);
  return global;
}

CIL::AllocaOp MLIRCodeGen::emitClassTypeAlloca(mlir::Location loc,
                                               StringRef name, mlir::Type type,
                                               clang::Expr *initExpr) {

  auto constructExpr = dyn_cast_or_null<CXXConstructExpr>(initExpr);
  if (!constructExpr || constructExpr->getConstructor()->isImplicit()) {
    return builder.create<CIL::AllocaOp>(loc, name, type);
  }

  assert(constructExpr);
  // Construct expressions needs to know the base expression.
  auto CD = constructExpr->getConstructor();
  emitCXXMethodDecl(CD);

  SmallVector<mlir::Value, 2> argList;
  for (auto II = 0; II < constructExpr->getNumArgs(); ++II) {
    argList.push_back(emitExpression(constructExpr->getArg(II)));
  }

  auto classDecl = cast<CXXRecordDecl>(CD->getParent());
  emitClassDecl(classDecl);
  auto mangledName = mangler.mangleFunctionName(CD);
  auto symRefAttr = builder.getSymbolRefAttr(
      mangledName, builder.getSymbolRefAttr(classDecl->getName()));
  return builder.create<CIL::AllocaOp>(loc, name, type, symRefAttr, argList);
}

CIL::MemberCallOp MLIRCodeGen::emitCXXConstructExpr(CXXConstructExpr *expr,
                                                    mlir::Value base) {

  auto CD = expr->getConstructor();
  if (CD->isImplicit()) {
    return {};
  }
  auto classDecl = cast<CXXRecordDecl>(CD->getParent());
  emitClassDecl(classDecl);

  SmallVector<clang::Expr *, 2> argList;
  for (auto II = 0; II < expr->getNumArgs(); ++II) {
    argList.push_back(expr->getArg(II));
  }
  return emitMemberCallOp(base, classDecl, CD, argList);
}

CIL::MemberCallOp
MLIRCodeGen::emitMemberCallOp(mlir::Value base, CXXRecordDecl *classDecl,
                              CXXMethodDecl *methodDecl,
                              SmallVector<clang::Expr *, 2> argsList) {
  emitCXXMethodDecl(methodDecl);

  auto loc = base.getLoc();
  SmallVector<mlir::Value, 2> argList;
  for (auto II = 0; II < argsList.size(); ++II) {
    argList.push_back(emitExpression(argsList[II]));
  }
  auto mangledName = mangler.mangleFunctionName(methodDecl);
  auto symRefAttr = builder.getSymbolRefAttr(
      mangledName, builder.getSymbolRefAttr(classDecl->getName()));
  SmallVector<mlir::Type, 2> returnTypes;
  if (!methodDecl->getReturnType()->isVoidType()) {
    returnTypes.push_back(
        cgTypes.convertClangType(methodDecl->getReturnType()));
  }
  return builder.create<CIL::MemberCallOp>(loc, base, symRefAttr, returnTypes,
                                           argList);
}

CIL::MemberCallOp MLIRCodeGen::emitCXXMemberCall(CXXMemberCallExpr *expr) {
  auto member = cast<MemberExpr>(expr->getCallee());
  auto methodDecl = expr->getMethodDecl();
  // Try to emit the cxx method declar

  auto classDecl = expr->getRecordDecl();
  assert(classDecl);
  emitClassDecl(cast<CXXRecordDecl>(classDecl));

  SmallVector<clang::Expr *, 2> argList;
  for (auto II = 0; II < expr->getNumArgs(); ++II) {
    argList.push_back((expr->getArg(II)));
  }
  return emitMemberCallOp(emitExpression(member->getBase()), classDecl,
                          methodDecl, argList);
}

bool MLIRCodeGen::emitConstructorInitalizers(CXXConstructorDecl *decl) {
  for (auto init : decl->inits()) {
    auto loc = getLoc(decl->getSourceRange());

    mlir::Value basePointer;
    if (auto baseClassType = init->getBaseClass()) {
      auto thisType = cgTypes.convertClangType(decl->getThisType());
      auto thisPtr = builder.create<CIL::ThisOp>(loc, thisType);
      auto baseClassPointer =
          builder.getCILPointerType(cgTypes.convertClangType(baseClassType));
      basePointer = builder.create<CIL::DerivedToBaseCastOp>(
          loc, baseClassPointer, thisPtr);
    } else if (auto fieldDecl = init->getMember()) {
      basePointer = emitFieldAccessOp(fieldDecl, decl->getEndLoc(), nullptr);
    } else {
      llvm_unreachable("Unhandled initializer kind");
    }
    if (auto constructExpr = dyn_cast<CXXConstructExpr>(init->getInit())) {
      auto constructor = constructExpr->getConstructor();
      auto func = emitCXXMethodDecl(constructor);
      builder.create<CIL::MemberCallOp>(
          loc, basePointer, builder.getSymbolRefAttr(func), llvm::None);
      continue;
    }
    auto exprVal = emitExpression(init->getInit());
    builder.create<CIL::CILStoreOp>(loc, exprVal, basePointer);
  }
  return true;
}