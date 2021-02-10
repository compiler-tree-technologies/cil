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

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"

#include "clang/AST/Mangle.h"

using namespace clang;

using namespace clang::mlir_codegen;

MLIRCodeGen::MLIRCodeGen(mlir::MLIRContext &mlirContext,
                         mlir::OwningModuleRef &TheModule,
                         clang::ASTContext &context)
    : mlirContext(mlirContext), TheModule(TheModule), context(context),
      builder(&mlirContext), cgTypes(mlirContext, builder, context),
      mangler(context) {}

static void emitFunctionAttributes(clang::FunctionDecl *funcDecl,
                                   mlir::FuncOp mlirFunc,
                                   mlir::OpBuilder &builder) {
  auto funcType = mlirFunc.getType().cast<mlir::FunctionType>();
  const FunctionProtoType *FPT =
      dyn_cast<FunctionProtoType>(funcDecl->getFunctionType());
  // Emit Function attributes.
  if (FPT && funcType.getNumInputs() > 0 && FPT->isVariadic()) {
    mlirFunc.setAttr("std.varargs", builder.getBoolAttr(true));
  }
  if (isa<CXXConstructorDecl>(funcDecl)) {
    mlirFunc.setAttr("is_constructor", builder.getBoolAttr(true));
  } else if (isa<CXXDestructorDecl>(funcDecl)) {
    mlirFunc.setAttr("is_destructor", builder.getBoolAttr(true));
  } else {
    if (!funcDecl->isOverloadedOperator())
      mlirFunc.setAttr("original_name",
                       builder.getStringAttr(funcDecl->getName()));
  }
}

mlir::FuncOp MLIRCodeGen::getFuncOpIfExists(clang::FunctionDecl *funcDecl) {
  std::string mangledName = mangler.mangleFunctionName(funcDecl);
  if (auto methodDecl = dyn_cast<CXXMethodDecl>(funcDecl)) {
    // Look in C++ class / struct / union scope.
    auto classDecl = methodDecl->getParent();
    auto classOp = TheModule->lookupSymbol<CIL::ClassOp>(
        mangler.mangleClassName(classDecl));
    if (!classOp) {
      return {};
    }
    // Remove from deferred.
    return classOp.lookupSymbol<mlir::FuncOp>(mangledName);
  }
  // Look in module scope.
  return TheModule->lookupSymbol<mlir::FuncOp>(mangledName);
}

mlir::FuncOp
MLIRCodeGen::emitFunctionDeclOp(clang::FunctionDecl *funcDecl,
                                mlir::OpBuilder::InsertPoint point) {
  std::string mangledName = mangler.mangleFunctionName(funcDecl);
  mlir::FuncOp funcOp;

  if (auto methodDecl = dyn_cast<CXXMethodDecl>(funcDecl)) {
    // Look in C++ class / struct / union scope.
    auto classDecl = methodDecl->getParent();
    emitClassDecl(classDecl);
    auto classOp = TheModule->lookupSymbol<CIL::ClassOp>(
        mangler.mangleClassName(classDecl));
    assert(classOp);
    // Remove from deferred.
    funcOp = classOp.lookupSymbol<mlir::FuncOp>(mangledName);
    auto &classBody = classOp.body().front();
    if (!point.getBlock()) {
      point = mlir::OpBuilder::InsertPoint(&classBody,
                                           Block::iterator(classBody.back()));
    }
  } else {
    // Look in module scope.
    funcOp = TheModule->lookupSymbol<mlir::FuncOp>(mangledName);
  }
  if (funcOp) {
    return funcOp;
  }
  auto type = funcDecl->getFunctionType();
  auto funcType = cgTypes.convertFunctionType(type);
  assert(funcType);
  auto loc = getLoc(funcDecl->getSourceRange());
  // Create the function.
  auto mlirFunc = mlir::FuncOp::create(loc, mangledName, funcType);
  emitFunctionAttributes(funcDecl, mlirFunc, builder);
  if (point.getBlock()) {
    builder.restoreInsertionPoint(point);
    builder.insert(mlirFunc);
  } else {
    TheModule->push_back(mlirFunc);
  }

  return mlirFunc;
}

mlir::FuncOp MLIRCodeGen::emitFunction(clang::FunctionDecl *funcDecl,
                                       mlir::OpBuilder::InsertPoint point) {

  auto mlirFunc = emitFunctionDeclOp(funcDecl, point);
  if (!funcDecl->hasBody() || !mlirFunc.isExternal()) {
    return mlirFunc;
  }

  auto body = funcDecl->getBody();
  auto block = mlirFunc.addEntryBlock();
  builder.setInsertionPointToStart(block);
  for (unsigned I = 0; I < funcDecl->getNumParams(); ++I) {
    auto param = funcDecl->getParamDecl(I);
    emitDecl(param);
    auto alloca = allocaMap[param];
    builder.create<CIL::CILStoreOp>(getLoc(param->getSourceRange()),
                                    block->getArgument(I), alloca);
  }

  // Initialize CXX constructor initializer.
  if (auto constructor = dyn_cast<CXXConstructorDecl>(funcDecl)) {
    emitConstructorInitalizers(constructor);
  }

  // Emit Function body.
  emitStmt(body);

  // Handle Function return.
  auto endBlock = builder.getBlock();
  if ((endBlock->getOperations().empty() ||
       endBlock->getOperations().back().isKnownNonTerminator())) {
    if (funcDecl->getReturnType()->isVoidType()) {
      builder.create<mlir::ReturnOp>(getLoc(funcDecl->getEndLoc()));
    } else {
      builder.create<CIL::UnreachableOp>(getLoc(funcDecl->getBeginLoc()));
    }
  }

  LabelMap.clear();
  return mlirFunc;
}

mlir::Location MLIRCodeGen::getLoc(clang::SourceRange range, bool isEnd) {
  auto sourceLoc = range.getBegin();
  if (isEnd) {
    sourceLoc = range.getEnd();
  }
  PresumedLoc PLoc = context.getSourceManager().getPresumedLoc(sourceLoc);
  if (PLoc.isInvalid()) {
    return builder.getUnknownLoc();
  }
  return builder.getFileLineColLoc(
      mlir::Identifier::get(PLoc.getFilename(), &mlirContext), PLoc.getLine(),
      PLoc.getColumn());
}

bool MLIRCodeGen::emitGlobal(clang::Decl *decl) {
  if (auto funcDecl = dyn_cast<FunctionDecl>(decl)) {
    // FIXME : This added to avoid generation of illegal instruction in main.
    //        Fix this correctly.
    if (!funcDecl->doesThisDeclarationHaveABody() ||
        (!funcDecl->isMain() &&
         decl->getASTContext().getLangOpts().CPlusPlus)) {
      DeferredDecls.insert(funcDecl);
      return true;
    }
  } else if (auto varDecl = dyn_cast<VarDecl>(decl)) {
    if (varDecl->hasExternalStorage() ||
        varDecl->getInitStyle() != VarDecl::CallInit) {
      DeferredDecls.insert(varDecl);
      return true;
    }
  } else if (auto classDecl = dyn_cast<CXXRecordDecl>(decl)) {
    DeferredDecls.insert(classDecl);
    return true;
  }
  return emitDecl(decl);
}

void MLIRCodeGen::HandleTranslationUnit(ASTContext &Ctx) {
  // Add body to the existing function declaration if not already done.
  for (auto decl : DeferredDecls) {
    auto funcDecl = dyn_cast<clang::FunctionDecl>(decl);
    if (!funcDecl || !funcDecl->hasBody()) {
      continue;
    }

    auto existingFuncOp = getFuncOpIfExists(funcDecl);
    if (!existingFuncOp || !existingFuncOp.isExternal())
      continue;

    if (funcDecl->isInStdNamespace() && funcDecl->isExternallyVisible()) {
      continue;
    }

    emitFunction(funcDecl);
  }
}

// Override the method that gets called for each parsed top-level
// declaration.
bool MLIRCodeGen::HandleTopLevelDecl(DeclGroupRef DR) {
  for (auto topLevelDecl : DR) {
    if (!emitGlobal(topLevelDecl)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<ASTConsumer>
clang::mlir_codegen::makeUniqueMLIRCodeGen(mlir::MLIRContext &context,
                                           mlir::OwningModuleRef &theModule,
                                           clang::ASTContext &astContext) {
  return std::make_unique<clang::mlir_codegen::MLIRCodeGen>(context, theModule,
                                                            astContext);
}
