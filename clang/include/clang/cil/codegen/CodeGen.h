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


#ifndef MLIR_CIL_CODEGEN_H
#define MLIR_CIL_CODEGEN_H

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "clang/cil/codegen/CGTypeConverter.h"
#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"
#include "clang/cil/mangler/CILMangle.h"

namespace clang {
namespace mlir_codegen {

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class MLIRCodeGen : public clang::ASTConsumer {
public:
  MLIRCodeGen(mlir::MLIRContext &mlirContext, mlir::OwningModuleRef &TheModule,
              clang::ASTContext &context);

  // Override the method that gets called for each parsed top-level
  // declaration.

  bool HandleTopLevelDecl(clang::DeclGroupRef DR) override;

  void HandleTranslationUnit(clang::ASTContext &ctx) override;

  bool emitDecl(Decl *decl);

  bool emitGlobal(Decl *decl);

  CIL::AllocaOp emitAllocaOp(mlir::Location loc, StringRef name,
                             mlir::Type type);

  CIL::GlobalOp emitGlobalClassVar(mlir::Location loc, clang::VarDecl *decl,
                                   mlir::Type type,
                                   CXXConstructExpr *constructExpr);

  void setGlobalInitializerRegion(CIL::GlobalOp op, clang::Expr *initExpr);

  bool emitDeclStmt(DeclStmt *stmt);

  mlir::Value emitBinaryOperator(BinaryOperator *binOp);

  mlir::Value emitUnaryOperator(UnaryOperator *binOp);

  mlir::Value emitConditionalOperator(ConditionalOperator *condOp);

  mlir::Value emitRelationalOperator(BinaryOperator *binOp, mlir::Location loc,
                                     mlir::Value, mlir::Value,
                                     BinaryOperatorKind kind);

  bool emitStmt(Stmt *stmt);

  void emitDeclContext(const DeclContext *DC);

  mlir::FuncOp emitCallee(Decl *expr);

  CIL::MemberCallOp emitCXXMemberCall(CXXMemberCallExpr *expr);

  mlir::Value emitCXXNewExpr(clang::CXXNewExpr *expr);

  CIL::MemberCallOp emitMemberCallOp(mlir::Value baseVal,
                                     CXXRecordDecl *classDecl,
                                     CXXMethodDecl *methodDecl,
                                     SmallVector<clang::Expr *, 2> argsList);

  CIL::MemberCallOp emitCXXConstructExpr(CXXConstructExpr *expr,
                                         mlir::Value baseVal);

  bool emitConstructorInitalizers(CXXConstructorDecl *decl);

  mlir::Value emitCallExpr(CallExpr *expr);

  mlir::Value emitIndirectCall(CallExpr *expr);

  mlir::Value emitMemberExpr(MemberExpr *expr);

  mlir::Value emitCXXMemberExpr(MemberExpr *expr);

  mlir::Value emitExpression(Expr *stmt);

  mlir::FuncOp emitFunctionDeclOp(clang::FunctionDecl *funcDecl,
                                  mlir::OpBuilder::InsertPoint point = {});

  mlir::FuncOp getFuncOpIfExists(clang::FunctionDecl *funcDecl);

  mlir::FuncOp emitFunction(FunctionDecl *decl,
                            mlir::OpBuilder::InsertPoint point = {});

  mlir::FuncOp emitCXXMethodDecl(CXXMethodDecl *decl,
                                 mlir::OpBuilder::InsertPoint point = {});

  mlir::Location getLoc(clang::SourceRange range, bool isEnd = false);

  // Expression related classes:

  mlir::Value emitImplicitCastExpr(ImplicitCastExpr *expr);

  mlir::Value emitIntegerLiteral(IntegerLiteral *expr);

  mlir::Value emitFloatingLiteral(FloatingLiteral *expr);

  mlir::Value emitStringLiteral(StringLiteral *expr);

  mlir::Value emitCast(mlir::Value, mlir::Type type);

  mlir::Value emitCastExpr(Expr *expr, clang::QualType toType);

  mlir::Value emitFloatCastExpr(Expr *expr, clang::QualType toType);

  mlir::Value emitArraySubscriptExpr(ArraySubscriptExpr *expr);

  mlir::Value emitDeclRefExpr(clang::DeclRefExpr *expr);

  mlir::Value emitGlobalVar(mlir::Location loc, clang::VarDecl *decl,
                            mlir::Type type);

  mlir::Attribute emitValueAttrForGlobal(mlir::Location loc,
                                         clang::Expr *initExp, mlir::Type type);

  mlir::Value emitAlloca(clang::NamedDecl *decl, StringRef name,
                         mlir::Type type);

  bool emitVarDecl(clang::VarDecl *varDecl);

  mlir::Value emitCompoundAssignment(CompoundAssignOperator *op);

  bool emitIfStmt(IfStmt *stmt);

  bool emitWhileStmt(WhileStmt *stmt);

  bool emitDoWhileStmt(DoStmt *stmt);

  bool emitForStmt(ForStmt *stmt);

  bool emitForStmtBB(ForStmt *stmt);

  bool emitSwitchStmt(SwitchStmt *stmt);

  bool emitGotoStmt(GotoStmt *stmt);

  bool emitLabelStmt(LabelStmt *stmt);

  mlir::Block *getGotoDestBB(LabelDecl *);

  mlir::Value getBoolTypeFor(mlir::Value v);

  // method to return increment of IV in a loop
  mlir::Value getIncr(mlir::Location loc, Stmt *IncrStmt, mlir::Type Ty);

  void computeLBoundAndIV(mlir::Location loc, Stmt *init, mlir::Value &lbound,
                          mlir::Value &IV);

  // Returns block at the end of the current builder region by default
  mlir::Block *getNewBlock(mlir::Block *insertBefore = nullptr);

  bool emitClassDecl(CXXRecordDecl *classDecl);

  mlir::Value emitFieldAccessOp(clang::FieldDecl *field,
                                clang::SourceLocation clangLoc,
                                clang::Expr *baseExpr = nullptr);

  mlir::Value emitPointerRefCast(mlir::Value from, mlir::Type to);

  CIL::AllocaOp emitClassTypeAlloca(mlir::Location loc, StringRef name,
                                    mlir::Type type, clang::Expr *initExpr);

  mlir::Value emitSizeOf(mlir::Location loc, const clang::Type *type);

  mlir::Value emitPredefinedExpr(PredefinedExpr *expr);

  // TODO Below functions should be templated functions
  void emitCondExprIf(Expr *cond, Block *trueBlock, Block *falseBlock);

  void emitCondExprFor(Expr *cond, Block *trueBlock, Block *falseBlock);

  void emitCondExprWhile(Expr *cond, Block *trueBlock, Block *falseBlock);

private:
  mlir::MLIRContext &mlirContext;
  mlir::OwningModuleRef &TheModule;
  clang::ASTContext &context;
  CIL::CILBuilder builder;
  CodeGenTypes cgTypes;
  CIL::CILMangle mangler;
  mlir::FuncOp currFunc;

  llvm::DenseMap<const NamedDecl *, mlir::Value> allocaMap;

  /// This contains all the decls which have definitions but/ which are deferred
  /// for emission and therefore should only be output if they are actually
  /// used. If a decl is in this, then it is known to have not been referenced
  /// yet.
  std::set<Decl *> DeferredDecls;

  SmallVector<std::pair<mlir::Block *, mlir::Block *>, 8> BreakContinueStack;

  // Keeps track of labels and corresponding blocks;
  llvm::DenseMap<LabelDecl *, Block *> LabelMap;
};

std::unique_ptr<ASTConsumer>
makeUniqueMLIRCodeGen(mlir::MLIRContext &context,
                      mlir::OwningModuleRef &theModule,
                      clang::ASTContext &astContext);

} // namespace mlir_codegen
} // namespace clang

#endif
