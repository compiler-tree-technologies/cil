#include "clang/cil/codegen/CodeGen.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

using namespace clang;
using namespace clang::mlir_codegen;

CIL::AllocaOp MLIRCodeGen::emitAllocaOp(mlir::Location loc, StringRef name,
                                        mlir::Type type) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToStart(&builder.getBlock()->getParent()->front());
  return builder.create<CIL::AllocaOp>(loc, name, type);
}

bool MLIRCodeGen::emitVarDecl(clang::VarDecl *varDecl) {
  auto type = cgTypes.convertClangType(varDecl->getType());
  auto namedDecl = varDecl->getUnderlyingDecl();
  auto loc = getLoc(varDecl->getSourceRange(), true);
  auto initExpr = varDecl->getInit();

  // Emit Global variable if it is a constant array kind.
  if (isa_and_nonnull<InitListExpr>(initExpr) || varDecl->isFileVarDecl()) {
    auto value = emitGlobalVar(loc, varDecl, type);
    allocaMap[namedDecl] = value;
    return true;
  }

  // Emit class type alloca.
  if (builder.getUnderlyingType(type).isa<CIL::ClassType>() &&
      !type.isa<CIL::PointerType>()) {
    auto classDecl = varDecl->getType()->getAsCXXRecordDecl();
    // TODO: FAILS sometimes. need to check.
    if (classDecl)
      emitClassDecl(classDecl);
    auto alloca = emitClassTypeAlloca(loc, varDecl->getName(), type, initExpr);
    allocaMap[namedDecl] = alloca.getResult();
    return true;
  }

  // Create local stack alloca operation.
  auto alloca = emitAllocaOp(loc, varDecl->getName(), type);
  allocaMap[namedDecl] = alloca.getResult();
  if (!initExpr) {
    return true;
  }

  auto initVal = emitExpression(initExpr);
  builder.create<CIL::CILStoreOp>(loc, initVal, alloca.getResult());
  return true;
}

bool MLIRCodeGen::emitDecl(clang::Decl *decl) {
  // Traverse the declaration using our AST visitor.
  switch (decl->getKind()) {
  case clang::Decl::Function: {
    return emitFunction(static_cast<clang::FunctionDecl *>(decl));
  }
  case clang::Decl::ParmVar: {
    auto varDecl = static_cast<clang::ParmVarDecl *>(decl);
    auto namedDecl = varDecl->getUnderlyingDecl();
    auto type = cgTypes.convertClangType(varDecl->getType());
    assert(!varDecl->isFileVarDecl());
    auto loc = getLoc(decl->getSourceRange(), true);
    auto alloca = emitAllocaOp(loc, varDecl->getName(), type);
    allocaMap[namedDecl] = alloca.getResult();
    if (varDecl->hasInit()) {
      auto initVal = emitExpression(varDecl->getInit());
      builder.create<CIL::CILStoreOp>(loc, initVal, alloca.getResult());
    }
    return true;
  }
  case clang::Decl::Var: {
    return emitVarDecl(static_cast<clang::VarDecl *>(decl));
  }
  case clang::Decl::LinkageSpec: {
    emitDeclContext(static_cast<clang::LinkageSpecDecl *>(decl));
    return true;
  }
  case clang::Decl::Enum:
  case clang::Decl::Record:
  case clang::Decl::Typedef: {
  } break;
  case clang::Decl::CXXRecord: {
    auto classDecl = static_cast<clang::CXXRecordDecl *>(decl);
    return emitClassDecl(classDecl);
  } break;
  case clang::Decl::FunctionTemplate: {
    auto funcTemp = static_cast<clang::FunctionTemplateDecl *>(decl);
    for (auto spec : funcTemp->specializations()) {
      DeferredDecls.insert(spec);
    }
  } break;
  case clang::Decl::ClassTemplate: {
    auto classTemp = static_cast<clang::ClassTemplateDecl *>(decl);
    for (auto spec : classTemp->specializations()) {
      DeferredDecls.insert(spec);
    }
  } break;
  case clang::Decl::CXXMethod: {
    DeferredDecls.insert(decl);
    break;
    auto methodDecl = static_cast<clang::CXXMethodDecl *>(decl);
    return emitCXXMethodDecl(methodDecl);
  } break;
  case clang::Decl::CXXConstructor: {
    auto constructDecl = static_cast<clang::CXXConstructorDecl *>(decl);
    return emitCXXMethodDecl(constructDecl);
  } break;
  case clang::Decl::CXXDestructor: {
    auto constructDecl = static_cast<clang::CXXConstructorDecl *>(decl);
    return emitCXXMethodDecl(constructDecl);
  } break;
  case clang::Decl::Namespace: {
    auto namespaceDecl = static_cast<clang::NamespaceDecl *>(decl);
    for (auto decl : namespaceDecl->decls()) {
      emitGlobal(decl);
    }
  } break;
  case clang::Decl::Using:
  case clang::Decl::UsingShadow:
  case clang::Decl::TypeAliasTemplate:
  case clang::Decl::UsingDirective:
  case clang::Decl::ClassTemplateSpecialization:
  case clang::Decl::TypeAlias:
  case clang::Decl::StaticAssert: {
    break;
  }
  default:
    decl->dump();
    llvm_unreachable("Declaration kind unhandled");
  }

  return true;
}

mlir::FuncOp
MLIRCodeGen::emitCXXMethodDecl(CXXMethodDecl *methodDecl,
                               mlir::OpBuilder::InsertPoint point) {
  auto insertPt = builder.saveInsertionPoint();
  auto funcOp = emitFunction(methodDecl, point);
  builder.restoreInsertionPoint(insertPt);
  return funcOp;
}

void MLIRCodeGen::emitDeclContext(const DeclContext *DC) {
  for (auto *I : DC->decls()) {
    // Unlike other DeclContexts, the contents of an ObjCImplDecl at TU scope
    // are themselves considered "top-level", so EmitTopLevelDecl on an
    // ObjCImplDecl does not recursively visit them. We need to do that in
    // case they're nested inside another construct (LinkageSpecDecl /
    // ExportDecl) that does stop them from being considered "top-level".
    if (auto *OID = dyn_cast<ObjCImplDecl>(I)) {
      for (auto *M : OID->methods())
        emitGlobal(M);
    }
    emitGlobal(I);
  }
}

bool MLIRCodeGen::emitDeclStmt(DeclStmt *stmt) {
  for (auto decl : stmt->decls()) {
    if (!emitDecl(decl)) {
      return false;
    }
  }
  return true;
}
