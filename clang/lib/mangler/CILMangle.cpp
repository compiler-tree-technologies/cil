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
//===- CILToLLVMLowering.cpp - Loop.for to affine.for conversion
//-----------===//
//
//===----------------------------------------------------------------------===//

#include "clang/cil/mangler/CILMangle.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"

using namespace CIL;
using namespace llvm;
using namespace clang;

CILMangle::CILMangle(clang::ASTContext &context) {
  mangler =
      clang::ItaniumMangleContext::create(context, context.getDiagnostics());
}

// TODO: Incomplete list.
static bool shouldMangleName(clang::FunctionDecl *decl) {
  if (isa<CXXConstructorDecl>(decl) || isa<CXXDestructorDecl>(decl)) {
    return true;
  }
  if (decl->isTemplateInstantiation()) {
    return true;
  }
  if (decl->isOverloadedOperator()) {
    return true;
  }
  if (decl->isMain())
    return false;
  // Do not mangle declarations for now. Example: printf
  if (!decl->hasBody()) {
    return false;
  }
  if (!decl->getASTContext().getLangOpts().CPlusPlus) {
    return false;
  }
  return true;
}

std::string CILMangle::mangleClassName(clang::CXXRecordDecl *decl) {

  static llvm::DenseMap<CXXRecordDecl *, std::string> declNameMap;
  if (declNameMap.find(decl) != declNameMap.end()) {
    return declNameMap.lookup(decl);
  }

  // Check if the class is under any namespace
  std::string namespacePrefix = "";
  auto namespaceDecl = decl->getEnclosingNamespaceContext();
  while (namespaceDecl) {
    auto nmDecl = dyn_cast<NamespaceDecl>(namespaceDecl);
    if (!nmDecl) {
      break;
    }
    namespacePrefix = nmDecl->getName().str() + "::" + namespacePrefix;
    namespaceDecl = nmDecl->getParent();
  }

  auto mangledName = namespacePrefix + decl->getName().str();
  static unsigned long long int counter = 0;
  static llvm::StringSet<> recordedNames;

  if (recordedNames.find(mangledName) != recordedNames.end()) {
    mangledName = mangledName + "." + std::to_string(counter++);
  }
  recordedNames.insert(mangledName);
  declNameMap[decl] = mangledName;
  return mangledName;
}

std::string CILMangle::mangleFunctionName(clang::FunctionDecl *decl) {
  if (!shouldMangleName(decl)) {
    return decl->getName().str();
  }

  std::string mangledName;
  llvm::raw_string_ostream stream(mangledName);
  if (auto constructor = llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
    mangler->mangleCXXCtor(constructor, clang::Ctor_Complete, stream);
  } else if (auto constructor =
                 llvm::dyn_cast<clang::CXXDestructorDecl>(decl)) {
    mangler->mangleCXXDtor(constructor, clang::Dtor_Complete, stream);
  } else {
    mangler->mangleCXXName(decl, stream);
  }
  stream.flush();
  return mangledName;
}

std::string CILMangle::mangleGlobalName(clang::NamedDecl *decl) {
  std::string mangledName;
  llvm::raw_string_ostream stream(mangledName);
  mangler->mangleCXXName(decl, stream);
  stream.flush();
  return mangledName;
}