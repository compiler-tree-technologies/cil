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

#include "clang/cil/codegen/CGTypeConverter.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

using namespace clang;
using namespace clang::mlir_codegen;

CodeGenTypes::CodeGenTypes(mlir::MLIRContext &mlirContext,
                           CIL::CILBuilder &builder, clang::ASTContext &context)
    : mlirContext(mlirContext), builder(builder), context(context),
      mangler(context) {}

mlir::FunctionType CodeGenTypes::convertFunctionType(const FunctionType *FT) {

  SmallVector<mlir::Type, 2> args;
  if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(FT)) {
    for (unsigned i = 0, e = FPT->getNumParams(); i != e; i++) {
      args.push_back(convertClangType(FPT->getParamType(i)));
    }
    // if (FPT->isVariadic())
    //  args.push_back(CIL::VarArgTy::get(&mlirContext));
  }
  if (FT->getReturnType()->isVoidType()) {
    return mlir::FunctionType::get(args, {}, &mlirContext);
  }
  auto returnType = convertClangType(FT->getReturnType());
  return mlir::FunctionType::get(args, {returnType}, &mlirContext);
}

mlir::FunctionType
CodeGenTypes::convertFunctionProtoType(const FunctionProtoType *FPT) {

  SmallVector<mlir::Type, 2> args;
  for (unsigned i = 0, e = FPT->getNumParams(); i != e; i++) {
    args.push_back(convertClangType(FPT->getParamType(i)));
  }
  // if (FPT->isVariadic())
  //  args.push_back(CIL::VarArgTy::get(&mlirContext));
  if (FPT->getReturnType()->isVoidType()) {
    return mlir::FunctionType::get(args, {}, &mlirContext);
  }
  auto returnType = convertClangType(FPT->getReturnType());
  return mlir::FunctionType::get(args, {returnType}, &mlirContext);
}

mlir::Type convertClangType(const clang::Type *type) { return {}; }

mlir::Type CodeGenTypes::convertBuiltinType(const clang::BuiltinType *BT) {

  if (BT->isCharType()) {
    auto charTy = BT->getCanonicalTypeInternal();
    CIL::CILQualifiers qual;
    if (charTy.isConstant(context))
      qual.setConst();
    return CIL::CharTy::get(qual, &mlirContext);
  }

  // FIXME: Make it 16/32 bit char.
  if (BT->isWideCharType()) {
    auto charTy = BT->getCanonicalTypeInternal();
    CIL::CILQualifiers qual;
    if (charTy.isConstant(context))
      qual.setConst();
    return CIL::IntegerTy::get(CIL::CILIntegerKind::IntKind, qual,
                               &mlirContext);
  }

  if (BT->isIntegerType()) {
    CIL::CILIntegerKind kind;
    CIL::CILQualifiers qual;
    switch (BT->getKind()) {
    case BuiltinType::Short: {
      kind = CIL::CILIntegerKind::ShortKind;
    } break;
    case BuiltinType::Int: {
      kind = CIL::CILIntegerKind::IntKind;
    } break;
    case BuiltinType::Long: {
      kind = CIL::CILIntegerKind::LongKind;
    } break;
    case BuiltinType::LongLong: {
      kind = CIL::CILIntegerKind::LongLongKind;
    } break;
    case BuiltinType::ULongLong: {
      kind = CIL::CILIntegerKind::LongLongKind;
      qual.setUnsigned();
    } break;
    case BuiltinType::ULong: {
      kind = CIL::CILIntegerKind::LongKind;
      qual.setUnsigned();
    } break;
    case BuiltinType::UInt: {
      kind = CIL::CILIntegerKind::IntKind;
      qual.setUnsigned();
    } break;
    case BuiltinType::UShort: {
      kind = CIL::CILIntegerKind::ShortKind;
      qual.setUnsigned();
    } break;
    case BuiltinType::Bool: {
      kind = CIL::CILIntegerKind::BoolKind;
      // FIXME: getCILBoolTy() and this function returns
      // two mismatching bool types if we uncomment the below line.
      // qual.setUnsigned();
    } break;
    default:
      if (BT->isBooleanType()) {
        kind = CIL::CILIntegerKind::BoolKind;
        qual.setUnsigned();
        break;
      }
      BT->dump();
      llvm_unreachable("unknown integral type");
    };
    return CIL::IntegerTy::get(kind, qual, &mlirContext);
  }
  if (BT->isUnsignedIntegerType()) {
    CIL::CILIntegerKind kind;
    switch (BT->getKind()) {
    case BuiltinType::Short: {
      kind = CIL::CILIntegerKind::ShortKind;
    } break;
    case BuiltinType::Int: {
      kind = CIL::CILIntegerKind::IntKind;
    } break;
    case BuiltinType::Long: {
      kind = CIL::CILIntegerKind::IntKind;
    } break;
    case BuiltinType::LongLong: {
      kind = CIL::CILIntegerKind::LongLongKind;
    } break;
    default:
      BT->dump();
      llvm_unreachable("unknown unsigned integral type");
    };
    CIL::CILQualifiers qual;
    qual.setUnsigned();
    return CIL::IntegerTy::get(kind, qual, &mlirContext);
  }

  if (BT->isFloatingType()) {
    QualType qType = BT->getCanonicalTypeInternal();
    CIL::CILQualifiers qual;
    if (qType.hasQualifiers()) {
      auto clangQual = qType.getQualifiers();
      if (clangQual.hasConst())
        qual.setConst();
      if (clangQual.hasVolatile())
        qual.setVolatile();
    }
    CIL::CILFloatingKind kind;
    switch (BT->getKind()) {
    case BuiltinType::Float:
      kind = CIL::CILFloatingKind::Float;
      break;
    case BuiltinType::Double:
      kind = CIL::CILFloatingKind::Double;
      break;
    case BuiltinType::LongDouble:
      kind = CIL::CILFloatingKind::LongDouble;
      break;
    default:
      BT->dump();
      llvm_unreachable("Unhandled floating type");
    }
    return CIL::FloatingTy::get(kind, qual, &mlirContext);
  }
  if (BT->isNullPtrType()) {
    return CIL::NullPtrTTy::get(&mlirContext);
  }
  if (BT->isVoidType()) {
    return CIL::VoidType::get(&mlirContext);
  }

  BT->dump();
  llvm_unreachable("unhandled builtin type");
  return {};
}

static std::string getRecordTypeName(const RecordDecl *RD) {
  SmallString<256> TypeName;
  llvm::raw_svector_ostream OS(TypeName);

  if (RD->getIdentifier()) {
    if (RD->getDeclContext())
      RD->printQualifiedName(OS);
    else
      RD->printName(OS);
  } else if (const TypedefNameDecl *TDD = RD->getTypedefNameForAnonDecl()) {
    if (TDD->getDeclContext())
      TDD->printQualifiedName(OS);
    else
      TDD->printName(OS);
  } else {
    OS << RD->getKindName() << '.';
    OS << "anon";
  }

  return OS.str().str();
}

mlir::Type CodeGenTypes::convertClangType(const clang::Type *type) {
  type = type->getUnqualifiedDesugaredType();

  // Builtin types.
  if (type->isBuiltinType()) {
    return convertBuiltinType(cast<clang::BuiltinType>(type));
  }
  // Function types.
  if (type->isFunctionProtoType()) {
    return convertFunctionProtoType(cast<clang::FunctionProtoType>(type));
  }

  // Pointer types.
  if (type->isPointerType()) {
    // Pointer to array decayed type.
    if (auto var = dyn_cast<clang::DecayedType>(type)) {
      type = var->getDecayedType().getTypePtr();
    }
    auto var = cast<clang::PointerType>(type);
    mlir::Type eleTy;
    if (var->getPointeeType()->isVoidType())
      eleTy = CIL::IntegerTy::get(CIL::CILIntegerKind::Char8Kind, &mlirContext);
    else
      eleTy = convertClangType(var->getPointeeType());
    return builder.getCILPointerType(eleTy);
  }

  // Reference types
  if (type->isReferenceType()) {
    auto refType = cast<clang::ReferenceType>(type);
    auto eleTy = convertClangType(refType->getPointeeType());
    return builder.getCILPointerType(eleTy, true);
  }
  // Array types.
  if (type->isArrayType()) {
    auto var = dyn_cast<clang::ConstantArrayType>(type);
    assert(var);
    auto shape = var->getSize().getSExtValue();
    auto eleTy = convertClangType(var->getElementType());
    return CIL::ArrayType::get({shape}, eleTy);
  }

  // Enum like types.
  if (type->isEnumeralType()) {
    return builder.getCILIntType();
  }

  if (auto recordType = dyn_cast<RecordType>(type)) {
    if (auto decl = dyn_cast<CXXRecordDecl>(recordType->getDecl())) {
      auto name = mangler.mangleClassName(decl);
      return builder.getCILClassType(name);
    }
  }

  // Structure types.
  if (type->isStructureType()) {
    llvm::SmallVector<mlir::Type, 2> types;

    auto recordTy = type->getAsStructureType();
    assert(recordTy);
    auto name = getRecordTypeName(recordTy->getDecl());
    if (RecordDeclTypes.find(type) == RecordDeclTypes.end()) {
      // Create a forward declaration.
      RecordDeclTypes[type] = CIL::StructType::get(&mlirContext, types, name);
    } else {
      return RecordDeclTypes[type];
    }

    auto currStructTy = RecordDeclTypes[type];
    auto decl = recordTy->getDecl();
    for (auto *field : decl->fields()) {
      currStructTy.addMemberType(convertClangType(field->getType()));
    }

    return currStructTy;
  }

  if (auto typdefType = dyn_cast<TypedefType>(type)) {
    return convertClangType(typdefType->desugar());
  }
  type->dump();
  llvm_unreachable("unhandled clang type");
  return {};
}

mlir::Type CodeGenTypes::convertClangType(const clang::QualType qType) {
  auto type = qType.getTypePtr();
  return convertClangType(type);
}
