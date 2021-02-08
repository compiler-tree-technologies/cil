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
//===- CILOps.h CIL specific Ops ----------------------------------===//

#ifndef MLIR_DIALECT_CIL_TYPES_H
#define MLIR_DIALECT_CIL_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;

namespace CIL {

namespace detail {
struct ArrayTypeStorage;
struct PointerTypeStorage;
struct StructTypeStorage;
struct IntegerTypeStorage;
struct UnsignedIntegerTypeStorage;
struct CharTypeStorage;
struct FloatTypeStorage;
struct ClassTypeStorage;
} // namespace detail

enum CILTypeKind {
  CIL_Char = mlir::Type::FIRST_FIR_TYPE,
  CIL_UnsignedChar,
  CIL_Integer,
  CIL_UnsignedInteger,
  CIL_FloatingTypeKind,
  CIL_Struct,
  CIL_Pointer,
  CIL_Array,
  CIL_Ref,
  CIL_VarArg,
  CIL_Void,
  CIL_Class,
  CIL_NullPtrT,
};

enum CILIntegerKind {
  BoolKind,
  Char8Kind,
  ShortKind,
  IntKind,
  LongKind,
  LongLongKind,
};

enum CILFloatingKind {
  Float,
  Double,
  LongDouble,
};

class CILQualifiers {
  enum CILQualifierKind {
    Default = 0x0,
    Unsigned = 0x1,
    Const = 0x2,
    Volatile = 0x4,
    Extern = 0x8,
    Static = 0x16,
  };
  uint32_t mask;

public:
  CILQualifiers() : mask(Default) {}

  void setUnsigned() { mask |= Unsigned; }
  bool isUnsigned() { return mask & Unsigned; }
  bool isConst() { return mask & Const; }
  bool isVolatile() { return mask & Volatile; }
  bool isExtern() { return mask & Extern; }
  bool isStatic() { return mask & Static; }
  void setConst() { mask |= Const; }
  void setVolatile() { mask |= Volatile; }
  uint32_t getMask() const { return mask; }
  bool operator==(const CILQualifiers &other) const {
    return other.getMask() == mask;
  }

  void print(raw_ostream &os);
};

llvm::hash_code hash_value(const CILQualifiers &);

enum CILAttrKind {
  CIL_Attr = mlir::Attribute::FIRST_FIR_ATTR,
};

// This is an opaque class type to represent the class. Differentiating
// different class types is done using "name". Name should be unique for every
// different class type.
class ClassType
    : public Type::TypeBase<ClassType, Type, detail::ClassTypeStorage> {
public:
  using Base::Base;

  static ClassType get(std::string name, MLIRContext *context);

  StringRef getName();

  static bool kindof(unsigned kind) { return kind == CIL::CIL_Class; }
};

/// Type to represent void type
class VoidType : public Type::TypeBase<VoidType, Type> {
public:
  using Base::Base;

  static VoidType get(MLIRContext *context);

  static bool kindof(unsigned kind) { return kind == CIL::CIL_Void; }
};

/// Type to represent nullptr_t type
class NullPtrTTy : public Type::TypeBase<NullPtrTTy, Type> {
public:
  using Base::Base;

  static NullPtrTTy get(MLIRContext *context);

  static bool kindof(unsigned kind) { return kind == CIL::CIL_NullPtrT; }
};

/// Index is a special integer-like type with unknown platform-dependent bit
/// width.
class VarArgTy : public Type::TypeBase<VarArgTy, Type> {
public:
  using Base::Base;

  /// Get an instance of the IndexType.
  static VarArgTy get(MLIRContext *context);

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == CIL::CIL_VarArg; }
};

class UnsignedChar : public mlir::Type::TypeBase<UnsignedChar, Type> {
public:
  using Base::Base;

  /// Get an instance of the IndexType.
  static UnsignedChar get(MLIRContext *context);

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == CIL::CIL_UnsignedChar; }
};

class CharTy : public Type::TypeBase<CharTy, Type, detail::CharTypeStorage> {
public:
  using Base::Base;

  static CharTy get(CILQualifiers qualifers, MLIRContext *context);
  static CharTy get(MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == CIL_Char; }

  CILQualifiers getQualifiers();
};

class FloatingTy
    : public Type::TypeBase<FloatingTy, Type, detail::FloatTypeStorage> {
public:
  using Base::Base;

  static FloatingTy get(CILFloatingKind kind, CILQualifiers qual,
                        MLIRContext *context);
  static FloatingTy get(CILFloatingKind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == CIL_FloatingTypeKind; }

  bool isFloatTy();
  bool isDoubleTy();

  CILFloatingKind getFloatingKind();
  CILQualifiers getQualifiers();
};

/// Integer types can have arbitrary bitwidth up to a large fixed limit.
class IntegerTy
    : public Type::TypeBase<IntegerTy, Type, detail::IntegerTypeStorage> {
public:
  using Base::Base;

  static IntegerTy get(CILIntegerKind kind, CILQualifiers qual,
                       MLIRContext *context);
  static IntegerTy get(CILIntegerKind kind, MLIRContext *context);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) { return kind == CIL_Integer; }

  bool isIntTy();
  bool isLongTy();
  bool isBoolTy();
  bool isCharTy();
  CILIntegerKind getIntegerKind();
  CILQualifiers getQualifiers();
  bool isBoolType();
};

class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type,
                                              CIL::detail::ArrayTypeStorage> {
public:
  static constexpr int64_t dynamicSizeValue = -1;
  static int64_t getDynamicSizeValue() { return dynamicSizeValue; }
  using Base::Base;
  using Shape = llvm::SmallVector<int64_t, 4>;

  mlir::Type getEleTy() const;

  Shape getShape() const;

  bool hasStaticShape() const {
    for (auto &dim : getShape()) {
      if (dim == ArrayType::getDynamicSizeValue()) {
        return false;
      }
    }
    return true;
  }
  unsigned getRank() const { return getShape().size(); }

  static ArrayType get(const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) { return kind == CILTypeKind::CIL_Array; }
};

bool operator==(const ArrayType::Shape &, const ArrayType::Shape &);

llvm::hash_code hash_value(const ArrayType::Shape &);

class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               CIL::detail::StructTypeStorage> {

public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == CILTypeKind::CIL_Struct; }

  static StructType get(MLIRContext *ctx,
                        llvm::SmallVector<mlir::Type, 2> elementTypes,
                        StringRef name);

  // TODO: Also add name of each field?
  static StructType get(MLIRContext *ctx,
                        llvm::SmallVector<mlir::Type, 2> elementTypes);

  llvm::ArrayRef<mlir::Type> getElementTypes();

  mlir::Type getStructElementType(unsigned i);

  llvm::StringRef getName();

  void addMemberType(mlir::Type type);

  void finalize();

  bool isCompleteType();

  void setBody(llvm::SmallVector<mlir::Type, 2> elementTypes);

  size_t getNumElementTypes() { return getElementTypes().size(); }
};

// Represents fortran pointer type.
class PointerType
    : public mlir::Type::TypeBase<PointerType, mlir::Type,
                                  CIL::detail::PointerTypeStorage> {
public:
  using Base::Base;
  mlir::Type getEleTy() const;
  static bool kindof(unsigned kind) { return kind == CILTypeKind::CIL_Pointer; }
  static PointerType get(mlir::Type elementType, bool isRef = false);
  bool isReference() const;

}; // namespace CIL

} // namespace CIL
#endif
