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
//===- CILOps.cpp - CIL Operations ---------------------------------===//

#include "clang/cil/dialect/CIL/CILTypes.h"
#include "clang/cil/dialect/CIL/CILDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include <map>

using namespace mlir;
using namespace CIL;

// compare if two shapes are equivalent
bool CIL::operator==(const ArrayType::Shape &shape1,
                     const ArrayType::Shape &shape2) {
  if (shape1.size() != shape2.size())
    return false;
  for (std::size_t i = 0, e = shape1.size(); i != e; ++i)
    if (shape1[i] != shape2[i])
      return false;
  return true;
}

// compute the hash of a Shape
llvm::hash_code CIL::hash_value(const CILQualifiers &sh) {
  return llvm::hash_combine(sh.getMask());
}

// compute the hash of a Shape
llvm::hash_code CIL::hash_value(const ArrayType::Shape &sh) {
  if (sh.empty()) {
    return llvm::hash_combine(0);
  }
  llvm::SmallVector<llvm::hash_code, 2> values(sh.size());
  for (auto &dim : sh) {
    values.push_back(dim);
  }

  return llvm::hash_combine_range(values.begin(), values.end());
}

namespace CIL {
namespace detail {

/// Integer Type Storage and Uniquing.
struct IntegerTypeStorage : public TypeStorage {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<CILIntegerKind, CILQualifiers>;
  IntegerTypeStorage(const KeyTy &key) : fields(key) {}

  static unsigned hashKey(const KeyTy &key) {
    int val = std::get<0>(key);
    llvm::hash_code val1 = CIL::hash_value(std::get<1>(key));
    return llvm::hash_combine(val, val1);
  }

  bool operator==(const KeyTy &key) const { return key == fields; }

  static IntegerTypeStorage *construct(TypeStorageAllocator &allocator,
                                       KeyTy &key) {
    return new (allocator.allocate<IntegerTypeStorage>())
        IntegerTypeStorage(key);
  }

  KeyTy fields;
};

/// Float Type Storage and Uniquing.
struct FloatTypeStorage : public TypeStorage {
  /// The hash key used for uniquing.
  using KeyTy = std::tuple<CILFloatingKind, CILQualifiers>;
  FloatTypeStorage(const KeyTy &key) : fields(key) {}

  static unsigned hashKey(const KeyTy &key) {
    int val = std::get<0>(key);
    llvm::hash_code val1 = CIL::hash_value(std::get<1>(key));
    return llvm::hash_combine(val, val1);
  }

  bool operator==(const KeyTy &key) const { return key == fields; }

  static FloatTypeStorage *construct(TypeStorageAllocator &allocator,
                                     KeyTy &key) {
    return new (allocator.allocate<FloatTypeStorage>()) FloatTypeStorage(key);
  }

  KeyTy fields;
};

struct CharTypeStorage : public TypeStorage {
  /// The hash key used for uniquing.
  using KeyTy = CILQualifiers;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{CIL::hash_value(key)};
    return llvm::hash_combine(shapeHash);
  }

  CharTypeStorage(const KeyTy &key) : fields(key) {}

  bool operator==(const KeyTy &key) const { return key == fields; }

  static CharTypeStorage *construct(TypeStorageAllocator &allocator,
                                    KeyTy &key) {
    return new (allocator.allocate<CharTypeStorage>()) CharTypeStorage(key);
  }

  KeyTy fields;
};

struct ClassTypeStorage : public TypeStorage {
  /// The hash key used for uniquing.
  using KeyTy = std::string;

  static unsigned hashKey(const KeyTy &key) {
    auto hash1 = llvm::hash_combine_range(key.begin(), key.end());
    return llvm::hash_combine(hash1);
  }

  ClassTypeStorage(const KeyTy &key) : fields(key) {}

  bool operator==(const KeyTy &key) const { return key == fields; }

  static ClassTypeStorage *construct(TypeStorageAllocator &allocator,
                                     KeyTy &key) {
    return new (allocator.allocate<ClassTypeStorage>()) ClassTypeStorage(key);
  }

  KeyTy fields;
};

struct ArrayTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<ArrayType::Shape, mlir::Type>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{CIL::hash_value(std::get<ArrayType::Shape>(key))};
    return llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getShape(), getElementType()};
  }

  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    auto *storage = allocator.allocate<ArrayTypeStorage>();
    return new (storage) ArrayTypeStorage{std::get<ArrayType::Shape>(key),
                                          std::get<mlir::Type>(key)};
  }

  ArrayType::Shape getShape() const { return shape; }

  mlir::Type getElementType() const { return eleTy; }

protected:
  ArrayType::Shape shape;
  mlir::Type eleTy;

private:
  ArrayTypeStorage() = delete;
  explicit ArrayTypeStorage(const ArrayType::Shape &shape, mlir::Type eleTy)
      : shape{shape}, eleTy{eleTy} {}
};

struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<llvm::SmallVector<mlir::Type, 2>, std::string>;

  StructTypeStorage(KeyTy elementTypes) : fields(elementTypes) {}

  bool operator==(const KeyTy &key) const {
    // FIXME : Is this correct ?
    //         key == fields is modified as there is type mis match when forward
    //         declarations are created.
    return std::get<1>(key) == std::get<1>(fields);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    auto hash1 = llvm::hash_combine_range(std::get<0>(key).begin(),
                                          std::get<0>(key).end());
    auto hash2 = llvm::hash_value(std::get<1>(key));
    return llvm::hash_value(
        std::pair<llvm::hash_code, llvm::hash_code>(hash1, hash2));
  }

  static KeyTy getKey(KeyTy fields) { return KeyTy(fields); }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(key);
  }

  llvm::ArrayRef<mlir::Type> getElementTypes() { return std::get<0>(fields); }

  llvm::StringRef getName() { return std::get<1>(fields); }

  KeyTy fields;

  bool completeType = false;
};

struct PointerTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<mlir::Type, bool>;

  static unsigned hashKey(const KeyTy &key) {

    return llvm::hash_combine(key.first, key.second);
  }

  bool operator==(const KeyTy &key) const {
    return key.first == getElementType() && key.second == isRef();
  }

  static PointerTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    auto *storage = allocator.allocate<PointerTypeStorage>();
    return new (storage) PointerTypeStorage{key};
  }

  mlir::Type getElementType() const { return eleTy; }

  bool isRef() const { return isReference; }

protected:
  mlir::Type eleTy;
  bool isReference;

private:
  PointerTypeStorage() = delete;
  explicit PointerTypeStorage(std::pair<mlir::Type, bool> key)
      : eleTy{key.first}, isReference{key.second} {}
};

} // namespace detail
} // namespace CIL

ArrayType ArrayType::get(const Shape &shape, mlir::Type elementType) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, CIL_Array, shape, elementType);
}

bool IntegerTy::isBoolType() {
  return getIntegerKind() == CILIntegerKind::BoolKind;
}

IntegerTy IntegerTy::get(CILIntegerKind kind, CILQualifiers qual,
                         MLIRContext *context) {
  return Base::get(context, CIL_Integer, std::make_tuple(kind, qual));
}

ClassType ClassType::get(std::string name, MLIRContext *context) {
  return Base::get(context, CIL_Class, name);
}

StringRef ClassType::getName() { return getImpl()->fields; }

CharTy CharTy::get(CILQualifiers qual, MLIRContext *context) {
  return Base::get(context, CIL_Char, qual);
}

VarArgTy VarArgTy::get(MLIRContext *context) {
  return Base::get(context, CIL_VarArg);
}

VoidType VoidType::get(MLIRContext *context) {
  return Base::get(context, CIL_Void);
}

NullPtrTTy NullPtrTTy::get(MLIRContext *context) {
  return Base::get(context, CIL_NullPtrT);
}

CharTy CharTy::get(MLIRContext *context) {
  return Base::get(context, CIL_Char, CILQualifiers());
}

CILQualifiers CharTy::getQualifiers() { return getImpl()->fields; }

IntegerTy IntegerTy::get(CILIntegerKind kind, MLIRContext *context) {
  return Base::get(context, CIL_Integer,
                   std::make_tuple(kind, CILQualifiers()));
}

CILIntegerKind IntegerTy::getIntegerKind() {
  return std::get<0>(getImpl()->fields);
}
CILQualifiers IntegerTy::getQualifiers() {
  return std::get<1>(getImpl()->fields);
}

bool IntegerTy::isIntTy() {
  return std::get<0>(getImpl()->fields) == CIL::IntKind;
}

bool IntegerTy::isLongTy() {
  return std::get<0>(getImpl()->fields) == CIL::LongKind;
}

bool IntegerTy::isBoolTy() {
  return std::get<0>(getImpl()->fields) == CIL::BoolKind;
}

bool IntegerTy::isCharTy() {
  return std::get<0>(getImpl()->fields) == CIL::Char8Kind;
}

FloatingTy FloatingTy::get(CILFloatingKind kind, MLIRContext *context) {
  return Base::get(context, CIL_FloatingTypeKind,
                   std::make_tuple(kind, CILQualifiers()));
}

FloatingTy FloatingTy::get(CILFloatingKind kind, CILQualifiers qualifier,
                           MLIRContext *context) {
  return Base::get(context, CIL_FloatingTypeKind,
                   std::make_tuple(kind, qualifier));
}

CILFloatingKind FloatingTy::getFloatingKind() {
  return std::get<0>(getImpl()->fields);
}

bool FloatingTy::isFloatTy() {
  return std::get<0>(getImpl()->fields) == CIL::Float;
}

bool FloatingTy::isDoubleTy() {
  return std::get<0>(getImpl()->fields) == CIL::Double;
}

CILQualifiers FloatingTy::getQualifiers() {
  return std::get<1>(getImpl()->fields);
}

void StructType::finalize() { getImpl()->completeType = true; }

bool StructType::isCompleteType() { return getImpl()->completeType; }

StructType StructType::get(MLIRContext *ctx,
                           llvm::SmallVector<mlir::Type, 2> elementTypes,
                           StringRef name) {
  return Base::get(ctx, CILTypeKind::CIL_Struct,
                   std::make_tuple(elementTypes, name));
}

StructType StructType::get(MLIRContext *ctx,
                           llvm::SmallVector<mlir::Type, 2> elementTypes) {
  return Base::get(ctx, CILTypeKind::CIL_Struct,
                   std::make_tuple(elementTypes, ""));
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  return getImpl()->getElementTypes();
}

/// Adds a new type to the end of current type list
void StructType::addMemberType(mlir::Type ty) {
  std::get<0>(getImpl()->fields).push_back(ty);
}

mlir::Type StructType::getStructElementType(unsigned i) {
  return getImpl()->getElementTypes()[i];
}

void StructType::setBody(llvm::SmallVector<mlir::Type, 2> eleTypes) {
  std::get<0>(getImpl()->fields).clear();
  std::get<0>(getImpl()->fields)
      .insert(std::get<0>(getImpl()->fields).begin(), eleTypes.begin(),
              eleTypes.end());
  finalize();
}

llvm::StringRef StructType::getName() { return getImpl()->getName(); }

PointerType PointerType::get(mlir::Type elementType, bool isReference) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, CIL_Pointer, std::make_pair(elementType, isReference));
}

mlir::Type PointerType::getEleTy() const { return getImpl()->getElementType(); }

bool PointerType::isReference() const { return getImpl()->isRef(); }

mlir::Type ArrayType::getEleTy() const { return getImpl()->getElementType(); }

ArrayType::Shape ArrayType::getShape() const { return getImpl()->getShape(); }

void CIL::CILQualifiers::print(raw_ostream &os) {
  if (isUnsigned())
    os << "unsigned ";
  if (isConst())
    os << "const ";
  if (isVolatile())
    os << "volatile ";
  if (isExtern())
    os << "extern ";
  if (isStatic())
    os << "static ";
}

std::map<std::string, bool> structPrinterMap;
void CIL::CILDialect::printType(mlir::Type ty,
                                mlir::DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  switch (ty.getKind()) {
  case CIL_Void: {
    os << "void";
    break;
  }
  case CIL_NullPtrT: {
    os << "nullptr_t";
    break;
  }
  case CIL_Class: {
    auto classTy = ty.cast<CIL::ClassType>();
    os << "class<" << classTy.getName() << ">";
    break;
  }
  case CIL_VarArg: {
    os << "...";
  } break;
  case CIL_Char: {
    auto ptr = ty.cast<CIL::CharTy>();
    auto typeQual = ptr.getQualifiers();
    if (typeQual.isConst()) {
      os << "const_char";
    } else {
      os << "char";
    }
    break;
  }
  case CIL_Integer: {
    auto ptr = ty.cast<CIL::IntegerTy>();
    auto kind = ptr.getIntegerKind();
    switch (kind) {
    case BoolKind:
      os << "bool";
      break;
    case Char8Kind:
      os << "char8";
      break;
    case ShortKind:
      os << "short_int";
      break;
    case IntKind:
      os << "int";
      break;
    case LongKind:
      os << "long_int";
      break;
    case LongLongKind:
      os << "long_long_int";
      break;
    };
  } break;
  case CIL_FloatingTypeKind: {
    auto floatTy = ty.cast<CIL::FloatingTy>();
    auto kind = floatTy.getFloatingKind();
    auto typeQual = floatTy.getQualifiers();
    typeQual.print(os);
    switch (kind) {
    case Float:
      os << "float";
      break;
    case Double:
      os << "double";
      break;
    case LongDouble:
      os << "long_double";
      break;
    }
    break;
  }
  case CIL_Pointer: {
    auto ptr = ty.cast<CIL::PointerType>();
    auto eleTy = ptr.getEleTy();
    if (ptr.isReference()) {
      os << "ref_";
    }
    os << "pointer<";
    p.printType(eleTy);
    os << ">";
  } break;
  case CIL_Array: {
    auto array = ty.cast<CIL::ArrayType>();
    auto eleTy = array.getEleTy();
    os << "array<";
    for (auto &dim : array.getShape()) {
      if (dim == -1) {
        os << "? x ";
      } else {
        os << dim << " x ";
      }
    }
    p.printType(eleTy);
    os << ">";
  } break;
  case CIL_Struct: {
    auto array = ty.cast<CIL::StructType>();
    if (structPrinterMap.find(array.getName().str()) !=
            structPrinterMap.end() &&
        !structPrinterMap[array.getName().str()]) {
      os << "struct.";
      os << array.getName();
      os << "<>";
      break;
    }
    structPrinterMap[array.getName().str()] = false;
    os << "struct.";
    os << array.getName();
    os << "<";
    auto eleTy = array.getElementTypes();
    for (unsigned I = 0; I < eleTy.size(); ++I) {
      if (I > 0)
        os << ", ";
      /*
      if (eleTy[I] == ty) {
        os << "struct." << array.getName();
        continue;
      }

      unsigned numPtrs = 0;
      auto currType = eleTy[I];
      while (auto ptrTy = currType.dyn_cast_or_null<CIL::PointerType>()) {
        currType = ptrTy.getEleTy();
        numPtrs++;
      }
      if (auto eleStruct = currType.dyn_cast_or_null<CIL::StructType>()) {
        if (numPtrs > 0) {
          os << "struct." << eleStruct.getName();
          for (unsigned k = 0; k < numPtrs; k++) {
            os << "*";
          }
          continue;
        }
      }
      */
      p.printType(eleTy[I]);
    }
    os << ">";
    structPrinterMap[array.getName().str()] = true;
  } break;
  default:
    llvm_unreachable("Unknown CIL dialect type");
  }
}

static Type failure(DialectAsmParser &parser) {
  parser.emitError(parser.getNameLoc(), "Failed to parse CIL type");
  return Type();
}

static mlir::Type parseArrayType(const CILDialect &dialect,
                                 mlir::DialectAsmParser &parser) {
  if (parser.parseLess())
    return failure(parser);

  SmallVector<int64_t, 4> dims;
  if (parser.parseDimensionList(dims, false))
    return failure(parser);

  mlir::Type eleType;
  if (parser.parseType(eleType) || parser.parseGreater())
    return failure(parser);

  return CIL::ArrayType::get(dims, eleType);
}

std::map<std::string, mlir::Type> structParserMap;
static mlir::Type parseCILStructType(const CILDialect &dialect,
                                     DialectAsmParser &parser,
                                     StringRef structName) {
  llvm::SmallVector<mlir::Type, 2> elementTypes;
  std::string name = structName.drop_front(7).str();
  if (parser.parseLess())
    return failure(parser);

  if (structParserMap.find(name) != structParserMap.end() &&
      structParserMap[name]) {
    if (parser.parseGreater())
      return failure(parser);
    return structParserMap[name];
  }

  auto structTy =
      CIL::StructType::get(dialect.getContext(), elementTypes, name);

  structParserMap[name] = structTy;

  while (true) {
    if (succeeded(parser.parseOptionalGreater()))
      break;
    mlir::Type eleType;
    if (parser.parseType(eleType))
      return failure(parser);
    if (!structTy.isCompleteType())
      structTy.addMemberType(eleType);
    if (succeeded(parser.parseOptionalComma()))
      continue;
    if (parser.parseGreater())
      return failure(parser);
    break;
  }

  structTy.finalize();
  structParserMap[name] = nullptr;
  return structTy;
}

static mlir::Type parsePointerType(const CILDialect &dialect,
                                   DialectAsmParser &parser) {
  if (parser.parseLess())
    return failure(parser);

  mlir::Type pointeeType;
  if (parser.parseType(pointeeType) || parser.parseGreater())
    return failure(parser);

  return PointerType::get(pointeeType);
}

static mlir::Type parseFloatType(CILFloatingKind kind, MLIRContext *ctx,
                                 DialectAsmParser &parser) {
  return FloatingTy::get(kind, ctx);
}

static mlir::Type parseIntegerType(CILIntegerKind kind, MLIRContext *ctx,
                                   DialectAsmParser &parser) {
  return IntegerTy::get(kind, ctx);
}

mlir::Type CILDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return failure(parser);

  if (keyword == "pointer")
    return parsePointerType(*this, parser);

  if (keyword == "array")
    return parseArrayType(*this, parser);

  if (keyword == "char8")
    return parseIntegerType(CIL::Char8Kind, getContext(), parser);

  if (keyword == "int")
    return parseIntegerType(CIL::IntKind, getContext(), parser);

  if (keyword == "bool")
    return parseIntegerType(CIL::BoolKind, getContext(), parser);

  if (keyword == "long_int")
    return parseIntegerType(CIL::LongKind, getContext(), parser);

  if (keyword == "short_int")
    return parseIntegerType(CIL::ShortKind, getContext(), parser);

  if (keyword == "float")
    return parseFloatType(CIL::Float, getContext(), parser);

  if (keyword == "double")
    return parseFloatType(CIL::Double, getContext(), parser);

  if (keyword == "char")
    return CIL::CharTy::get(CILQualifiers(), getContext());

  if (keyword.substr(0, 7) == "struct.")
    return parseCILStructType(*this, parser, keyword);

  parser.emitError(parser.getNameLoc(), "CIL type: ") << keyword;
  return failure(parser);
}
