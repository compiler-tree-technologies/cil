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

#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace CIL;
using llvm::dbgs;

#define DEBUG_TYPE "fortran-ops"

//===----------------------------------------------------------------------===//
// CILDialect
//===----------------------------------------------------------------------===//

CILDialect::CILDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<ArrayType, StructType, IntegerTy, UnsignedChar, CharTy, PointerType,
           VarArgTy, FloatingTy, VoidType, NullPtrTTy, ClassType>();
  // addAttributes<SubscriptRangeAttr, StringInfoAttr>();
  addOperations<
#define GET_OP_LIST
#include "clang/cil/dialect/CIL/CILOps.cpp.inc"
      >();
}

void PrintOp::build(Builder *builder, OperationState &result,
                    ArrayRef<Value> args) {
  result.addOperands(args);
}

void GlobalOp::build(Builder *builder, OperationState &result, mlir::Type type,
                     bool isConstant, StringRef name, Attribute value) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.types.push_back(type);
  if (isConstant)
    result.addAttribute("constant", builder->getUnitAttr());
  if (value)
    result.addAttribute("value", value);
  result.addRegion();
}

void GlobalOp::build(Builder *builder, OperationState &result, mlir::Type type,
                     bool isConstant, StringRef name,
                     SymbolRefAttr constructSym) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  result.types.push_back(type);
  if (isConstant)
    result.addAttribute("constant", builder->getUnitAttr());
  result.addAttribute("constructSym", constructSym);
  result.addRegion();
}

void CILLoadOp::build(Builder *builder, OperationState &result,
                      Value pointerVal, ArrayRef<Value> indicesList) {

  result.addOperands(pointerVal);
  result.addOperands(indicesList);
  result.addTypes(pointerVal.getType().cast<CIL::PointerType>().getEleTy());
}

void CILStoreOp::build(Builder *builder, OperationState &result,
                       Value valToStore, Value pointerVal,
                       ArrayRef<Value> indicesList) {
  result.addOperands({valToStore, pointerVal});
  result.addOperands(indicesList);
}

void GetPointerToOp::build(Builder *builder, OperationState &result,
                           Value CILRef) {
  result.addOperands(CILRef);
  auto CILPointerType = CILRef.getType().cast<CIL::PointerType>();
  auto eleTy = CILPointerType.getEleTy();
  auto ptrType = PointerType::get(eleTy);
  result.addTypes(ptrType);
}

void CastToMemRefOp::build(Builder *builder, OperationState &result,
                           Value CILRef) {
  result.addOperands(CILRef);
  auto CILPointerType = CILRef.getType().cast<CIL::PointerType>();
  auto eleTy = CILPointerType.getEleTy();

  llvm::SmallVector<int64_t, 2> shape;
  if (auto CILArrType = eleTy.dyn_cast_or_null<CIL::ArrayType>()) {
    eleTy = CILArrType.getEleTy();

    auto dims = CILArrType.getShape();
    if (!CILPointerType.getEleTy().cast<CIL::ArrayType>().hasStaticShape()) {
      auto numBounds = dims.size();
      llvm::SmallVector<int64_t, 2> shape(numBounds, -1);
      llvm::SmallVector<int64_t, 2> strides(
          numBounds, mlir::MemRefType::getDynamicStrideOrOffset());
      int64_t offset = mlir::MemRefType::getDynamicStrideOrOffset();
      auto affineMap = mlir::makeStridedLinearLayoutMap(strides, offset,
                                                        CILRef.getContext());
      auto memRefType = mlir::MemRefType::get(shape, eleTy, affineMap);
      result.addTypes(memRefType);
      return;
    }

    for (auto &dim : dims) {
      shape.push_back(dim);
    }

    llvm::SmallVector<mlir::AffineExpr, 2> exprs;
    unsigned dimVal = 0;
    mlir::AffineExpr strideExpr;
    unsigned long sizeTillNow = 1, offsetTillNow = 0;

    for (unsigned I = 0; I < dims.size(); ++I) {
      auto dim = builder->getAffineDimExpr(dimVal++);
      if (I == 0) {
        strideExpr = dim;
        offsetTillNow = 0;
        continue;
      }
      sizeTillNow *= shape[I - 1];
      auto sizeVal = builder->getAffineConstantExpr(sizeTillNow);
      auto currExpr =
          mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mul, dim, sizeVal);
      strideExpr = mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Add,
                                               currExpr, strideExpr);
    }
    strideExpr = mlir::getAffineBinaryOpExpr(
        mlir::AffineExprKind::Add, strideExpr,
        builder->getAffineConstantExpr(offsetTillNow));

    auto affineMap = mlir::AffineMap::get(shape.size(), 0, {strideExpr});
    auto memRefTypeMemRefType = MemRefType::get(shape, eleTy, affineMap);
    result.addTypes(memRefTypeMemRefType);
    return;
  }

  auto memRefTypeMemRefType = MemRefType::get(shape, eleTy);
  result.addTypes(memRefTypeMemRefType);
}

GlobalOp CIL::GlobalAddressOfOp::getGlobal() {
  auto module = getParentOfType<mlir::ModuleOp>();
  assert(module && "unexpected operation outside of a module");
  return module.lookupSymbol<GlobalOp>(global_name());
}

void CILCallOp::build(Builder *builder, OperationState &result, FuncOp callee,
                      ArrayRef<Value> operands) {
  auto sym = builder->getSymbolRefAttr(callee);
  CILCallOp::build(builder, result, sym, callee.getType().getResults(),
                   operands);
}

void CILCallOp::build(Builder *builder, OperationState &result,
                      SymbolRefAttr symbolScopeList, ArrayRef<Type> results,
                      ArrayRef<Value> operands) {
  result.addOperands(operands);
  result.addAttribute("callee", symbolScopeList);
  result.addAttribute("num_operands",
                      builder->getI32IntegerAttr(operands.size()));
  result.addTypes(results);
}

void MemberCallOp::build(Builder *builder, OperationState &result,
                         Value baseClassObj, SymbolRefAttr symbolScopeList,
                         ArrayRef<Type> results, ArrayRef<Value> operands) {
  result.addOperands(baseClassObj);
  result.addOperands(operands);
  result.addAttribute("callee", symbolScopeList);
  result.addAttribute("num_operands",
                      builder->getI32IntegerAttr(operands.size()));
  result.addTypes(results);
}

void CILDialect::printAttribute(mlir::Attribute attr,
                                mlir::DialectAsmPrinter &p) const {
  assert(false && "unknown attribute type to print");
}

// NOTE: Code copied from MLIR

// Returns an array of mnemonics for CmpFPredicates indexed by values thereof.
static inline const char *const *getCmpFPredicateNames() {
  static const char *predicateNames[] = {
      /*AlwaysFalse*/ "false",
      /*OEQ*/ "oeq",
      /*OGT*/ "ogt",
      /*OGE*/ "oge",
      /*OLT*/ "olt",
      /*OLE*/ "ole",
      /*ONE*/ "one",
      /*ORD*/ "ord",
      /*UEQ*/ "ueq",
      /*UGT*/ "ugt",
      /*UGE*/ "uge",
      /*ULT*/ "ult",
      /*ULE*/ "ule",
      /*UNE*/ "une",
      /*UNO*/ "uno",
      /*AlwaysTrue*/ "true",
  };
  static_assert(std::extent<decltype(predicateNames)>::value ==
                    (size_t)CmpFPredicate::NumPredicates,
                "wrong number of predicate names");
  return predicateNames;
}
// NOTE: Copy from MLIR source end.

static void printCILLoad(CILLoadOp op, OpAsmPrinter &p) {
  p << "cil.load " << op.getPointer();
  SmallVector<Value, 2> operands(op.getIndices());
  if (!operands.empty()) {
    p << '[';
    for (unsigned I = 0; I < operands.size(); ++I) {
      if (I > 0) {
        p << ", ";
      }
      p << operands[I];
    }
    p << ']';
  }
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getPointer().getType() << " -> " << op.getType();
}

static void printCILStore(CILStoreOp op, OpAsmPrinter &p) {
  p << "cil.store " << op.getValueToStore();
  p << ", " << op.getPointer();
  SmallVector<Value, 2> operands(op.getIndices());
  if (!operands.empty()) {
    p << '[';
    for (unsigned I = 0; I < operands.size(); ++I) {
      if (I > 0) {
        p << ", ";
      }
      p << operands[I];
    }
    p << ']';
  }
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getPointer().getType();
}

Region &ForLoopOp::getLoopBody() { return region(); }

bool ForLoopOp::isDefinedOutsideOfLoop(Value value) {
  return !region().isAncestor(value.getParentRegion());
}

LogicalResult ForLoopOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (auto *op : ops)
    op->moveBefore(this->getOperation());
  return success();
}

// Parsers
static ParseResult failure(OpAsmParser &parser) {
  return parser.emitError(parser.getNameLoc(), "Failed to parse CIL operation");
}

static ParseResult parseCILGlobalOp(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr name;
  // FIXME: not using sym_name attribute read here.
  SmallVector<mlir::NamedAttribute, 2> attrs;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(), attrs))
    return failure(parser);

  mlir::Type type;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure(parser);

  Region &initRegion = *result.addRegion();
  if (parser.parseOptionalRegion(initRegion, {}, {}))
    return failure(parser);

  // result.addAttribute(SymbolTable::getSymbolAttrName(), name);
  result.types.push_back(type);
  return success();
}

static ParseResult parserCILConstantOp(OpAsmParser &parser,
                                       OperationState &result) {

  Attribute attr;
  mlir::Type type;
  if (parser.parseLParen() ||
      parser.parseAttribute(attr, "value", result.attributes) ||
      parser.parseRParen() || parser.parseColon() || parser.parseType(type))
    return failure(parser);

  result.types.push_back(type);
  return success();
}

static ParseResult parseCILReturnOp(OpAsmParser &parser,
                                    OperationState &result) {

  SmallVector<OpAsmParser::OperandType, 1> operands;
  Type type;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure(parser);
  if (operands.empty())
    return success();

  if (parser.parseColonType(type) ||
      parser.resolveOperand(operands[0], type, result.operands))
    return failure(parser);

  return success();
}

static ParseResult parseCILAddressOfOp(OpAsmParser &parser,
                                       OperationState &result) {
  StringAttr name;
  if (parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure(parser);

  mlir::Type type;
  if (parser.parseColonType(type))
    return failure(parser);

  auto builder = parser.getBuilder();
  result.addAttribute("global_name", builder.getSymbolRefAttr(name.getValue()));
  result.types.push_back(type);
  return success();
}

static ParseResult parseCILBinaryOp(OpAsmParser &parser,
                                    OperationState &result) {
  OpAsmParser::OperandType lhs, rhs;
  Type type;

  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseColonType(type))
    return failure(parser);

  if (parser.resolveOperand(lhs, type, result.operands) ||
      parser.resolveOperand(rhs, type, result.operands))
    return failure(parser);

  result.addTypes(type);
  return success();
}

static mlir::Type getElementType(Type type) {
  assert(type);
  if (auto ptrType = type.dyn_cast_or_null<CIL::PointerType>())
    return ptrType.getEleTy();

  llvm_unreachable("Unhandled type\n");
}

static ParseResult parseCILStoreOp(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::OperandType addr, value;
  Type type;

  // TODO Yet to handle array stores(with indices)
  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(addr) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(type))
    return failure(parser);

  auto eleTy = getElementType(type);

  if (parser.resolveOperand(value, eleTy, result.operands) ||
      parser.resolveOperand(addr, type, result.operands))
    return failure(parser);

  return success();
}

static CmpIPredicate getPredicateByName(StringRef pred) {
  if (pred == "eq")
    return CmpIPredicate::eq;
  if (pred == "ne")
    return CmpIPredicate::ne;
  if (pred == "slt")
    return CmpIPredicate::slt;
  if (pred == "sle")
    return CmpIPredicate::sle;
  if (pred == "sgt")
    return CmpIPredicate::sgt;
  if (pred == "sge")
    return CmpIPredicate::sge;
  if (pred == "ult")
    return CmpIPredicate::ult;
  if (pred == "ule")
    return CmpIPredicate::ule;
  if (pred == "ugt")
    return CmpIPredicate::ugt;
  if (pred == "uge")
    return CmpIPredicate::uge;

  llvm_unreachable("Unknown predicate");
}

static ParseResult parseCILCmpOp(OpAsmParser &parser, OperationState &result,
                                 bool isInt = true) {
  Attribute predicateAttr;
  mlir::Type opType, type;
  OpAsmParser::OperandType lhs, rhs;
  StringRef predicateAttrName;

  if (isInt)
    predicateAttrName = CmpIOp::getPredicateAttrName();
  else
    predicateAttrName = CmpFOp::getPredicateAttrName();

  if (parser.parseAttribute(predicateAttr, predicateAttrName,
                            result.attributes) ||
      parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseColonType(opType) ||
      parser.parseArrow() || parser.parseType(type) ||
      parser.resolveOperand(lhs, opType, result.operands) ||
      parser.resolveOperand(rhs, opType, result.operands))
    return failure(parser);

  // Rewrite string attribute to an enum value.
  StringRef predicateName = predicateAttr.cast<StringAttr>().getValue();
  if (isInt) {
    auto predicate = getPredicateByName(predicateName);
    result.attributes[0].second =
        parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(predicate));
  } else {
    auto predicate = CmpFOp::getPredicateByName(predicateName);
    if (predicate == CmpFPredicate::NumPredicates)
      return parser.emitError(parser.getNameLoc(),
                              "unknown comparison predicate \"" +
                                  predicateName + "\"");

    result.attributes[0].second =
        parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(predicate));
  }

  result.types.push_back(type);
  return success();
}

static ParseResult parseCILAllocaOp(OpAsmParser &parser,
                                    OperationState &result) {
  mlir::Type memType, type;
  if (parser.parseType(memType) || parser.parseColonType(type))
    return failure(parser);

  result.types.push_back(type);
  return success();
}

static ParseResult parseCILLoadOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType addr;
  Type type, opType;

  // TODO Yet to handle array load(with indices)
  if (parser.parseOperand(addr) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(opType) || parser.parseArrow() || parser.parseType(type))
    return failure(parser);

  if (parser.resolveOperand(addr, opType, result.operands))
    return failure(parser);

  result.types.push_back(type);
  return success();
}

static ParseResult parseCILCallOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> operands;
  Type type;
  SymbolRefAttr funcAttr;
  if (parser.parseAttribute(funcAttr, "callee", result.attributes))
    return failure(parser);

  if (parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseColonType(type))
    return failure(parser);

  auto funcType = type.cast<FunctionType>();
  auto inputTypes = funcType.getInputs();
  assert(inputTypes.size() == operands.size());
  for (unsigned i = 0; i < operands.size(); ++i) {
    if (parser.resolveOperand(operands[i], inputTypes[i], result.operands))
      return failure(parser);
  }

  result.addAttribute("num_operands",
                      parser.getBuilder().getI32IntegerAttr(operands.size()));

  for (auto resType : funcType.getResults())
    result.types.push_back(resType);

  return success();
}

static ParseResult parseCILUnreachableOp(OpAsmParser &parser,
                                         OperationState &result) {
  return success();
}

static ParseResult parseCILGEPOp(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::OperandType, 8> operands;
  OpAsmParser::OperandType pointer;

  mlir::Type idxType, ptrType, type;
  if (parser.parseOperand(pointer) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Square) ||
      parser.parseColon() || parser.parseLParen() ||
      parser.parseType(ptrType) || parser.parseComma() ||
      parser.parseType(idxType) || parser.parseRParen() ||
      parser.parseArrow() || parser.parseType(type))
    return failure(parser);

  if (parser.resolveOperand(pointer, ptrType, result.operands) ||
      parser.resolveOperands(operands, idxType, result.operands))
    return failure(parser);

  result.types.push_back(type);
  return success();
}

#define GET_OP_CLASSES
#include "clang/cil/dialect/CIL/CILOps.cpp.inc"
