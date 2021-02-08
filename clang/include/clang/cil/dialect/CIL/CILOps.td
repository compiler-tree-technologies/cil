// Copyright (c) 2019, Compiler Tree Technologies Pvt Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// Defines MLIR CIL operations.
// FIXME:
// 1. Missing verifier , parser functions.
// 2. Summary and description of operations.
// 3. Move non-trivial functions to cpp file.

#ifndef CIL_OPS
#define CIL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Analysis/CallInterfaces.td"
include "mlir/Transforms/LoopLikeInterface.td"

def CIL_Dialect : Dialect {
  let name = "cil";
  let cppNamespace = "CIL";
}

def CIL_RefType : Type<CPred<"$_self.isa<CIL::PointerType>()">,
    "Ref type">;

def CIL_PointerType : Type<CPred<"$_self.isa<CIL::PointerType>() || $_self.isa<CIL::NullPtrTTy>()">,"Pointer type">;

def CIL_CILType : Type<CPred<"$_self.isa<CIL::ArrayType>()">,
    "array type">;

def CIL_IntegerType:  Type<CPred<"$_self.isa<CIL::IntegerTy>()">, "integer type">;

def CIL_CharType:  Type<CPred<"$_self.isa<CIL::CharTy>()">, "char type">;

def CIL_ClassType:  Type<CPred<"$_self.isa<CIL::ClassType>()">, "class type">;

def CIL_FloatType:  Type<CPred<"$_self.isa<CIL::FloatingTy>()">, "float type">;

def CIL_BoolType : Type<CPred<"$_self.isa<CIL::IntegerTy>() && $_self.cast<CIL::IntegerTy>().isBoolType()">,
    "bool type">, BuildableType<"CIL::IntegerTy::get(CIL::BoolKind, $_builder.getContext())">;

def CIL_ClassPointerType : Type<CPred<"$_self.isa<CIL::PointerType>() && $_self.cast<CIL::PointerType>().getEleTy().isa<CIL::ClassType>()">,
    "class pointer type">;

def CIL_NullPtrTType:  Type<CPred<"$_self.isa<CIL::NullPtrTTy>()">, "nullptr_t type">;

def AnyRef : TypeConstraint<CIL_RefType.predicate, "any ref">;

def AnyClass : TypeConstraint<CIL_ClassType.predicate, "any class">;

def AnyPointer : TypeConstraint<CIL_PointerType.predicate, "any pointer">;

def AnyClassPointer : TypeConstraint<CIL_ClassPointerType.predicate, "any class pointer">;

def AnyRefLike : TypeConstraint<Or<[CIL_RefType.predicate, CIL_PointerType.predicate]>, "any ref like">;

def AnyArray : TypeConstraint<CIL_CILType.predicate, "any array">;

def AnyCharInt : TypeConstraint<Or<[CIL_IntegerType.predicate, CIL_CharType.predicate]>, "any int or char">;

def AnyCILInteger : TypeConstraint<CIL_IntegerType.predicate, "any integer">;

def AnyCILFloat : TypeConstraint<CIL_FloatType.predicate, "any float">;

def BoolType : TypeConstraint<CIL_BoolType.predicate, "boolType">;

// Base class for CIL dialect ops.
class CIL_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<CIL_Dialect, mnemonic, traits> {

  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

// Include the class related operations.
include "CILClassOps.td"

def CIL_AllocaOp : CIL_Op<"alloca", []> {
  let summary = "static memory allocation operation";
  let description = [{
    Allocates a new cil.PointerType of specified type with a fixed base pointer location
    in memory. usually allocates in stack.
    Eg.:
    %A = alloca() : cil.PointerType<i32>
  }];

  let arguments = (ins OptionalAttr<SymbolRefAttr>:$constructSym, Variadic<AnyType>:$constructArgs);

  let results = (outs CIL_RefType);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef name, Type type", [{
       result.types.push_back(CIL::PointerType::get(type));
       result.addAttribute("name", builder->getStringAttr(name));
     }]>,
     OpBuilder<
    "Builder *builder, OperationState &result, StringRef name, Type type, SymbolRefAttr constructSymAttr, ArrayRef<mlir::Value> constructArgs", [{
       result.types.push_back(CIL::PointerType::get(type));
       result.addAttribute("name", builder->getStringAttr(name));
       result.addAttribute("constructSym", constructSymAttr);
       result.addOperands(constructArgs);
     }]
  >];

  let extraClassDeclaration = [{
    CIL::PointerType getType() { return getResult().getType().cast<CIL::PointerType>(); }

    mlir::Type getAllocatedType() { return getType().getEleTy(); }

    StringRef getName() {
      return getAttrOfType<StringAttr>("name").getValue();
    }
  }];

  let verifier = ? ;
  let printer = [{
    p << "cil.alloca " << getType().getEleTy();
    if (constructSymAttr()) {
      p << ", " << constructSymAttr();
      p <<  "(";
      SmallVector<Value , 2> ops(constructArgs());
      for (unsigned I = 0; I < ops.size(); ++I) {
        if (I > 0)
         p << ", ";
        p << ops[I];
      }
      p << ")";
    }
    p << " : " << getType();
  }];
  let parser = "return parseCILAllocaOp(parser, result);";
}

def CIL_DeallocaOp : CIL_Op<"deallocate"> {
  let summary = "memory deallocation operation";
  let description = [{
    The "dealloc" operation frees the region of memory referenced by a ref
    only if it is malloced.
  }];

  let arguments = (ins AnyRef:$ref);

  let verifier = ? ;
  let printer = [{
    p << "cil.deallocate " << ref();
    p.printOptionalAttrDict(getAttrs());
    p << " : " << ref().getType();
  }];
  let parser = ? ;
}

def CIL_CILCmpIOp : CIL_Op<"cmpi"> {
  let summary = "Compares two integer returns boolean result";
  let description = [{

  }];

  let arguments = (ins AnyCILInteger:$lhs, AnyCILInteger:$rhs);
  let results = (outs BoolType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpIPredicate predicate,"
    "Value lhs, Value rhs", [{
      result.addOperands({lhs, rhs});
      assert(lhs.getType()==rhs.getType());
      result.types.push_back(CIL::IntegerTy::get(BoolKind,rhs.getType().getContext()));
      result.addAttribute(
      CmpIOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
    }]>];

  let extraClassDeclaration = [{
    CmpIPredicate getPredicate() {
      return (CmpIPredicate)getAttrOfType<IntegerAttr>(CmpIOp::getPredicateAttrName())
          .getInt();
    }

  }];

  // TODO: add compatibility tests.
  let verifier = ? ;

  let printer = [{
    p << "cil.cmpi " << "\"";
    p << stringifyCmpIPredicate(getPredicate()) << "\"";
    p << " " << lhs() << ", " << rhs();
    p << " : " << lhs().getType() << " -> " << getResult().getType();
  }];
  let parser = "return parseCILCmpOp(parser, result);";

  //let assemblyFormat = "$predicate `,` $lhs `,` $rhs  `:` type($lhs) `->` type($res)";
}

def CIL_CILCmpFOp : CIL_Op<"cmpf"> {
  let summary = "Compares two integer like arrays and returns boolean result";
  let description = [{

  }];

  let arguments = (ins AnyCILFloat:$lhs, AnyCILFloat:$rhs);
  let results = (outs BoolType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      result.addOperands({lhs, rhs});
      assert(lhs.getType()==rhs.getType());
      result.types.push_back(CIL::IntegerTy::get(BoolKind,rhs.getType().getContext()));
      //CIL::ArrayType::Shape shape = lhs.getType().cast<ArrayType>().getShape();
      //result.types.push_back(CIL::ArrayType::get(shape,builder->getIntegerType(1)));
      result.addAttribute(
      CmpFOp::getPredicateAttrName(),
      builder->getI64IntegerAttr(static_cast<int64_t>(predicate)));
    }]>];

  let extraClassDeclaration = [{
    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<IntegerAttr>(CmpFOp::getPredicateAttrName())
          .getInt();
    }

  }];

  // TODO: add compatibility tests.
  let verifier = ? ;

  let printer = [{
    p << "cil.cmpf " << "\"";
    p << getCmpFPredicateNames()[(unsigned)getPredicate()] << "\"";
    p << " " << lhs() << ", " << rhs();
    p << " : " << lhs().getType() << " -> " << getResult().getType();
  }];

  let parser = "return parseCILCmpOp(parser, result, false);";
}


class CIL_PointerBinOp<string mnemonic, list<OpTrait> traits = []> :
  CIL_Op<mnemonic, traits> {

  let summary = "Binary Operations";
  let description = [{

  }];

  // TODO: make it any scalar type.
  let arguments = (ins AnyPointer:$lhs, AnyCILInteger:$rhs);
  let results = (outs AnyPointer:$res);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value opd1, Value opd2",
    [{
      result.addOperands({opd1, opd2});
      result.types.push_back(opd1.getType());
    }]>];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = [{
    p << getOperationName();
    p << " " << lhs() << ", " << rhs();
    p << " : (" << lhs().getType() << "," << rhs().getType() << ")";
    p << " -> " << getResult().getType();
  }];
  let parser = ? ;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

def CIL_CILPointerAddOp : CIL_PointerBinOp<"pointer_index", []>;
//def CIL_CILPointerSubOp : CIL_PointerBinOp<"pointer_sub", []>;

class CIL_BinOp<string mnemonic, list<OpTrait> traits = []> :
  CIL_Op<mnemonic, traits> {

  let summary = "Binary Operations";
  let description = [{

  }];

  // TODO: make it any scalar type.
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value opd1, Value opd2",
    [{
      auto type1 = opd1.getType();
      auto type2 = opd2.getType();
      auto intTy1 = type1.dyn_cast<CIL::IntegerTy>();
      auto intTy2 = type2.dyn_cast<CIL::IntegerTy>();
      if (getOperationName() !=  "cil.shr" && getOperationName() != "cil.shl")
      assert((type1 == type2) || ( intTy1 && intTy2 && (intTy1.getIntegerKind() == intTy2.getIntegerKind())));
      result.addOperands({opd1, opd2});
      result.types.push_back(opd2.getType());
    }]>];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = [{
    p << getOperationName();
    p << " " << lhs() << ", " << rhs();
    p << " : " << getResult().getType();
  }];
  let parser = "return parseCILBinaryOp(parser, result);";
}

def CIL_CILAddIOp : CIL_BinOp<"addi", [Commutative]>;
def CIL_CILAddFOp : CIL_BinOp<"addf", []>;
def CIL_CILSubIOp : CIL_BinOp<"subi", []>;
def CIL_CILSubFOp : CIL_BinOp<"subf", []>;
def CIL_CILMulIOp : CIL_BinOp<"muli", [Commutative]>;
def CIL_CILMulFOp : CIL_BinOp<"mulf", []>;
def CIL_CILDivIOp : CIL_BinOp<"divi", []>;
def CIL_CILDivFOp : CIL_BinOp<"divf", []>;
def CIL_CILModIOp : CIL_BinOp<"modi", []>; // Only for singed
def CIL_CILModFOp : CIL_BinOp<"modf", []>;
def CIL_CILModUIOp : CIL_BinOp<"modui", []>; // only for unsigned

// Bitwise ops
def CIL_CILShlOp : CIL_BinOp<"shl", []>;
def CIL_CILAShrOp : CIL_BinOp<"shr", []>;
def CIL_CILAndOp : CIL_BinOp<"and", []>;
def CIL_CILOrOp  : CIL_BinOp<"or", []>;
def CIL_CILXOrOp : CIL_BinOp<"xor", []>;

class CIL_LogicalBinOp<string mnemonic, list<OpTrait> traits = []> :
  CIL_Op<mnemonic, traits> {

  let summary = "logical binary Operations";
  let description = [{

  }];

  // TODO: make it any scalar type.
  let arguments = (ins CIL_BoolType:$lhs, CIL_BoolType:$rhs);
  let results = (outs CIL_BoolType:$res);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value opd1, Value opd2",
    [{
      result.addOperands({opd1, opd2});
      result.types.push_back(opd1.getType());
    }]>];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = [{
    p << getOperationName();
    p << " " << lhs() << ", " << rhs();
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";

}

def CIL_CILLogicalAndOp : CIL_LogicalBinOp<"logical_and",[]>;
def CIL_CILLogicalOrOp : CIL_LogicalBinOp<"logical_or",[]>;

def CIL_CILLoadOp : CIL_Op<"load", []> {
  let summary = "load operation";
  let description = [{CIL load pointer type operation}];
  let results = (outs AnyType:$res);

  let arguments = (ins AnyRefLike :$pointer, Variadic<Index>:$indices);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value pointerVal, ArrayRef<Value> indices = {}">];

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
    Type getType() { return getResult().getType(); }
  }];

  let verifier = ? ;
  let printer = [{ ::printCILLoad(*this, p);}];
  let parser = "return parseCILLoadOp(parser, result);";
}

def CIL_CILStoreOp : CIL_Op<"store", []> {
  let summary = "store operation";
  let description = [{CIL store for pointer type}];
  let arguments = (ins AnyType:$valueToStore, AnyRefLike:$pointer, Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value valueToStore, Value pointerVal, ArrayRef<Value> indices = {}">];

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    mlir::Value getValueToStore() { return valueToStore(); }
    operand_range getIndices() { return {operand_begin() + 2, operand_end()}; }
    }];

  let verifier = ? ;
  let printer = [{ ::printCILStore(*this, p);}];
  let parser = "return parseCILStoreOp(parser, result);";
}

def CIL_PrintOp : CIL_Op<"print", []> {
  let summary = "print operation";
  let description = [{Operation for CIL print statement }];
  let arguments = (ins Variadic<AnyType> : $args);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"Builder *builder, OperationState &result, "
                            "ArrayRef<Value > args">];
  let extraClassDeclaration = [{
    operand_range getArgs() { return args(); }
    }];

  let verifier = ? ;
  let printer = [{
    p << "cil.print ";
    SmallVector<Value , 2> operands(getArgs());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

// Way to represent the constant.
def CIL_CILConstantOp : CIL_Op<"constant", []> {
  let summary = "constant operation";
  let description = [{ CIL constant}];

  let arguments = (ins AnyAttr:$value);
  // FIXME: More specific type.
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Attribute value", [{
      result.addAttribute("value", value);
    }]>,
    OpBuilder<
    "Builder *builder, OperationState &result, Type type, int64_t val", [{
      result.addTypes(type);
      result.addAttribute("value", builder->getI32IntegerAttr(val));
    }]>
  ];



  let verifier = ? ;
  let printer = [{ p << getOperationName()
                      << "( " <<  value() << " )";
                    p << ": " << getResult().getType();
                }];

  let parser = "return parserCILConstantOp(parser, result);";
}

// Way to represent the constant string.
// Returns character array.
def CIL_StringOp : CIL_Op<"constant_string", []> {
  let summary = "string operation";
  let description = [{ CIL constant string}];

  let arguments = (ins StrAttr:$value);
  // FIXME: More specific type.
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringAttr attr", [{
      result.addAttribute("value", attr);
    }]>];

  let verifier = ? ;
  let printer = [{ p << getOperationName()
                      << "(\"" << value() << "\")";
                    p << ": " << getResult().getType();
                }];

  let parser = ? ;
}

// Used for CIL::ast::CastExpr
def CIL_CastOp : CIL_Op<"cast", []> {
  let summary = "Cast operation";
  let description = [{CIL cast expression}];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value v", [{
      result.addOperands(v);
    }]>];

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

// MemReCILastOp is a nicer name, but it's already used by standard-ops.
def CIL_CastToPointerOp : CIL_Op<"cast_to_pointer", []> {
  let summary = " Pointer cast operation";
  let description = [{Converts the cil.CIL::PointerType to cil.PointerType}];
  let arguments = (ins AnyRef:$CILRef);
  let results = (outs AnyPointer);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value ref", [{
      result.addOperands(ref);
      auto type = ref.getType().cast<CIL::PointerType>().getEleTy();
      result.addTypes(PointerType::get(type));
    }]>];

  let extraClassDeclaration = [{
    mlir::Value getRef() { return CILRef(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.cast_to_memref " << getRef() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

// Operation to case Int's with different qualifiers
def CIL_IntCastOp : CIL_Op<"int_cast", []> {
  let summary = " Integer cast operation";
  let description = [{cil.IntegerTy<a> cast to cil.IntegerTy<b>}];
  let arguments = (ins AnyCharInt:$inval);
  let results = (outs AnyCharInt:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.int_cast " << inval() << " : "
      << inval().getType() << " -> " << getResult().getType();
  }];

  let parser = ?;
  let assemblyFormat = "$inval `:` type($inval) attr-dict `->` type($res)";
}

def CIL_ZeroExtendOp : CIL_Op<"zero_extend", []> {
  let summary = " Integer cast operation";
  let description = [{cil.IntegerTy<a> extended to cil.IntegerTy<b>}];
  let arguments = (ins AnyCharInt:$inval);
  let results = (outs AnyCharInt:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.zero_extend " << inval() << " : "
      << inval().getType() << " -> " << getResult().getType();
  }];

  let parser = ?;
  let assemblyFormat = "$inval `:` type($inval) attr-dict `->` type($res)";
}

def CIL_SignExtendOp : CIL_Op<"sign_extend", []> {
  let summary = " Integer cast operation";
  let description = [{cil.IntegerTy<a> extended to cil.IntegerTy<b>}];
  let arguments = (ins AnyCILInteger:$inval);
  let results = (outs AnyCILInteger:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.sign_extend " << inval() << " : "
      << inval.getType() << " -> " << getResult().getType();
  }];

  let parser = ?;
  let assemblyFormat = "$inval attr-dict `:` type($inval) `->` type($res)";
}

def CIL_TruncateOp : CIL_Op<"truncate", []> {
  let summary = " Integer cast operation";
  let description = [{cil.IntegerTy<a> extended to cil.IntegerTy<b>}];
  let arguments = (ins AnyCharInt:$inval);
  let results = (outs AnyCharInt:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.truncate " << inval() << " : "
      << inval().getType() << " -> "
      << getResult().getType();
  }];

  let parser = ?;
  let assemblyFormat = "$inval attr-dict `:` type($inval) `->` type($res)";
}

class CIL_FPCastOp<string mnemonic, list<OpTrait> traits = []> :
  CIL_Op<mnemonic, traits> {

  let summary = "FP cast operations";
  let description = [{ "Base class for all fp cast operations" }];

  let arguments = (ins AnyCILFloat:$inval);
  let results = (outs AnyCILFloat:$res);

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
    mlir::Type getDestType() { return getResult().getType(); }
  }];

  let verifier = ?;
  let printer = [{
    p << getOperationName() << " " << inval() << " : "
      << inval().getType() << " -> " << getDestType();
  }];

  let parser = ?;
  let assemblyFormat = "$inval attr-dict `:` type($inval) `->` type($res)";
}

def CIL_CILFPTruncOp : CIL_FPCastOp<"fp_trunc", []>;
def CIL_CILFPExtOp : CIL_FPCastOp<"fp_ext", []>;

def CIL_CILSIToFPOp : CIL_Op<"si_to_fp", []> {
  let summary = " Pointer cast operation";
  let description = [{int to floating type}];
  let arguments = (ins AnyCILInteger:$int_val);
  let results = (outs AnyCILFloat:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return int_val(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.si_to_fp " << int_val() << " : "
      << int_val().getType() << " -> " << getResult().getType();
  }];

  let parser = ?;
  let assemblyFormat = "$int_val attr-dict `:` type($int_val) `->` type($res)";
}

def CIL_CILFPToSIOp : CIL_Op<"fp_to_si", []> {
  let summary = " FloatingPoint to integer operation";
  let description = [{Float to integer type}];
  let arguments = (ins AnyCILFloat:$int_val);
  let results = (outs AnyCILInteger:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return int_val(); }
  }];

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$int_val attr-dict `:` type($int_val) `->` type($res)";
}

def CIL_CILIntToPtrOp : CIL_Op<"int_to_ptr", []> {
  let summary = " Pointer cast operation";
  let description = [{int to floating type}];
  let arguments = (ins AnyCILInteger:$int_val);
  let results = (outs AnyPointer:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return int_val(); }
  }];

  let verifier = ?;

  let parser = ?;
  let assemblyFormat = "$int_val attr-dict `:` type($int_val) `->` type($res)";
}

def CIL_CILPtrToIntOp : CIL_Op<"ptr_to_int", []> {
  let summary = " Pointer cast operation";
  let description = [{int to floating type}];
  let arguments = (ins AnyPointer:$ptr_val);
  let results = (outs AnyCILInteger:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return ptr_val(); }
  }];

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$ptr_val attr-dict `:` type($ptr_val) `->` type($res)";
}

def CIL_PointerBitCastOp : CIL_Op<"pointer_bitcast", []> {
  let summary = " Pointer cast operation";
  let description = [{Converts the cil.PointerType<a> to cil.PointerType<b>}];
  let arguments = (ins AnyPointer:$pointer_val);
  let results = (outs AnyPointer:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return pointer_val(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.pointer_bitcast " << pointer_val() << " : "
      << pointer_val().getType() << " -> " << getResult().getType();
  }];

  let parser = ?;
  let assemblyFormat = "$pointer_val attr-dict `:` type($pointer_val) `->` type($res)";
}

def CIL_CastToStdBoolOp : CIL_Op<"cast_to_std_i1", []> {
  let summary = "Cast to std i1";
  let description = [{Cast to std i1 from cil.bool}];
  let arguments = (ins BoolType:$invalue);
  let results = (outs I1);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value val", [{
      result.addOperands(val);
      result.addTypes(builder->getIntegerType(1));
    }]>];

  let extraClassDeclaration = [{
    mlir::Value getValue() { return invalue(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.cast_to_std_i1 " << invalue() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

// MemReCILastOp is a nicer name, but it's already used by standard-ops.
def CIL_CastToMemRefOp : CIL_Op<"cast_to_memref", []> {
  let summary = "Experimental MemRef cast operation";
  let description = [{Converts the cil.PointerType to std.memref}];
  let arguments = (ins AnyRef:$CILRef);
  let results = (outs AnyMemRef);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value ref">];

  let extraClassDeclaration = [{
    mlir::Value getRef() { return CILRef(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.cast_to_memref " << getRef() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

// Similar to LLVMAddressOfOp.
// TODO: rename as global_address_of_op.
def CIL_GlobalAddressOfOp
    : CIL_Op<"global_address_of", []> {

  let summary = " Address of Operation";
  let description = [{Global Address of operation}];
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs AnyPointer);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, PointerType resType,"
              "StringRef name, ArrayRef<NamedAttribute> attrs = {}", [{
      result.addAttribute("global_name", builder->getSymbolRefAttr(name));
      result.addAttributes(attrs);
      result.addTypes(resType);}]>,

    OpBuilder<"Builder *builder, OperationState &result, GlobalOp global, "
              "ArrayRef<NamedAttribute> attrs = {}", [{
      build(builder, result, PointerType::get(global.getType()), global.sym_name(),
            attrs);}]>
  ];

  let extraClassDeclaration = [{
    PointerType getType() {
      return getResult().getType().cast<PointerType>();
    }
    /// Return the cil.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal();
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.global_address_of @";
    p << global_name();
    p << " : " << getType();
  }];

  let parser = "return parseCILAddressOfOp(parser, result);";
}

def CIL_GlobalOp : CIL_Op<"global", [Symbol]> {
  let summary = "Global Operation";
  let description = [{Global operation}];
  let arguments = (ins StrAttr: $sym_name, OptionalAttr<AnyAttr>:$value, OptionalAttr<SymbolRefAttr>:$constructSym);
  let results = (outs AnyType);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, mlir::Type type, "
              "bool isConstant, StringRef name, "
              "Attribute value">,
    OpBuilder<"Builder *builder, OperationState &result, mlir::Type type, "
              "bool isConstant, StringRef name, "
              "SymbolRefAttr constructSymAttr">
  ];
  let regions = (region AnyRegion:$initializer);

  let extraClassDeclaration = [{
    /// Return the MLIR type of the global.
    mlir::Type getType() {
      return getResult().getType();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return value().getValueOr(Attribute());
    }

    /// Return the initializer region. This may be empty, but if it is not it
    /// terminates in an `std.return` op with the initializer value.
    Region &getInitializerRegion() {
      return getOperation()->getRegion(0);
    }
    /// Return the initializer block. If the initializer region is empty this
    /// is nullptr. If it is not nullptr, it terminates with an `std.return`
    /// op with the initializer value.
    Block *getInitializerBlock() {
      return getInitializerRegion().empty() ?
        nullptr : &getInitializerRegion().front();
    }

  }];

  let verifier = ?;
  let printer = [{
    p << "cil.global ";
    p.printSymbolName(sym_name());
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getType();
    if (getInitializerBlock())
      p.printRegion(getInitializerRegion());
  }];
  let parser = "return parseCILGlobalOp(parser, result);";
}

def CIL_GlobalCtorOp : CIL_Op<"global_ctors", []> {
  let summary = "Global constructor initialzer Operation";
  let description = [{Global constructor initialzer Operation}];
  let arguments = (ins SymbolRefAttr:$constructor);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, SymbolRefAttr constructorRef", [{
      result.addAttribute("constructor", constructorRef);
  }]>];

  let verifier = ?;
  let printer = [{
    p << "cil.global_ctors " << constructor();
  }];
  let parser = ?;
}

def CIL_CILArrayIndexOp : CIL_Op<"array_index", []> {
  let summary = "ArrayEle Operation";
  let description = [{CIL Array Element operation}];
  let arguments = (ins AnyRefLike:$array, AnyCILInteger: $idx);
  let results = (outs AnyRefLike:$res);

  let extraClassDeclaration = [{
    mlir::Value getBase() { return array(); }
    mlir::Value getIndex() { return idx(); }
    }];

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value base, Value idx", [{
      result.addOperands({base, idx});
      auto baseArrayTy = base.getType().cast<CIL::PointerType>().getEleTy();
      auto type = baseArrayTy.cast<CIL::ArrayType>().getEleTy();
      result.addTypes(CIL::PointerType::get(type));
  }]>];

  let verifier = ?;
  let assemblyFormat = "$array `,` $idx attr-dict `:` `(` type($array) `,` type($idx) `)` `->` type($res)";
  let parser = ?;
}

// TODO: Fix name:. This is like llvm::GEP for
// IO related operations.
def CIL_CILEleOp : CIL_Op<"arrayele", []> {
  let summary = "ArrayEle Operation";
  let description = [{CIL Array Element operation}];
  let arguments = (ins AnyRef:$pointer, Variadic<Index>:$indices);
  let results = (outs AnyRef);

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    }];

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def CIL_UndefOp : CIL_Op<"undef", []> {
  let summary = "Undef Operation";
  let description = [{Undef value operation of any type}];
  let results = (outs AnyType);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, PointerType resType", [{
      result.addTypes(resType);}]>
  ];

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def CIL_StoIAOp : CIL_Op<"stoia", []> {
  let summary = "CILStoIAOp Operation";
  let description = [{CIL string to int array op}];
  let arguments = (ins AnyRef:$unit, AnyRef:$expr);

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def CIL_StoIOp : CIL_Op<"stoi", []> {
  let summary = "CILStoIOp Operation";
  let description = [{CIL char to int op}];
  let arguments = (ins AnyRef:$unit);
  let results = (outs I32);

  let verifier = ?;
  let printer = ?;
  let parser = ?;
}

def CIL_ItoSOp : CIL_Op<"itos", []> {
  let summary = "CIL_ItoSOp  Operation";
  let description = [{CIL int to string operation}];
  let arguments = (ins AnyRef:$unit, I32:$expr);

  let verifier = ?;
  let printer = [{
    p << "cil.stoi " << unit() << ", ";
    p << expr();
  }];
  let parser = ?;
}

def CIL_SprintfOp : CIL_Op<"sprintf", []> {
  let summary = "sprintf operation";
  let description = [{CIL sprintf operation}];

  let arguments = (ins AnyRef:$unit, Variadic<AnyType> : $args);

  let verifier = ? ;
  let printer = [{
    p << "cil.sprintf "<< unit() << ", ";
    SmallVector<Value , 2> operands(args());
    if (!operands.empty()) {
      for (unsigned I = 0; I < operands.size() - 1; ++I) {
        p << operands[I] << ", ";
      }
      p << operands.back();
    }
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
}

def CIL_LNotOp : CIL_Op<"lnot", []> {
  let summary = "Logical not operation";
  let description = [{Logical not operation}];
  let arguments = (ins CIL_BoolType:$arg);
  let results = (outs CIL_BoolType);

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$arg attr-dict";
}

// Get the pointer for the given ref type.
def CIL_GetPointerToOp : CIL_Op<"get_pointer_to", []> {
  let summary = "Get Pointer to Operation";
  let description = [{Used for CIL pointer assignments}];
  let arguments = (ins AnyRef:$CILRef);
  let results = (outs AnyPointer);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value ref">];

  let extraClassDeclaration = [{
    mlir::Value getRef() { return CILRef(); }
  }];

  let verifier = ?;
  let printer = [{
    p << "cil.get_pointer_to " << getRef() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

def CIL_NullPointerOp : CIL_Op<"constant_null_pointer"> {
  let summary = "Generates null pointer";
  let description = [{

  }];

  let results = (outs AnyPointer:$res);

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, PointerType resType", [{
      result.addTypes(resType);}]>
  ];
  let verifier = ? ;
  let printer = [{
    p << "cil.constant_null_pointer : " << getResult().getType();
  }];
  let parser = ? ;
  let assemblyFormat  = "attr-dict `:` type($res)";
}

// Node for conditional expression.
def CIL_ConditionalOp : CIL_Op<"conditional", []> {
  let summary = "Compares two pointer";
  let description = [{Used for CIL pointer assignments}];
  let arguments = (ins BoolType:$cond, AnyType:$trueVal, AnyType:$falseVal);
  let results = (outs AnyType);
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, mlir::Value Cond, mlir::Value trueVal, mlir::Value falseVal", [{
      result.addOperands({Cond, trueVal, falseVal});
      assert(trueVal.getType() == falseVal.getType());
      result.addTypes(trueVal.getType());}]>
  ];
  let verifier = ?;
  let skipDefaultBuilders = 1;
  let printer = [{
    p << "cil.conditional " <<  cond() << " " << trueVal() << ", " << falseVal() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}

// Get the pointer for the given ref type.
def CIL_CmpPointerEqualOp : CIL_Op<"cmp_pointer_equal", []> {
  let summary = "Compares two pointer";
  let description = [{Used for CIL pointer assignments}];
  let arguments = (ins AnyPointer:$lhs, AnyPointer:$rhs);
  let results = (outs BoolType:$res);

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

// Operation to represent Subroutine and Fortran function calls.
def CIL_CILCallOp : CIL_Op<"call", [CallOpInterface]> {
  let summary = "CIL call operation";
  let description = [{
    The "call" operation represents a direct call to a function.
  }];

  let arguments = (ins Variadic<AnyType>:$operands, SymbolRefAttr:$callee, I32Attr:$num_operands);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, FuncOp callee, ArrayRef<Value > operands = {}">,
    OpBuilder<
    "Builder *builder, OperationState &result, SymbolRefAttr symbolScope,"
    "ArrayRef<Type> results, ArrayRef<Value > operands = {}">
  ];

  let extraClassDeclaration = [{
    SymbolRefAttr getCallee() { return callee(); }
    FunctionType getCalleeType() {
      auto range = getResultTypes();
      SmallVector<Type, 4> resultTypes;
      resultTypes.append(range.begin(), range.end());
      SmallVector<Type, 8> argTypes(getOperandTypes());
      return FunctionType::get(argTypes, resultTypes, getContext());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let verifier = ? ;
  let printer = [{
    p << "cil.call ";
    p.printAttribute(callee());
    p <<  "(";
    SmallVector<Value , 2> ops(getArgOperands());
    SmallVector<Type, 2> argTypes;
    for (unsigned I = 0; I < ops.size(); ++I) {
      if (I > 0)
       p << ", ";
      p << ops[I];
      argTypes.push_back(ops[I].getType());
    }
    p << ")";
    //p.printOptionalAttrDict(getAttrs());
    auto fty = FunctionType::get(argTypes, getResultTypes(), getContext());
    p  << " : " << fty;
  }];
  let parser = "return parseCILCallOp(parser, result);";
}

def CIL_StructElementOp : CIL_Op<"struct_element", []> {
  let summary = "CIL struct element operation";
  let description = [{
      Operation to represent struct field acceses
  }];
  let arguments = (ins AnyPointer:$ptr, AnyCILInteger:$index);
  let results  = (outs AnyPointer:$res);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Type type, Value ptr, Value index", [{
      result.addOperands({ptr, index});
      result.addTypes(type);
    }]
  >];


  let verifier = ?;
  let printer = [{
    p << "struct_element " << ptr() << ", " << index();
    p << " : (" << ptr().getType() << ", " << index().getType() << ")";
    p << " -> " << getResult().getType();

  }];

  let parser = ?;
  let assemblyFormat = "$ptr `,` $index attr-dict `:` `(` type($ptr) `,` type($index) `)` `->` type($res)";
}

// Copied from std.cond_branch implementation.
class CIL_CILTerminatorLikeOp<string mnemonic> : CIL_Op<mnemonic, [Terminator]> {
  let summary = "CIL conditional branch operation";
  let description = [{
  }];

  let successors = (successor AnySuccessor:$successbr,
                              AnySuccessor:$failurebr);

  let arguments = (ins CIL_BoolType:$cond);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value condition,"
    "Block *trueDest, ValueRange trueOperands,"
    "Block *falseDest, ValueRange falseOperands", [{
      result.addOperands(condition);
      result.addSuccessor(trueDest, trueOperands);
      result.addSuccessor(falseDest, falseOperands);
  }]>];

  // CondBranchOp is fully verified by traits.
  let verifier = ?;
let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // The condition operand is the first operand in the list.
    Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getSuccessor(falseIndex);
    }

    // Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_iterator true_operand_begin() {
      return operand_begin() + getTrueDestOperandIndex();
    }
    operand_iterator true_operand_end() {
      return true_operand_begin() + getNumTrueOperands();
   }
    operand_range getTrueOperands() {
      return {true_operand_begin(), true_operand_end()};
    }

    unsigned getNumTrueOperands()  {
      return getNumSuccessorOperands(trueIndex);
    }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getOperation()->eraseSuccessorOperand(trueIndex, index);
    }

    // Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_iterator false_operand_begin() { return true_operand_end(); }
    operand_iterator false_operand_end() {
      return false_operand_begin() + getNumFalseOperands();
    }
    operand_range getFalseOperands() {
      return {false_operand_begin(), false_operand_end()};
    }

    unsigned getNumFalseOperands() {
      return getNumSuccessorOperands(falseIndex);
    }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getOperation()->eraseSuccessorOperand(falseIndex, index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let printer = [{
    p << getOperationName() << ' ' << getOperand(0) << ", ";
    p.printSuccessorAndUseList(getOperation(), 0);
    p << ", ";
    p.printSuccessorAndUseList(getOperation(), 1);
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ? ;
  let assemblyFormat = "$cond `,` successors attr-dict";
}

def CIL_CILIfOp : CIL_CILTerminatorLikeOp<"if">;
def CIL_CILWhileOp : CIL_CILTerminatorLikeOp<"while">;
def CIL_CILDoWhileOp : CIL_CILTerminatorLikeOp<"do_while">;
def CIL_CILForOp : CIL_CILTerminatorLikeOp<"for">;

def CIL_UnreachableOp : CIL_Op<"unreachable", [Terminator]> {
  let summary = "unreachable operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<"Builder *b, OperationState &result", [{ }]>,
    OpBuilder<"Builder *b, OperationState &result, mlir::Value valueToReturn", [{
      result.addOperands(valueToReturn);
    }]>
    ];

  let verifier = ?;
  let printer = [{
    p << getOperationName() << " ";
    if (!operands().empty())
      p << *operand_begin();
  }];

  let parser = "return parseCILUnreachableOp(parser, result);";
}

def CIL_ForLoopTerminatorOp :
    CIL_Op<"terminator", [Terminator]> {
  let summary = "for loop terminator operation";
  let description = [{
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

def CIL_CILReturnOp : CIL_Op<"return", [Terminator]> {
  let summary = "return operation for any region. Example, global initializer region";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  // No custom parsing/printing form.
  let parser = "return parseCILReturnOp(parser, result);";
  let printer = [{
    p << "cil.return ";
    SmallVector<Value , 2> ops(operands());
    for (unsigned I = 0; I < ops.size(); ++I) {
      if (I > 0)
       p << ", ";
      p << ops[I];
    }
    p.printOptionalAttrDict(getAttrs());
    if (ops.size() > 0)
      p << " : " << getOperand(0).getType();
  }];

  // Fully specified by traits.
  let verifier = ?;
}


def CIL_ForLoopOp : CIL_Op<"for_loop",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"CIL::ForLoopTerminatorOp">]> {
  let summary = "perfect loop operation";
  let description = [{
  }];
  let arguments = (ins Variadic<AnyType>:$args);  // Should be isIntOrIndex()
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value lowerBound, Value upperBound, Value step, Type inductionVartype", [{
    result.addOperands({lowerBound, upperBound, step});
    Region *bodyRegion = result.addRegion();
    ForLoopOp::ensureTerminator(*bodyRegion, *builder, result.location);
    bodyRegion->front().addArgument(inductionVartype);
   }]>
  ];

  let extraClassDeclaration = [{
    Block *getBody() { return &region().front(); }
    Value getIndVar() { return getBody()->getArgument(0); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) {
      getOperation()->setOperand(2, step); }
    mlir::Value getLowerBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(0); }
    mlir::Value getUpperBound() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(1); }
    mlir::Value getStep() {
      if (args().empty()) return nullptr;
      return getOperation()->getOperand(2); }
  }];

  let verifier = ?;
  let printer = [{
    SmallVector<Value , 2> operands(args());
    p << "cil.for_loop ";
    if (!operands.empty()) {
      p << getIndVar() << " = " << operands[0] << ", ";
      p << operands[1] << ", " << operands[2];
    }
    p.printRegion(region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
    p.printOptionalAttrDict(getAttrs());
  }];
  let parser = ?;
}

def CIL_InsertValueOp : CIL_Op<"insertvalue"> {

  let summary = "Insert value Op";
  let description = [{
    Similar to LLVM::InsertValue
  }];

  let arguments = (ins AnyType:$container, AnyType:$value, ArrayAttr:$position);
  let results = (outs AnyType);

  let verifier = ? ;

  let printer = [{
    p << "cil.insertvalue " << value() << ", " << container()
    	<< position();

  	p.printOptionalAttrDict(getAttrs(), {"position"});
  	p << " : " << container().getType();
  }];

  let parser = ? ;
}

def CIL_ExtractValueOp : CIL_Op<"extractvalue"> {

  let summary = "Extract value Op";
  let description = [{
    Similar to LLVM::ExtractValue
  }];

  let arguments = (ins AnyType:$container, ArrayAttr:$position);
  let results = (outs AnyType);

  let parser = ? ;
  let verifier = ? ;

  let printer = [{
    p << "cil.extractvalue " << container() << ", " << position();
  	p.printOptionalAttrDict(getAttrs(), {"position"});
  	p << " : " << container().getType();
  }];
}

def CIL_CILGEPOp : CIL_Op<"gep", []> {
  let summary = "GEP Operation";
  let description = [{CIL GEP operation}];
  let arguments = (ins AnyType:$pointer, Variadic<CIL_IntegerType>:$indices);
  let results = (outs AnyType);

  let extraClassDeclaration = [{
    mlir::Value getPointer() { return pointer(); }
    }];

  let verifier = ?;
  let printer = [{
    llvm::SmallVector<mlir::Value, 2> subs(indices());
    p << "cil.gep " << pointer() << "[";
    for (unsigned i = 0; i < subs.size(); ++i) {
      p << subs[i];
      if (i != subs.size() - 1)
        p << ", ";
    }
    p << "] : ";
    p << "(" << pointer().getType() << ", " << subs[0].getType() << ") -> ";
    p << getResult().getType();
  }];

  let parser = "return parseCILGEPOp(parser, result);";
}

def CIL_CILIndexCastOp : CIL_Op<"index_cast", []> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between integer scalars and 'index' scalars.  Index is an integer of
    platform-specific bit width.  If casting to a wider integer, the value is
    sign-extended.  If casting to a narrower integer, the value is truncated.
  }];

  let arguments = (ins AnyType:$inval);
  let results = (outs AnyType:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
  }];

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$inval `:` type($inval) attr-dict `->` type($res)";
}

def CIL_CILBoolCastOp : CIL_Op<"bool_cast", []> {
  let summary = "cast from cil bool to std::bool";
  let description = [{
  }];

  let arguments = (ins CIL_BoolType:$inval);
  let results = (outs I1:$res);

  let builders = ?;

  let extraClassDeclaration = [{
    mlir::Value getValue() { return inval(); }
  }];

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$inval `:` type($inval) attr-dict `->` type($res)";
}

def CIL_CILSelectOp : CIL_Op<"select",
                       [TypesMatchWith<"True type and false type should match",
                       "trueVal", "falseVal", "$_self">]> {
  let summary = "select op";
  let description = [{
  }];

  let arguments = (ins CIL_BoolType:$cond, AnyType:$trueVal, AnyType:$falseVal);
  let results = (outs AnyType:$res);

  let builders = [OpBuilder<
      "Builder *builder, OperationState &result, Value condition,"
      "Value trueValue, Value falseValue", [{
        result.addOperands({condition, trueValue, falseValue});
        result.addTypes(trueValue.getType());
      }]>];

  let verifier = ?;
  let parser = ?;
  let assemblyFormat = "$cond `,` $trueVal `,` $falseVal attr-dict `:` type($trueVal) `->` type($res)";
}

def CIL_CILCallIndirectOp : CIL_Op<"call_indirect", [
      CallOpInterface,
      TypesMatchWith<"callee input types match argument types",
                     "callee", "operands",
        "$_self.cast<CIL::PointerType>().getEleTy().cast<FunctionType>().getInputs()">,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
       "$_self.cast<CIL::PointerType>().getEleTy().cast<FunctionType>().getResults()">
    ]> {
  let summary = "indirect call operation";
  let description = [{
  }];

  let arguments = (ins AnyPointer:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value callee,"
    "ValueRange operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      auto ptrTy = callee.getType().dyn_cast_or_null<CIL::PointerType>();
      assert(ptrTy);
      result.addTypes(ptrTy.getEleTy().cast<FunctionType>().getResults());
  }]>];

  let extraClassDeclaration = [{
    Value getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  let verifier = ?;

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` type($callee)";
}

def CIL_SizeOfOp : CIL_Op<"sizeof"> {
  let summary = "CIL sizeof operation";
  let description = [{

  }];

  let arguments = (ins AnyPointer:$ptr);
  let results = (outs AnyCILInteger:$res);

  let verifier = ? ;
  let assemblyFormat  = "$ptr attr-dict `:` type($ptr) `->` type($res)";
}


#endif // CIL_OPS
