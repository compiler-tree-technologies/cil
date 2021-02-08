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

def CIL_ClassOp : CIL_Op<"class", [NativeOpTrait<"IsIsolatedFromAbove">,
                         Symbol, SymbolTable,
                         SingleBlockImplicitTerminator<"CIL::ClassReturnOp">]> {
  let summary = "OPeration for representing class type";

  let regions = (region SizedRegion<1>:$body);

  let arguments = (SymbolRefAttr:$name OptionalAttr<ArrayAttr>:$baseClasses);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringRef name", [{
    Region *bodyRegion = result.addRegion();
    result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
    ClassOp::ensureTerminator(*bodyRegion, *builder, result.location);
   }]>,
   OpBuilder<"Builder *builder, OperationState &result, StringRef name, ArrayAttr baseClasses", [{
    Region *bodyRegion = result.addRegion();
    result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder->getStringAttr(name));
  if (!baseClasses.getValue().empty())
    result.addAttribute("baseClasses", baseClasses);
    ClassOp::ensureTerminator(*bodyRegion, *builder, result.location);
   }]>
  ];

  let extraClassDeclaration = [{
    static ClassOp create(Location location, StringRef name) {
       OperationState state(location, "cil.class");
       Builder builder(location->getContext());
       CIL::ClassOp::build(&builder, state, name);
       return llvm::cast<ClassOp>(Operation::create(state));
    }
    void push_back(Operation* op) {
      auto terminator = body().front().getTerminator();
      body().front().push_back(op);
      op->moveBefore(terminator);
    }
    StringRef getName() {
     return getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
         .getValue();
    }

  }];

  let verifier = ?;
  let printer = [{
    p << "cil.class " << getName();
     if (baseClasses().hasValue()) {
      p  << " inherits ";
      p.printAttribute(baseClassesAttr());
     }
    p.printRegion(body(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  }];
  let parser = ?;
}

// Teriminator for the class operation region.
// FIXME: std.return only works for regions nested in std.func
def CIL_ClassReturnOp: CIL_Op<"class_return", [Terminator]> {
  let summary = "class region return operation";
  let description = [{
  }];


  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *b, OperationState &result", [{ }]>];

  let verifier = ?;
  let printer = [{
    p << getOperationName();
  }];

  let parser = ?;
}

// Class field declaration op.
def FC_FieldDeclOp : CIL_Op<"field_decl", [Symbol]> {
  let summary = "return operation";
  let description = [{
  }];

  let arguments = (ins TypeAttr:$type);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Type type, StringRef name", [{
    result.addAttribute("type",
                        mlir::TypeAttr::get(type));
    result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      b->getStringAttr(name));
    }]>
    ];

  let extraClassDeclaration = [{
    StringRef getName() {
     return getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
         .getValue();
    }
  }];

  let verifier = ?;
  let printer = [{
    p << getOperationName() <<  " " << getName() << " :   " << type();
  }];

  let parser = ?;
}

// Used for calss field access operation.
def FC_FieldAccessOp : CIL_Op<"field_access", []> {
  let summary = "class field access operation";
  let description = [{
  }];

  let arguments = (ins AnyClassPointer:$base, SymbolRefAttr:$field_name);
  let results  = (outs AnyPointer);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Type type, Value classPtr, SymbolRefAttr fieldName", [{
    result.addOperands(classPtr);
    result.addAttribute("field_name", fieldName);
    result.addTypes(type);
    }]>
  ];

  let extraClassDeclaration = [{
    StringRef getFieldName() {
     return getAttrOfType<StringAttr>("field_name")
         .getValue();
    }
  }];

  let verifier = ?;
  let printer = [{
    p << getOperationName() <<  " " << base() << ", " << field_name();
  }];

  let parser = ?;
}

// Used for calss field access operation.
def FC_ThisOp : CIL_Op<"cxx_this", []> {
  let summary = "this-> operation";
  let description = [{
  }];

  let results  = (outs AnyClassPointer);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Type type", [{
    result.addTypes(type);
    }]>
  ];

  let extraClassDeclaration = [{
  }];

  let verifier = ?;
  let printer = [{
    p << getOperationName() <<  " : " << getResult().getType();
  }];

  let parser = ?;
}

// Operation to represent Subroutine and Fortran function calls.
def CIL_MemberCallOp : CIL_Op<"member_call", [CallOpInterface]> {
  let summary = "CIL member function call operation";
  let description = [{
    CIL member function call operation.
  }];

  let arguments = (ins AnyClassPointer:$base, SymbolRefAttr:$callee, Variadic<AnyType>:$operands, I32Attr:$num_operands);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<
    "Builder *builder, OperationState &result, Value classPtr, SymbolRefAttr symbolScope,"
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
    p << "cil.member_call " << base() << ", ";
    p.printAttribute(callee());
    p <<  "(";
    SmallVector<Value , 2> ops(getArgOperands());
    for (unsigned I = 1; I < ops.size(); ++I) {
      if (I > 1)
       p << ", ";
      p << ops[I];
    }
    p << ")";
    //p.printOptionalAttrDict(getAttrs());
    for (auto type : getResultTypes())
      p << " : " << type  << ", ";
  }];
  let parser = ? ;
}

// Used for CIL::ast::CastExpr
def CIL_DerivedToBaseCastOp : CIL_Op<"derived_to_base", []> {
  let summary = "Derived to base class conversion operation";
  let description = [{CIL cast expression}];
  let arguments = (ins AnyClassPointer:$ref);
  let results = (outs AnyClassPointer);


  let extraClassDeclaration = [{
    ClassType getBaseClassType() {
      auto retType = getResult().getType();
      return retType.cast<PointerType>().getEleTy().cast<ClassType>();
    }
  }];
  let verifier = ?;
  let printer = [{
    p << "cil.derived_to_base " << ref() << " : "
      << getResult().getType();
  }];

  let parser = ?;
}
