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
//===----------------------------------------------------------------------===//

#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#define PASS_NAME "ClassLowering"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace mlir;
using namespace CIL;

struct ClassLoweringPass : public ModulePass<ClassLoweringPass> {
private:
  struct ClassOpInfo {
    CIL::StructType type;
    llvm::StringMap<unsigned int> fieldIndex;
    llvm::StringMap<mlir::FuncOp> methodInfo;
  };

  llvm::StringMap<ClassOpInfo> classInfoMap;

public:
  virtual void runOnModule();

  bool lowerClassOperations(mlir::ModuleOp module);

  bool lowerClassOp(CIL::ClassOp op);

  bool handleClassAlloca(CIL::AllocaOp op);

  bool handleFieldAccess(CIL::FieldAccessOp op);

  bool lowerMemberFunction(ClassOp currClass, ClassOpInfo &classInfo,
                           mlir::FuncOp op);

  bool replaceClassTypeInOp(mlir::Operation *op);

  mlir::Type replaceClassType(mlir::Type currType);

  bool handleMemberCall(CIL::MemberCallOp op);
};

mlir::Type ClassLoweringPass::replaceClassType(mlir::Type currType) {
  CIL::CILBuilder builder(currType.getContext());
  switch (currType.getKind()) {
  case CIL::CIL_Class: {
    // TODO: Anything better than cast. static_cast?
    auto classType = currType.cast<CIL::ClassType>();
    if (classInfoMap.find(classType.getName()) == classInfoMap.end()) {
      currType.dump();
      llvm_unreachable("Unknown class type");
    }
    return classInfoMap[classType.getName()].type;
  } break;
  case CIL::CIL_Pointer: {
    auto ptrType = currType.cast<CIL::PointerType>();
    auto eleTy = ptrType.getEleTy();
    eleTy = replaceClassType(ptrType.getEleTy());
    return builder.getCILPointerType(eleTy, ptrType.isReference());
  } break;
  case CIL::CIL_Array: {
    auto arrTy = currType.cast<CIL::ArrayType>();
    auto eleTy = replaceClassType(arrTy.getEleTy());
    return CIL::ArrayType::get(arrTy.getShape(), eleTy);
  } break;
  case CIL::CIL_Struct: {

    // Can be ignored as there can be no struct types
    // from C++ code.
    break;
    auto strType = currType.cast<CIL::StructType>();
    auto eleTypes = strType.getElementTypes();
    SmallVector<mlir::Type, 2> newTypes;
    for (auto eleTy : eleTypes) {
      newTypes.push_back(replaceClassType(eleTy));
    }
    return CIL::StructType::get(strType.getContext(), newTypes,
                                strType.getName());
  } break;
  case mlir::Type::Function: {
    auto functype = currType.cast<mlir::FunctionType>();
    auto inputTypes = functype.getInputs();
    auto outputTypes = functype.getResults();
    SmallVector<mlir::Type, 2> inputs, outputs;
    for (auto eleTy : inputTypes) {
      inputs.push_back(replaceClassType(eleTy));
    }
    for (auto eleTy : outputTypes) {
      outputs.push_back(replaceClassType(eleTy));
    }
    return mlir::FunctionType::get(inputs, outputs, functype.getContext());
  } break;
  case CIL::CIL_Integer:
  case CIL::CIL_FloatingTypeKind:
  case CIL::CIL_Char:
  case CIL::CIL_UnsignedChar:
  case CIL::CIL_VarArg:
  case CIL::CIL_Void:
    break;
  default:
    llvm_unreachable("unhandled type for conversion");
  };
  return currType;
}

bool ClassLoweringPass::replaceClassTypeInOp(mlir::Operation *op) {
  // Set all the types for all the operands.
  for (auto II = 0; II < op->getNumOperands(); ++II) {
    auto curr = op->getOperand(II);
    curr.setType(replaceClassType(curr.getType()));
    op->setOperand(II, curr);
  }
  for (auto II = 0; II < op->getNumResults(); ++II) {
    auto curr = op->getResult(II);
    curr.setType(replaceClassType(curr.getType()));
  }
  return true;
}

bool ClassLoweringPass::handleClassAlloca(CIL::AllocaOp op) {
  CIL::CILBuilder builder(op);
  auto type = builder.getUnderlyingType(op.getType());
  if (!type.isa<CIL::ClassType>()) {
    return true;
  }
  type = replaceClassType(op.getType().getEleTy());
  auto newAlloca =
      builder.create<CIL::AllocaOp>(op.getLoc(), op.getName(), type);
  op.replaceAllUsesWith(newAlloca.getResult());
  op.erase();
  return true;
}

static StringRef getClassName(mlir::Type type) {
  if (auto classTy = type.dyn_cast<ClassType>()) {
    return classTy.getName();
  }
  return type.cast<CIL::StructType>().getName();
}

bool ClassLoweringPass::handleMemberCall(MemberCallOp op) {
  auto base = op.base();
  auto className = getClassName(base.getType().cast<PointerType>().getEleTy());
  auto classInfoIter = classInfoMap.find(className);
  assert(classInfoIter != classInfoMap.end());
  auto &classInfo = classInfoIter->getValue();

  auto funcName = op.callee().getRootReference();
  auto nameIter = classInfo.methodInfo.find(funcName);
  assert(nameIter != classInfo.methodInfo.end());
  auto funcOp = nameIter->getValue();
  CIL::CILBuilder builder(op);

  SmallVector<Value, 2> operands;
  operands.append(op.arg_operand_begin(), op.arg_operand_end());
  auto call = builder.create<CIL::CILCallOp>(op.getLoc(), funcOp, operands);
  op.replaceAllUsesWith(call);
  op.erase();
  replaceClassTypeInOp(call);
  return true;
}

bool ClassLoweringPass::handleFieldAccess(CIL::FieldAccessOp op) {
  auto fieldAttr = op.field_name();
  auto className =
      getClassName(op.base().getType().cast<PointerType>().getEleTy());
  auto classInfoIter = classInfoMap.find(className);
  assert(classInfoIter != classInfoMap.end());
  auto &classInfo = classInfoIter->getValue();

  auto nameIter = classInfo.fieldIndex.find(fieldAttr.getRootReference());
  assert(nameIter != classInfo.fieldIndex.end());
  auto indexVal = nameIter->getValue();

  CIL::CILBuilder builder(op);
  auto intKind =
      CIL::IntegerTy::get(CIL::CILIntegerKind::IntKind, op.getContext());
  auto index = builder.create<CIL::CILConstantOp>(
      op.getLoc(), intKind, builder.getI32IntegerAttr(indexVal));
  auto newAlloca = builder.create<CIL::StructElementOp>(
      op.getLoc(), replaceClassType(op.getType()), op.base(), index);
  op.replaceAllUsesWith(newAlloca.getResult());
  op.erase();
  return true;
}

bool ClassLoweringPass::lowerMemberFunction(
    ClassOp currClass, ClassLoweringPass::ClassOpInfo &classInfo,
    mlir::FuncOp op) {
  CIL::CILBuilder builder(op);
  op.getOperation()->moveBefore(currClass);
  auto structPointerType = builder.getCILPointerType(classInfo.type);

  if (!op.isExternal()) {
    auto &entryBlock = op.getBlocks().front();
    entryBlock.insertArgument((unsigned int)0, structPointerType);
  }

  // Change the funciton type.
  auto currType = op.getType();
  SmallVector<mlir::Type, 2> inputs;
  inputs.push_back(structPointerType);
  inputs.append(currType.getInputs().begin(), currType.getInputs().end());
  auto newFuncType = builder.getFunctionType(inputs, currType.getResults());
  op.setType(newFuncType);

  classInfo.methodInfo[op.getName()] = op;
  return true;
}

bool ClassLoweringPass::lowerClassOp(ClassOp op) {
  // Make sure the class is already processed.
  assert(classInfoMap.find(op.getName()) != classInfoMap.end());

  // Collect all the members fields in the class.
  SmallVector<CIL::FieldDeclOp, 2> fields;
  SmallVector<mlir::FuncOp, 2> functions;

  for (auto &member : op.body().front()) {
    if (auto field = dyn_cast<CIL::FieldDeclOp>(member)) {
      fields.push_back(field);
      continue;
    }
    if (auto func = dyn_cast<mlir::FuncOp>(member)) {
      functions.push_back(func);
      continue;
    }
    if (isa<CIL::ClassReturnOp>(member))
      continue;
    llvm_unreachable("Unknown operation in class op");
  }

  auto &classInfo = classInfoMap[op.getName()];
  // Handle member fields:
  // Create a struct type for the fields.
  SmallVector<mlir::Type, 2> fieldTypes;
  unsigned index = 0;
  for (auto field : fields) {
    classInfo.type.addMemberType(replaceClassType(field.type()));
    classInfo.fieldIndex[field.getName()] = index++;
  }

  // FIXME: Clang adds empty i8 type if there are no class member variables.
  // Doing the same here to be consistent.
  if (fields.empty()) {
    classInfo.type.addMemberType(CIL::CharTy::get(op.getContext()));
  }

  // Handle member functions:
  // move all the member functions to out of class operation
  // TODO: Name manling.
  for (auto method : functions) {
    lowerMemberFunction(op, classInfo, method);
  }
  classInfoMap[op.getName()] = classInfo;
  op.erase();
  return true;
}

bool ClassLoweringPass::lowerClassOperations(mlir::ModuleOp module) {
  // Collect all the class operations in the module.
  SmallVector<CIL::ClassOp, 2> classList;
  module.walk([&](CIL::ClassOp op) { classList.push_back(op); });

  // First lower all the class operations to structure + module level
  // functions.

  // Create dummy struct types.
  for (auto op : classList) {
    ClassOpInfo classInfo;
    classInfo.type = CIL::StructType::get(op.getContext(), {}, op.getName());
    classInfoMap[op.getName()] = classInfo;
  }

  for (auto classOp : classList) {
    lowerClassOp(classOp);
  }

  // Now convert all the class based operations to the lowered form.
  module.walk([&](mlir::Operation *op) {
    if (auto allocaOp = dyn_cast<AllocaOp>(op)) {
      handleClassAlloca(allocaOp);
      return;
    }
    if (auto funcOp = dyn_cast<mlir::FuncOp>(op)) {
      auto type = funcOp.getType();
      funcOp.setType(replaceClassType(type).cast<FunctionType>());
      if (funcOp.isExternal()) {
        return;
      }
      for (auto &arg : funcOp.getArguments()) {
        arg.setType(replaceClassType(arg.getType()));
      }
      return;
    }
    if (auto fieldAccess = dyn_cast<FieldAccessOp>(op)) {
      handleFieldAccess(fieldAccess);
      return;
    }
    if (auto thisOp = dyn_cast<ThisOp>(op)) {
      auto currFunc = thisOp.getParentOfType<mlir::FuncOp>();
      assert(currFunc);
      // FIXME: Check if this is the member function.
      thisOp.replaceAllUsesWith(currFunc.getArgument(0));
      thisOp.erase();
      return;
    }
    if (auto memberCall = dyn_cast<MemberCallOp>(op)) {
      handleMemberCall(memberCall);
      return;
    }

    replaceClassTypeInOp(op);
  });
  return true;
}

void ClassLoweringPass::runOnModule() {
  classInfoMap.clear();
  auto context = &getContext();
  auto module = getModule();
  CIL::CILBuilder builder(context);

  if (!lowerClassOperations(module)) {
    signalPassFailure();
  }
}

namespace CIL {
std::unique_ptr<mlir::Pass> createClassLoweringPass() {
  return std::make_unique<ClassLoweringPass>();
}
} // namespace CIL