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
// Pass to convert the class inheritance to the class composition!
//===----------------------------------------------------------------------===//

#include "clang/cil/dialect/CIL/CILBuilder.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

#define PASS_NAME "ClassInheritanceLowering"
#define DEBUG_TYPE PASS_NAME
// TODO: Add DEBUG WITH TYPE and STATISTIC

using namespace mlir;
using namespace CIL;

struct ClassInheritanceLoweringPass
    : public ModulePass<ClassInheritanceLoweringPass> {
public:
  virtual void runOnModule();
};

static constexpr const char *BASE_CLASS_FIELD_NAME_PREFIX = "inherited.";

// Convert all the inherited classes as current class member.
static bool handleClassInheritance(CIL::ClassOp classOp) {

  auto baseClassesOpt = classOp.baseClasses();
  if (!baseClassesOpt.hasValue()) {
    return true;
  }
  auto baseTypeAttrs = baseClassesOpt.getValue().getValue();
  if (baseTypeAttrs.empty()) {
    return true;
  }
  CILBuilder builder(classOp);
  builder.setInsertionPointToStart(&classOp.body().front());
  // For each base class type, add it as a member field.
  for (auto attr : baseTypeAttrs) {
    auto type = attr.cast<TypeAttr>().getValue();
    auto classType = type.cast<ClassType>();
    builder.create<CIL::FieldDeclOp>(classOp.getLoc(), classType,
                                     BASE_CLASS_FIELD_NAME_PREFIX +
                                         classType.getName().str());
  }
  return true;
}

// Convert the op as regular field access.
static bool handleDerivedToBaseCast(DerivedToBaseCastOp op) {
  CILBuilder builder(op);
  auto baseClassType = op.getBaseClassType();
  auto fieldName = BASE_CLASS_FIELD_NAME_PREFIX + baseClassType.getName().str();

  auto symRefAttr = builder.getSymbolRefAttr(
      fieldName, builder.getSymbolRefAttr(baseClassType.getName()));
  auto accessOp = builder.create<CIL::FieldAccessOp>(op.getLoc(), op.getType(),
                                                     op.ref(), symRefAttr);
  op.replaceAllUsesWith(accessOp.getResult());
  op.erase();
  return true;
}

void ClassInheritanceLoweringPass::runOnModule() {
  auto module = getModule();

  module.walk([&](mlir::Operation *op) {
    if (auto classOp = dyn_cast<CIL::ClassOp>(op)) {
      handleClassInheritance(classOp);
      return;
    }
    if (auto derivedCastOp = dyn_cast<CIL::DerivedToBaseCastOp>(op)) {
      handleDerivedToBaseCast(derivedCastOp);
      return;
    }
  });
}

namespace CIL {
std::unique_ptr<mlir::Pass> createClassInheritanceLoweringPass() {
  return std::make_unique<ClassInheritanceLoweringPass>();
}
} // namespace CIL