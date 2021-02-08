#include "clang/cil/codegen/CodeGen.h"
#include "clang/cil/dialect/CIL/CILOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"

using namespace clang;
using namespace clang::mlir_codegen;

// Expression related classes:
mlir::Value MLIRCodeGen::emitIntegerLiteral(IntegerLiteral *expr) {
  auto value = expr->getValue().getSExtValue();
  auto type = cgTypes.convertClangType(expr->getType());
  return builder.getCILIntegralConstant(getLoc(expr->getSourceRange()), type,
                                        value);
}

mlir::Value MLIRCodeGen::emitFloatingLiteral(FloatingLiteral *expr) {
  auto value = expr->getValue().convertToDouble();
  auto type = cgTypes.convertClangType(expr->getType());
  return builder.getCILFloatingConstant(getLoc(expr->getSourceRange()), type,
                                        value);
}

// Expression related classes:
mlir::Value MLIRCodeGen::emitStringLiteral(StringLiteral *expr) {
  static unsigned long long strTemp1 = 0;
  std::string name = "__str_tmp" + std::to_string(strTemp1++);
  auto val = expr->getString().str() + '\0';
  auto stringAttr = builder.getStringAttr(val);
  auto type = cgTypes.convertClangType(expr->getType());
  auto savePoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(TheModule->getBody());
  auto globalOp = builder.create<CIL::GlobalOp>(getLoc(expr->getSourceRange()),
                                                type, true, name, stringAttr);

  // builder.insert(globalOp);
  builder.restoreInsertionPoint(savePoint);

  return builder.create<CIL::GlobalAddressOfOp>(getLoc(expr->getSourceRange()),
                                                globalOp);
}

mlir::Value MLIRCodeGen::emitImplicitCastExpr(ImplicitCastExpr *implicitCast) {
  auto expr = implicitCast->getSubExpr();
  switch (implicitCast->getCastKind()) {
  case CK_LValueToRValue: {
    auto subValue = emitExpression(expr);
    return builder.create<CIL::CILLoadOp>(subValue.getLoc(), subValue);
  }
  case CK_IntegralToBoolean: {
    return emitCastExpr(expr, implicitCast->getType());
  }
  case CK_IntegralToFloating: {
    auto subValue = emitExpression(expr);
    auto toTy = cgTypes.convertClangType(implicitCast->getType());
    return builder.create<CIL::CILSIToFPOp>(subValue.getLoc(), toTy, subValue);
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    auto subValue = emitExpression(expr);
    auto ptrTy = subValue.getType().cast<CIL::PointerType>();
    auto arrTy = ptrTy.getEleTy().cast<CIL::ArrayType>();
    auto toType = builder.getCILPointerType(arrTy.getEleTy());
    auto Bitcast = builder.create<CIL::PointerBitCastOp>(subValue.getLoc(),
                                                         toType, subValue);
    return Bitcast;
  }
  case clang::CastKind::CK_NoOp: {
    return emitExpression(expr);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto dstType = implicitCast->getType();
    auto val = implicitCast->getSubExpr();
    return emitCastExpr(val, dstType);
  }
  case clang::CastKind::CK_FloatingCast: {
    auto dstType = implicitCast->getType();
    auto val = implicitCast->getSubExpr();
    return emitFloatCastExpr(val, dstType);
  }
  case clang::CastKind::CK_BitCast: {
    auto rhs = emitExpression(expr);
    auto toTy = cgTypes.convertClangType(implicitCast->getType());
    auto Bitcast =
        builder.create<CIL::PointerBitCastOp>(rhs.getLoc(), toTy, rhs);
    return Bitcast;
  }
  case clang::CastKind::CK_NullToPointer: {
    auto toTy = cgTypes.convertClangType(implicitCast->getType());
    auto loc = getLoc(expr->getSourceRange());
    return builder.create<CIL::NullPointerOp>(loc, toTy);
  }
  case clang::CastKind::CK_DerivedToBase:
  case clang::CastKind::CK_UncheckedDerivedToBase: {
    auto rhs = emitExpression(expr);
    for (auto path : implicitCast->path()) {
      auto toTy = cgTypes.convertClangType(path->getType());
      auto loc = getLoc(expr->getSourceRange());
      rhs = builder
                .create<CIL::DerivedToBaseCastOp>(
                    loc, builder.getCILPointerType(toTy), rhs)
                .getResult();
    }
    return rhs;
  }
  case clang::CastKind::CK_PointerToBoolean: {
    auto rhs = emitExpression(expr);
    auto nullPointer = builder.create<CIL::NullPointerOp>(
        rhs.getLoc(), rhs.getType().cast<CIL::PointerType>());
    auto pointerEq = builder.create<CIL::CmpPointerEqualOp>(
        rhs.getLoc(), builder.getCILBoolType(), rhs, nullPointer);
    return builder.create<CIL::LNotOp>(rhs.getLoc(), builder.getCILBoolType(),
                                       pointerEq);
  }
  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto loc = getLoc(expr->getSourceRange());
    auto toTy = cgTypes.convertClangType(implicitCast->getType());
    auto rhs = emitExpression(expr);
    return builder.create<CIL::PointerBitCastOp>(loc, toTy, rhs);
  }
  default:
    implicitCast->dump();
    llvm_unreachable("unhandled implictcastexpr");
  };
}

mlir::Value MLIRCodeGen::emitFloatCastExpr(Expr *expr, clang::QualType type) {
  auto srcType = expr->getType();
  assert(srcType->isFloatingType() && type->isFloatingType());

  auto value = emitExpression(expr);
  auto toType = cgTypes.convertClangType(type).dyn_cast<CIL::FloatingTy>();
  auto fromType = value.getType().dyn_cast<CIL::FloatingTy>();
  assert(toType && fromType);
  if (toType == fromType)
    return value;

  auto loc = getLoc(expr->getSourceRange());
  if (fromType.getFloatingKind() < toType.getFloatingKind())
    return builder.create<CIL::CILFPExtOp>(loc, toType, value);
  return builder.create<CIL::CILFPTruncOp>(loc, toType, value);
}

mlir::Value MLIRCodeGen::emitCast(mlir::Value value, mlir::Type toTy) {

  auto fromTy = value.getType();
  if (fromTy == toTy)
    return value;

  auto loc = value.getLoc();
  auto fromIntTy = fromTy.dyn_cast_or_null<CIL::IntegerTy>();
  auto toIntTy = toTy.dyn_cast_or_null<CIL::IntegerTy>();

  if (fromIntTy && toIntTy) {

    if (fromIntTy.getIntegerKind() == toIntTy.getIntegerKind()) {
      // llvm_unreachable("emitCastExpr: integers of same type?");
      // FIXME: this might need some handling!!
      return builder.create<CIL::IntCastOp>(loc, toTy, value);
    }
    if (fromIntTy.getIntegerKind() < toIntTy.getIntegerKind()) {
      if (toIntTy.getQualifiers().isUnsigned()) {
        return builder.create<CIL::ZeroExtendOp>(loc, toTy, value);
      } else {
        return builder.create<CIL::SignExtendOp>(loc, toTy, value);
      }
    } else {
      if (toIntTy.getIntegerKind() == CIL::BoolKind) {
        // We can not use trunc for int to bool
        // For example, from llvm langref
        //    %Z = trunc i32 122 to i1 ; yields i1:false
        auto zero = builder.getCILIntegralConstant(loc, fromIntTy, 0);
        return builder.create<CIL::CILCmpIOp>(loc, CmpIPredicate::ne, value,
                                              zero);
      }
      return builder.create<CIL::TruncateOp>(loc, toTy, value);
    }
  }
  auto toCharTy = toTy.dyn_cast_or_null<CIL::CharTy>();
  if (fromIntTy && toCharTy) {
    return builder.create<CIL::TruncateOp>(loc, toTy, value);
  }
  auto fromCharTy = fromTy.dyn_cast_or_null<CIL::CharTy>();
  if (fromCharTy && toIntTy) {
    return builder.create<CIL::ZeroExtendOp>(loc, toTy, value);
  }

  value.dump();
  llvm::errs() << " to \n";
  toTy.dump();
  llvm_unreachable("Unhandled cast");
}

mlir::Value MLIRCodeGen::emitCastExpr(Expr *expr, clang::QualType toType) {
  auto srcType = expr->getType();
  auto value = emitExpression(expr);
  auto loc = getLoc(expr->getSourceRange());
  if (auto BT = dyn_cast<clang::BuiltinType>(toType)) {
    if (BT->isVoidType()) {
      return {};
    }
  }
  auto mlirType = cgTypes.convertClangType(toType);
  auto fromType = cgTypes.convertClangType(srcType);

  auto fromClangTy = srcType->getUnqualifiedDesugaredType();

  switch (fromClangTy->getTypeClass()) {
  case clang::Type::TypeClass::Elaborated:
    if (auto typeDef = dyn_cast<ElaboratedType>(srcType)) {
      srcType = typeDef->desugar();
    }
  case clang::Type::TypeClass::Typedef:
    if (auto typeDef = dyn_cast<TypedefType>(srcType)) {
      srcType = typeDef->desugar();
    }
  case clang::Type::TypeClass::Builtin: {
    if (srcType->isBooleanType() && toType->isIntegerType()) {
      return builder.create<CIL::ZeroExtendOp>(loc, mlirType, value);
    }

    if (srcType->isIntegerType() && toType->isPointerType()) {
      return builder.create<CIL::CILIntToPtrOp>(loc, mlirType, value);
    }

    auto fromIntTy = fromType.dyn_cast_or_null<CIL::IntegerTy>();
    auto toIntTy = mlirType.dyn_cast_or_null<CIL::IntegerTy>();

    if (fromIntTy && toIntTy) {
      if (fromIntTy == toIntTy)
        return value;

      if (fromIntTy.getIntegerKind() == toIntTy.getIntegerKind()) {
        // llvm_unreachable("emitCastExpr: integers of same type?");
        // FIXME: this might need some handling!!
        return builder.create<CIL::IntCastOp>(loc, mlirType, value);
      }
      if (fromIntTy.getIntegerKind() < toIntTy.getIntegerKind()) {
        if (toIntTy.getQualifiers().isUnsigned()) {
          return builder.create<CIL::ZeroExtendOp>(loc, mlirType, value);
        } else {
          return builder.create<CIL::SignExtendOp>(loc, mlirType, value);
        }
      } else {
        if (fromIntTy.getIntegerKind() == CIL::BoolKind) {
          // We can not use trunc for int to bool
          // For example, from llvm langref
          //    %Z = trunc i32 122 to i1 ; yields i1:false
          auto zero = builder.getCILIntegralConstant(loc, fromIntTy, 0);
          return builder.create<CIL::CILCmpIOp>(loc, CmpIPredicate::ne, value,
                                                zero);
        }
        return builder.create<CIL::TruncateOp>(loc, mlirType, value);
      }
    }
    auto toCharTy = mlirType.dyn_cast_or_null<CIL::CharTy>();
    if (fromIntTy && toCharTy) {
      return builder.create<CIL::TruncateOp>(loc, mlirType, value);
    }
    auto fromCharTy = fromType.dyn_cast_or_null<CIL::CharTy>();
    if (fromCharTy && toIntTy) {
      return builder.create<CIL::ZeroExtendOp>(loc, mlirType, value);
    }

    // FIXME: How do we handle (void)var ?
    if (toType->isVoidType())
      return value;

    if (toIntTy && fromType.isa<CIL::FloatingTy>()) {
      return builder.create<CIL::CILFPToSIOp>(loc, mlirType, value);
    }

    if (fromIntTy && mlirType.isa<CIL::FloatingTy>()) {
      return builder.create<CIL::CILSIToFPOp>(loc, mlirType, value);
    }
    fromType.dump();
    llvm::errs() << "\n";
    mlirType.dump();
    llvm_unreachable("Unknown int cast");
  }
  case clang::Type::TypeClass::Pointer: {
    return builder.create<CIL::PointerBitCastOp>(loc, mlirType, value);
  }
  // FIXME: hardcoded for STL case
  case clang::Type::Enum: {
    return builder.create<CIL::IntCastOp>(loc, mlirType, value);
  }
  default:
    // FIXME :  Can the unsigned integer (for enum types) directly casted to
    // signed integers?
    if (expr->getType()->isEnumeralType() &&
        (toType->isUnsignedIntegerType() || toType->isIntegerType())) {
      return value;
    }
    break;
  };
  expr->dump();
  toType->dump();
  llvm_unreachable("unknown cast expression");
}

mlir::Value MLIRCodeGen::emitArraySubscriptExpr(ArraySubscriptExpr *expr) {
  auto base = expr->getBase();
  auto loc = getLoc(expr->getSourceRange());
  auto implict = dyn_cast<ImplicitCastExpr>(base);
  assert(implict);
  base = implict->getSubExpr();
  bool isValidParam = true;
  if (auto parenExpr = dyn_cast<clang::ParenExpr>(base)) {
    if (!isa<DeclRefExpr>(parenExpr->getSubExpr()))
      isValidParam = false;
  }
  assert(isa<DeclRefExpr>(base) || isa<ArraySubscriptExpr>(base) ||
         isValidParam);
  auto baseArray = emitExpression(base);
  auto baseArrayTy = baseArray.getType().cast<CIL::PointerType>().getEleTy();
  auto offset = emitExpression(expr->getIdx());
  if (implict->getCastKind() == CK_ArrayToPointerDecay) {
    assert(baseArrayTy.isa<CIL::ArrayType>());
    return builder.create<CIL::CILArrayIndexOp>(loc, baseArray, offset);
  }
  if (implict->getCastKind() == CK_LValueToRValue) {
    // TODO: tag with array expression metadata?
    auto load = builder.create<CIL::CILLoadOp>(loc, baseArray);
    return builder.create<CIL::CILPointerAddOp>(loc, load, offset);
  }
  implict->dump();
  llvm_unreachable("unknown array expression implict kind");
}

mlir::Value MLIRCodeGen::emitUnaryOperator(UnaryOperator *unaryOp) {
  auto loc = getLoc(unaryOp->getSourceRange());
  switch (unaryOp->getOpcode()) {
    // TODO: Currently these are just being used as pass through.
  // We can create two operations like address_of and dereference to
  // explicitly denote dereference.
  case UnaryOperatorKind::UO_Deref:
  case UnaryOperatorKind::UO_AddrOf: {
    return emitExpression(unaryOp->getSubExpr());
  }
  case UnaryOperatorKind::UO_PostInc:
  case UnaryOperatorKind::UO_PreInc: {
    auto subExpr = emitExpression(unaryOp->getSubExpr());
    auto load = builder.create<CIL::CILLoadOp>(loc, subExpr);
    if (load.getType().isa<CIL::PointerType>()) {
      auto one =
          builder.getCILIntegralConstant(loc, builder.getCILIntType(), 1);
      auto incr = builder.create<CIL::CILGEPOp>(loc, load.getType(), load, one);
      builder.create<CIL::CILStoreOp>(loc, incr, subExpr);
      if (unaryOp->getOpcode() == UnaryOperatorKind::UO_PostInc)
        return load;
      else {
        load = builder.create<CIL::CILLoadOp>(loc, subExpr);
        return load;
      }
    }
    assert(load.getType().isa<CIL::IntegerTy>());
    auto one = builder.getCILIntegralConstant(loc, load.getType(), 1);
    auto add = builder.create<CIL::CILAddIOp>(loc, load, one);
    builder.create<CIL::CILStoreOp>(loc, add, subExpr);
    if (unaryOp->getOpcode() == UnaryOperatorKind::UO_PostInc)
      return load;
    else {
      load = builder.create<CIL::CILLoadOp>(loc, subExpr);
      return load;
    }
  }
  case UnaryOperatorKind::UO_PostDec:
  case UnaryOperatorKind::UO_PreDec: {
    auto subExpr = emitExpression(unaryOp->getSubExpr());
    auto load = builder.create<CIL::CILLoadOp>(loc, subExpr);
    if (load.getType().isa<CIL::PointerType>()) {
      auto one =
          builder.getCILIntegralConstant(loc, builder.getCILIntType(), -1);
      auto incr = builder.create<CIL::CILGEPOp>(loc, load.getType(), load, one);
      builder.create<CIL::CILStoreOp>(loc, incr, subExpr);
      if (unaryOp->getOpcode() == UnaryOperatorKind::UO_PostDec)
        return load;
      else {
        load = builder.create<CIL::CILLoadOp>(loc, subExpr);
        return load;
      }
    }
    assert(load.getType().isa<CIL::IntegerTy>());
    auto one = builder.getCILIntegralConstant(loc, load.getType(), 1);
    auto sub = builder.create<CIL::CILSubIOp>(loc, load, one);
    builder.create<CIL::CILStoreOp>(loc, sub, subExpr);
    if (unaryOp->getOpcode() == UnaryOperatorKind::UO_PostDec)
      return load;
    else {
      load = builder.create<CIL::CILLoadOp>(loc, subExpr);
      return load;
    }
  }
  case UnaryOperatorKind::UO_LNot: {
    auto mlirType = cgTypes.convertClangType(unaryOp->getType());
    auto subExpr = emitExpression(unaryOp->getSubExpr());
    if (subExpr.getType().isa<CIL::IntegerTy>()) {
      auto zero = builder.getCILIntegralConstant(loc, subExpr.getType(), 0);
      auto cmpOp =
          builder.create<CIL::CILCmpIOp>(loc, CmpIPredicate::eq, subExpr, zero);
      auto zext = builder.create<CIL::ZeroExtendOp>(
          loc, builder.getCILIntType(), cmpOp);
      return emitCast(zext, mlirType);
    }

    if (subExpr.getType().isa<CIL::PointerType>()) {
      auto null = builder.create<CIL::NullPointerOp>(loc, subExpr.getType());
      auto lhs = builder.create<CIL::CILPtrToIntOp>(
          loc, builder.getCILLongIntType(), subExpr);
      // FIXME: This is just workaround, we should be able create null value of
      //        anytype
      auto rhs = builder.create<CIL::CILPtrToIntOp>(
          loc, builder.getCILLongIntType(), null);
      return builder.create<CIL::CILCmpIOp>(loc, CmpIPredicate::eq, lhs, rhs);
    }
    llvm_unreachable("Unhandled not");
  }
  case UnaryOperatorKind::UO_Plus: {
    auto subExpr = emitExpression(unaryOp->getSubExpr());
    return subExpr;
  }
  case UnaryOperatorKind::UO_Minus: {
    auto subExpr = emitExpression(unaryOp->getSubExpr());
    auto resTy = subExpr.getType();
    if (resTy.isa<CIL::IntegerTy>()) {
      auto zero = builder.getCILIntegralConstant(loc, resTy, 0);
      return builder.create<CIL::CILSubIOp>(loc, zero, subExpr);
    }

    if (resTy.isa<CIL::FloatingTy>()) {
      auto zero = builder.getCILFloatingConstant(loc, resTy, 0);
      return builder.create<CIL::CILSubFOp>(loc, zero, subExpr);
    }
    resTy.dump();
    llvm_unreachable("Unknown type for unary minus");
  }
  case UnaryOperatorKind::UO_Not: {
    // UO_Not is for bitwise negation
    auto subExpr = emitExpression(unaryOp->getSubExpr());
    auto resTy = subExpr.getType();
    if (resTy.isa<CIL::IntegerTy>()) {
      auto one = builder.getCILIntegralConstant(loc, resTy, 1);
      auto zero = builder.getCILIntegralConstant(loc, resTy, 0);
      auto incrementSub = builder.create<CIL::CILAddIOp>(loc, subExpr, one);
      return builder.create<CIL::CILSubIOp>(loc, zero,
                                            incrementSub.getResult());
    }
    resTy.dump();
    llvm_unreachable("Unknown type for unary not");
  }
  default:
    unaryOp->dump();
    llvm_unreachable("unhandled unary operator");
  }
}

mlir::Value MLIRCodeGen::emitDeclRefExpr(clang::DeclRefExpr *declRef) {
  auto loc = getLoc(declRef->getSourceRange());
  if (auto *enumConst =
          llvm::dyn_cast<const EnumConstantDecl>(declRef->getDecl())) {
    auto I32 = builder.getCILIntType();
    return builder.getCILIntegralConstant(
        loc, I32, enumConst->getInitVal().getSExtValue());
  }
  const NamedDecl *varDecl = declRef->getDecl()->getUnderlyingDecl();
  if (auto funcDecl = dyn_cast<FunctionDecl>(varDecl)) {
    auto funcOp = emitCallee(declRef->getDecl());
    auto ptrTy = CIL::PointerType::get(funcOp.getType());
    auto funcPtr = builder.create<CIL::CILConstantOp>(
        loc, ptrTy, builder.getSymbolRefAttr(funcOp));
    return funcPtr;
  }
  auto allocaItr = allocaMap.lookup(varDecl);
  if (!allocaItr) {
    emitDecl(declRef->getDecl());
    allocaItr = allocaMap.lookup(varDecl);
  }
  assert(allocaItr);
  auto addrOfOp = allocaItr.getDefiningOp();
  if (auto addrOf = dyn_cast_or_null<CIL::GlobalAddressOfOp>(addrOfOp)) {
    auto cloned = addrOf.clone();
    builder.insert(cloned);
    return cloned;
  }
  auto ptrType = allocaItr.getType().cast<CIL::PointerType>();
  if (builder.isCILRefType(ptrType.getEleTy())) {
    return builder.create<CIL::CILLoadOp>(loc, allocaItr);
  }
  return allocaItr;
}

mlir::Value MLIRCodeGen::emitExpression(Expr *expr) {
  switch (expr->getStmtClass()) {
  case Expr::ConstantExprClass: {
    return emitExpression(
        static_cast<clang::ConstantExpr *>(expr)->getSubExpr());
  }
  case Expr::IntegerLiteralClass: {
    return emitIntegerLiteral(static_cast<clang::IntegerLiteral *>(expr));
  }
  case Expr::CXXBoolLiteralExprClass: {
    auto boolLiteral = static_cast<clang::CXXBoolLiteralExpr *>(expr);
    return builder.getCILIntegralConstant(getLoc(expr->getBeginLoc()),
                                          builder.getCILBoolType(),
                                          boolLiteral->getValue());
  }
  case Expr::CharacterLiteralClass: {
    auto loc = getLoc(expr->getSourceRange());
    auto charLit = static_cast<clang::CharacterLiteral *>(expr);
    auto value = charLit->getValue();
    auto type = builder.getCILUCharType();
    auto valueAttr = builder.getI8IntegerAttr(value);
    return builder.create<CIL::CILConstantOp>(loc, type, valueAttr);
  }
  case Expr::StringLiteralClass: {
    return emitStringLiteral(static_cast<clang::StringLiteral *>(expr));
  }
  case Expr::FloatingLiteralClass: {
    return emitFloatingLiteral(static_cast<clang::FloatingLiteral *>(expr));
  }
  case Stmt::BinaryOperatorClass: {
    auto binOperator = static_cast<clang::BinaryOperator *>(expr);
    return emitBinaryOperator(binOperator);
  }
  case Stmt::UnaryOperatorClass: {
    auto unaryOperator = static_cast<clang::UnaryOperator *>(expr);
    return emitUnaryOperator(unaryOperator);
  }
  case Stmt::ConditionalOperatorClass: {
    return emitConditionalOperator(
        static_cast<clang::ConditionalOperator *>(expr));
  }
  case Stmt::ArraySubscriptExprClass: {
    auto arrySubs = static_cast<clang::ArraySubscriptExpr *>(expr);
    return emitArraySubscriptExpr(arrySubs);
  }
  case Expr::DeclRefExprClass: {
    return emitDeclRefExpr(static_cast<clang::DeclRefExpr *>(expr));
  }
  case Expr::CXXThisExprClass: {
    auto thisExpr = static_cast<CXXThisExpr *>(expr);
    auto type = cgTypes.convertClangType(thisExpr->getType());
    return builder.create<CIL::ThisOp>(getLoc(thisExpr->getSourceRange()),
                                       type);
  }
  case Expr::ParenExprClass: {
    auto parenExpr = static_cast<clang::ParenExpr *>(expr);
    return emitExpression(parenExpr->getSubExpr());
  }
  case Expr::CStyleCastExprClass: {
    auto castExpr = static_cast<clang::CStyleCastExpr *>(expr);
    auto subExpr = castExpr->getSubExpr();
    return emitCastExpr(subExpr, castExpr->getType());
  }
  case Expr::ImplicitCastExprClass: {
    return emitImplicitCastExpr(static_cast<clang::ImplicitCastExpr *>(expr));
  }
  case Stmt::MemberExprClass: {
    return emitMemberExpr(static_cast<clang::MemberExpr *>(expr));
  }
  case Expr::CallExprClass: {
    return emitCallExpr(static_cast<clang::CallExpr *>(expr));
  } break;
  case Stmt::CXXMemberCallExprClass: {
    auto call =
        emitCXXMemberCall(static_cast<clang::CXXMemberCallExpr *>(expr));
    if (call.getNumResults() == 0)
      return {};
    assert(call.getNumResults() == 1);
    return call.getResult(0);
  } break;
  case Stmt::CXXOperatorCallExprClass: {
    auto operatorCall = static_cast<clang::CXXOperatorCallExpr *>(expr);
    return emitCallExpr(operatorCall);
  } break;
  case Stmt::UnaryExprOrTypeTraitExprClass: {
    auto typeTraitExpr = static_cast<clang::UnaryExprOrTypeTraitExpr *>(expr);
    QualType qType = typeTraitExpr->getTypeOfArgument();
    auto loc = getLoc(expr->getSourceRange());

    // TODO: Create an operation for sizeof. Size is not known at CIL level.
    if (typeTraitExpr->getKind() == UETT_SizeOf) {
      auto charSize = context.getTypeSizeInChars(qType);
      auto resTy = builder.getCILULongIntType();
      auto valueAttr = builder.getI64IntegerAttr(charSize.getQuantity());
      return builder.create<CIL::CILConstantOp>(loc, resTy, valueAttr);
    }

    if (typeTraitExpr->getKind() == UETT_AlignOf) {
      auto alignSize = context.getTypeAlign(qType);
      auto resTy = builder.getCILULongIntType();
      auto valueAttr = builder.getI64IntegerAttr(alignSize);
      return builder.create<CIL::CILConstantOp>(loc, resTy, valueAttr);
    }

    llvm_unreachable("Unhandled");
  }
  case Stmt::CXXConstCastExprClass: {
    auto castExpr = static_cast<clang::CXXConstCastExpr *>(expr);
    return emitExpression(castExpr->getSubExpr());
  }
  case Stmt::CXXFunctionalCastExprClass: {
    auto castExpr = static_cast<clang::CXXFunctionalCastExpr *>(expr);
    if (castExpr->isListInitialization()) {
      auto initListExpr = dyn_cast<clang::InitListExpr>(castExpr->getSubExpr());
      assert(initListExpr && "not sure which other expression can be here");
      initListExpr->dump();

      // FIXME: completely hardcoded for STL
      auto ty = initListExpr->getType();
      auto initTy = ty.getTypePtr();
      initTy = initTy->getUnqualifiedDesugaredType();
      if (auto recordTy = dyn_cast<RecordType>(initTy)) {
        if (auto decl = dyn_cast<CXXRecordDecl>(recordTy->getDecl())) {
          emitClassDecl(decl);

          for (auto method : decl->methods()) {
            if (auto CD = dyn_cast<CXXConstructorDecl>(method)) {
              auto className = mangler.mangleClassName(decl);
              auto classType =
                  builder.getCILPointerType(builder.getCILClassType(className));
              auto loc = builder.getUnknownLoc();
              auto base = builder.create<CIL::ThisOp>(loc, classType);

              SmallVector<clang::Expr *, 2> argList;
              emitMemberCallOp(base.getResult(), decl, CD, argList);
              return base.getResult();
            }
          }
        }
      }

      llvm_unreachable("not sure how to handle");
    } else {
      return emitExpression(castExpr->getSubExpr());
    }
  } break;
  case Stmt::CXXStaticCastExprClass: {
    auto castExpr = static_cast<clang::CXXStaticCastExpr *>(expr);
    if (castExpr->getCastKind() == clang::CK_NoOp) {
      return emitExpression(castExpr->getSubExpr());
    }
    if (castExpr->getCastKind() == clang::CK_BitCast) {
      return emitExpression(castExpr->getSubExpr());
    }
    castExpr->dump();
    llvm_unreachable("unhandled cast expression");
  }
  case Stmt::ImplicitValueInitExprClass: {
    auto valExpr = static_cast<clang::ImplicitValueInitExpr *>(expr);
    auto type = valExpr->getType();
    assert(type->isPointerType());
    auto loc = builder.getUnknownLoc();
    return builder.create<CIL::NullPointerOp>(loc,
                                              cgTypes.convertClangType(type));
  }
  case Stmt::ExprWithCleanupsClass: {
    auto castExpr = static_cast<clang::ExprWithCleanups *>(expr);
    castExpr->getSubExpr()->dump();
    return emitExpression(castExpr->getSubExpr());
  }
  case Stmt::MaterializeTemporaryExprClass: {
    auto castExpr = static_cast<clang::MaterializeTemporaryExpr *>(expr);
    return emitExpression(castExpr->getSubExpr());
  }
  case Stmt::CXXNewExprClass: {
    return emitCXXNewExpr(static_cast<clang::CXXNewExpr *>(expr));
  }
  case Stmt::CXXDefaultArgExprClass: {
    auto defArgExpr = static_cast<clang::CXXDefaultArgExpr *>(expr);
    return emitExpression(defArgExpr->getExpr());
  }
  case Stmt::CXXScalarValueInitExprClass: {
    auto scalarInitExpr = static_cast<clang::CXXScalarValueInitExpr *>(expr);
    // TODO: May be we need to emit a null value
    auto loc = builder.getUnknownLoc();
    auto type = scalarInitExpr->getType();
    return builder.create<CIL::NullPointerOp>(loc,
                                              cgTypes.convertClangType(type));
  }
  // TODO: Should be handled along with try , catch and throw expressions
  // handling.
  case Stmt::CXXNoexceptExprClass: {
    auto noExceptExpr = static_cast<clang::CXXNoexceptExpr *>(expr);
    auto loc = getLoc(noExceptExpr->getBeginLoc());
    auto boolTy = builder.getCILBoolType();
    auto valueAttr = builder.getBoolAttr(1);
    return builder.create<CIL::CILConstantOp>(loc, boolTy, valueAttr);
  }
  case Stmt::TypeTraitExprClass: {
    auto typeTraitExpr = static_cast<clang::TypeTraitExpr *>(expr);
    auto loc = getLoc(typeTraitExpr->getBeginLoc());
    auto boolTy = builder.getCILBoolType();
    auto val = typeTraitExpr->getValue();
    auto valueAttr = builder.getBoolAttr(val);
    return builder.create<CIL::CILConstantOp>(loc, boolTy, valueAttr);
  }
  case Stmt::CXXConstructExprClass: {
    auto constructExpr = static_cast<clang::CXXConstructExpr *>(expr);
    auto CD = constructExpr->getConstructor();
    auto classDecl = cast<CXXRecordDecl>(CD->getParent());
    auto loc = getLoc(constructExpr->getLocation());
    auto className = mangler.mangleClassName(classDecl);
    auto classType = builder.getCILClassType(className);
    emitClassDecl(classDecl);

    if (CD->isImplicit()) {
      auto alloc = builder.create<CIL::AllocaOp>(loc, "", classType);
      return builder.create<CIL::CILLoadOp>(loc, alloc);
    }

    emitCXXMethodDecl(CD);

    SmallVector<mlir::Value, 2> argList;
    for (auto II = 0; II < constructExpr->getNumArgs(); ++II) {
      argList.push_back(emitExpression(constructExpr->getArg(II)));
    }

    auto mangledName = mangler.mangleFunctionName(CD);
    auto symRefAttr = builder.getSymbolRefAttr(
        mangledName, builder.getSymbolRefAttr(classDecl->getName()));
    auto alloc =
        builder.create<CIL::AllocaOp>(loc, "", classType, symRefAttr, argList);
    return builder.create<CIL::CILLoadOp>(loc, alloc);
  }
  case Stmt::CXXTemporaryObjectExprClass: {
    auto tempObjStmt = static_cast<clang::CXXTemporaryObjectExpr *>(expr);

    auto CD = tempObjStmt->getConstructor();
    auto classDecl = cast<CXXRecordDecl>(CD->getParent());
    emitClassDecl(classDecl);

    auto loc = getLoc(tempObjStmt->getLocation());
    auto className = mangler.mangleClassName(classDecl);
    auto classType =
        builder.getCILPointerType(builder.getCILClassType(className));
    auto base = builder.create<CIL::ThisOp>(loc, classType);

    SmallVector<clang::Expr *, 2> args;
    for (auto II = 0; II < tempObjStmt->getNumArgs(); ++II) {
      args.push_back(tempObjStmt->getArg(II));
    }

    emitMemberCallOp(base.getResult(), classDecl, CD, args);
    return base.getResult();
  }
  case Stmt::CXXStdInitializerListExprClass: {
    auto stdInitList = static_cast<clang::CXXStdInitializerListExpr *>(expr);
    if (auto tempExpr = dyn_cast<clang::MaterializeTemporaryExpr>(
            stdInitList->getSubExpr())) {
      if (auto initList =
              dyn_cast<clang::InitListExpr>(tempExpr->getSubExpr())) {
        auto mlirType = cgTypes.convertClangType(stdInitList->getType());
        auto loc = getLoc(expr->getSourceRange());
        auto valueAttr = emitValueAttrForGlobal(loc, initList, mlirType);
        static unsigned long long strTemp1 = 0;
        std::string name = "__cxx_str_tmp" + std::to_string(strTemp1++);
        auto init = builder.create<CIL::GlobalOp>(loc, mlirType, false, name,
                                                  valueAttr);
        return init;
      }
    }
  }
  case Stmt::CXXNullPtrLiteralExprClass: {
    auto nullPtrExpr = static_cast<clang::CXXNullPtrLiteralExpr *>(expr);
    auto mlirType = cgTypes.convertClangType(nullPtrExpr->getType());
    auto loc = getLoc(nullPtrExpr->getLocation());
    return builder.create<CIL::NullPointerOp>(loc, mlirType);
  }
  case Stmt::SubstNonTypeTemplateParmExprClass: {
    auto tempParamExpr =
        static_cast<clang::SubstNonTypeTemplateParmExpr *>(expr);
    return emitExpression(tempParamExpr->getReplacement());
  }
  case Stmt::PredefinedExprClass: {
    return emitPredefinedExpr(static_cast<clang::PredefinedExpr *>(expr));
  }
  case Stmt::CompoundAssignOperatorClass: {
    return emitCompoundAssignment(static_cast<CompoundAssignOperator *>(expr));
  }
  default: {
    emitStmt(expr);
    return {};
  }
  }
  return {};
}

mlir::Value MLIRCodeGen::emitCXXNewExpr(clang::CXXNewExpr *newExpr) {
  auto loc = getLoc(newExpr->getBeginLoc());
  auto newFn = newExpr->getOperatorNew();
  auto currBlock = builder.getBlock();

  auto func = emitFunction(newFn);

  builder.setInsertionPointToEnd(currBlock);
  auto allocType = newExpr->getAllocatedType();
  // FIXME :use sizeof operation.
  auto charSize = context.getTypeSizeInChars(allocType);
  auto val = builder.getCILLongConstant(loc, charSize.getQuantity());
  auto call = builder.create<CIL::CILCallOp>(loc, func, val);
  auto toType = cgTypes.convertClangType(newExpr->getType());
  auto bitcast =
      builder.create<CIL::PointerBitCastOp>(loc, toType, call.getResult(0));
  auto constructExpr = newExpr->getConstructExpr();
  if (!constructExpr)
    return bitcast;
  emitCXXConstructExpr(const_cast<CXXConstructExpr *>(constructExpr),
                       bitcast.getResult());
  return bitcast;
}

mlir::Value MLIRCodeGen::emitCXXMemberExpr(MemberExpr *expr) {
  auto valueDecl = expr->getMemberDecl();
  Expr *baseExpr = expr->getBase();
  auto field = llvm::dyn_cast<clang::FieldDecl>(valueDecl);
  assert(field);
  return emitFieldAccessOp(field, expr->getBeginLoc(), baseExpr);
}

mlir::Value MLIRCodeGen::emitMemberExpr(MemberExpr *expr) {
  auto valueDecl = expr->getMemberDecl();
  auto field = llvm::dyn_cast<clang::FieldDecl>(valueDecl);
  assert(field);
  if (field->getParent()->isClass()) {
    return emitCXXMemberExpr(expr);
  }
  assert(field);
  Expr *baseExpr = expr->getBase();
  auto baseValue = emitExpression(baseExpr);
  auto basePtrType = baseValue.getType().cast<CIL::PointerType>();
  assert(baseValue);
  auto loc = getLoc(expr->getSourceRange());
  auto resType = builder.getCILPointerType(
      cgTypes.convertClangType(expr->getType()), basePtrType.isReference());
  auto I32 = builder.getCILIntType();
  auto index = builder.getCILIntegralConstant(loc, I32, field->getFieldIndex());
  auto structEle =
      builder.create<CIL::StructElementOp>(loc, resType, baseValue, index);
  return structEle;
}

mlir::FuncOp MLIRCodeGen::emitCallee(Decl *decl) {

  auto funcDecl = dyn_cast<FunctionDecl>(decl);
  assert(funcDecl);

  // Remove from deferred.
  if (DeferredDecls.find(funcDecl) != DeferredDecls.end()) {
    DeferredDecls.erase(funcDecl);
  }
  auto savept = builder.saveInsertionPoint();
  auto funcOp = emitFunction(funcDecl);
  builder.restoreInsertionPoint(savept);
  return funcOp;
}

mlir::Value MLIRCodeGen::emitIndirectCall(CallExpr *expr) {
  auto varDecl = dyn_cast<VarDecl>(expr->getCalleeDecl());
  assert(varDecl);
  auto allocaItr = allocaMap.lookup(varDecl);
  assert(allocaItr);

  auto loc = getLoc(expr->getSourceRange());
  SmallVector<mlir::Value, 2> argList;
  for (auto II = 0; II < expr->getNumArgs(); ++II) {
    argList.push_back(emitExpression(expr->getArg(II)));
  }

  if (allocaItr.getType().isa<CIL::PointerType>())
    allocaItr = builder.create<CIL::CILLoadOp>(loc, allocaItr);

  auto callOp = builder.create<CIL::CILCallIndirectOp>(loc, allocaItr, argList);
  if (callOp.getNumResults() == 0)
    return {};
  assert(callOp.getNumResults() == 1);
  return callOp.getResult(0);
}

mlir::Value MLIRCodeGen::emitCallExpr(CallExpr *expr) {
  if (isa<VarDecl>(expr->getCalleeDecl())) {
    return emitIndirectCall(expr);
  }

  auto funcOp = emitCallee(expr->getCalleeDecl());
  assert(funcOp);
  auto loc = getLoc(expr->getSourceRange());
  SmallVector<mlir::Value, 2> argList;
  for (auto II = 0; II < expr->getNumArgs(); ++II) {
    argList.push_back(emitExpression(expr->getArg(II)));
  }
  auto callOp = builder.create<CIL::CILCallOp>(loc, funcOp, argList);
  if (callOp.getNumResults() == 0)
    return {};
  assert(callOp.getNumResults() == 1);
  return callOp.getResult(0);
}

mlir::Value MLIRCodeGen::emitRelationalOperator(BinaryOperator *binOp,
                                                mlir::Location loc,
                                                mlir::Value lhs,
                                                mlir::Value rhs,
                                                BinaryOperatorKind kind) {
  bool isFloat = false;

  if (lhs.getType().isa<CIL::FloatingTy>())
    isFloat = true;

  if (lhs.getType() != rhs.getType()) {
    if (lhs.getType().isa<CIL::CharTy>() || rhs.getType().isa<CIL::IntegerTy>())
      lhs = builder.create<CIL::ZeroExtendOp>(loc, rhs.getType(), lhs);
    if (rhs.getType().isa<CIL::CharTy>() || lhs.getType().isa<CIL::IntegerTy>())
      rhs = builder.create<CIL::ZeroExtendOp>(loc, lhs.getType(), rhs);
  }

  assert(lhs.getType() == rhs.getType());

  if (isFloat) {
    CmpFPredicate pred;
    switch (kind) {
    case BinaryOperatorKind::BO_LT:
      pred = CmpFPredicate::OLT;
      break;
    case BinaryOperatorKind::BO_LE:
      pred = CmpFPredicate::OLE;
      break;
    case BinaryOperatorKind::BO_EQ:
      pred = CmpFPredicate::OEQ;
      break;
    case BinaryOperatorKind::BO_GT:
      pred = CmpFPredicate::OGT;
      break;
    case BinaryOperatorKind::BO_GE:
      pred = CmpFPredicate::OGE;
      break;
    case BinaryOperatorKind::BO_NE:
      pred = CmpFPredicate::ONE;
      break;
    default:
      binOp->dump();
      llvm_unreachable("unhandled relational operator");
    };
    auto value = builder.create<CIL::CILCmpFOp>(loc, pred, lhs, rhs);
    if (!binOp->getType()->isBooleanType()) {
      auto toType = cgTypes.convertClangType(binOp->getType());
      return builder.create<CIL::ZeroExtendOp>(loc, toType, value);
    }
    return value;
  }

  if (lhs.getType().isa<CIL::PointerType>()) {
    lhs = builder.create<CIL::CILPtrToIntOp>(loc, builder.getCILLongIntType(),
                                             lhs);
  }

  if (rhs.getType().isa<CIL::PointerType>()) {
    rhs = builder.create<CIL::CILPtrToIntOp>(loc, builder.getCILLongIntType(),
                                             rhs);
  }

  CmpIPredicate pred;
  switch (kind) {
  case BinaryOperatorKind::BO_LT:
    pred = CmpIPredicate::slt;
    break;
  case BinaryOperatorKind::BO_LE:
    pred = CmpIPredicate::sle;
    break;
  case BinaryOperatorKind::BO_EQ:
    pred = CmpIPredicate::eq;
    break;
  case BinaryOperatorKind::BO_GT:
    pred = CmpIPredicate::sgt;
    break;
  case BinaryOperatorKind::BO_GE:
    pred = CmpIPredicate::sge;
    break;
  case BinaryOperatorKind::BO_NE:
    pred = CmpIPredicate::ne;
    break;
  default:
    binOp->dump();
    llvm_unreachable("unhandled relational operator");
  };
  auto value = builder.create<CIL::CILCmpIOp>(loc, pred, lhs, rhs);
  // Zero extend to type of the binary operator.
  if (!binOp->getType()->isBooleanType()) {
    auto toType = cgTypes.convertClangType(binOp->getType());
    return builder.create<CIL::ZeroExtendOp>(loc, toType, value);
  }
  return value;
}

mlir::Value MLIRCodeGen::emitBinaryOperator(BinaryOperator *binOp) {
  auto lhs = emitExpression(binOp->getLHS());
  auto rhs = emitExpression(binOp->getRHS());
  auto loc = getLoc(binOp->getSourceRange(), true);
  assert(lhs && rhs);

  if (binOp->isRelationalOp() ||
      binOp->getOpcode() == BinaryOperatorKind::BO_EQ ||
      binOp->getOpcode() == BinaryOperatorKind::BO_NE) {
    return emitRelationalOperator(binOp, loc, lhs, rhs, binOp->getOpcode());
  }
  switch (binOp->getOpcode()) {
  case BinaryOperatorKind::BO_Assign: {
    builder.create<CIL::CILStoreOp>(loc, rhs, lhs);
    // FIXME: This causes loads for assignment statements as well. Need to do
    // it selectively.
    return builder.create<CIL::CILLoadOp>(loc, lhs);
  }
  case BinaryOperatorKind::BO_Add: {
    if (binOp->getType()->isPointerType()) {
      return builder.create<CIL::CILPointerAddOp>(loc, lhs, rhs);
    }
    if (binOp->getType()->isIntegerType()) {
      auto finalTy = cgTypes.convertClangType(binOp->getType());
      if (lhs.getType() != finalTy)
        lhs = emitCast(lhs, finalTy);
      if (rhs.getType() != finalTy)
        rhs = emitCast(rhs, finalTy);
      return builder.create<CIL::CILAddIOp>(loc, lhs, rhs);
    }

    if (binOp->getType()->isFloatingType()) {
      return builder.create<CIL::CILAddFOp>(loc, lhs, rhs);
    }
  }
  case BinaryOperatorKind::BO_Sub: {
    if (binOp->getType()->isPointerType()) {
      auto zero = builder.getCILIntegralConstant(loc, rhs.getType(), 0);
      rhs = builder.create<CIL::CILSubIOp>(loc, zero, rhs);
      return builder.create<CIL::CILPointerAddOp>(loc, lhs, rhs);
    }
    if (lhs.getType().isa<CIL::PointerType>()) {
      auto finalTy = cgTypes.convertClangType(binOp->getType());
      if (rhs.getType().isa<CIL::IntegerTy>()) {
        auto zero =
            builder.getCILIntegralConstant(rhs.getLoc(), rhs.getType(), 0);
        auto idx = builder.create<CIL::CILSubIOp>(loc, zero, rhs);
        return builder.create<CIL::CILGEPOp>(loc, finalTy, lhs,
                                             idx.getResult());
      }
      assert(rhs.getType().isa<CIL::PointerType>());
      assert(finalTy.isa<CIL::IntegerTy>());
      auto ptrTy = rhs.getType().cast<CIL::PointerType>();
      lhs = builder.create<CIL::CILPtrToIntOp>(loc, finalTy, lhs);
      rhs = builder.create<CIL::CILPtrToIntOp>(loc, finalTy, rhs);
      auto ptrSub = builder.create<CIL::CILSubIOp>(loc, lhs, rhs);

      // size of calculatation
      auto nullPtr = builder.create<CIL::NullPointerOp>(loc, ptrTy);
      auto one = builder.getCILIntegralConstant(loc, finalTy, 1);
      auto gep = builder.create<CIL::CILGEPOp>(loc, ptrTy, nullPtr,
                                               ArrayRef<mlir::Value>{one});
      auto size = builder.create<CIL::CILPtrToIntOp>(loc, finalTy, gep);

      return builder.create<CIL::CILDivIOp>(loc, ptrSub, size);
    }
    if (binOp->getType()->isIntegerType()) {
      // FIXME: Hardcoded for STL vector testcase
      auto lhsTy = lhs.getType();
      auto rhsTy = rhs.getType();
      auto intTy1 = lhsTy.dyn_cast<CIL::IntegerTy>();
      auto intTy2 = rhsTy.dyn_cast<CIL::IntegerTy>();
      if ((intTy1 && intTy2) && lhsTy != rhsTy) {
        auto castRhs = builder.create<CIL::IntCastOp>(loc, lhsTy, rhs);
        return builder.create<CIL::CILSubIOp>(loc, lhs, castRhs);
      }
      return builder.create<CIL::CILSubIOp>(loc, lhs, rhs);
    }
    if (binOp->getType()->isFloatingType()) {
      return builder.create<CIL::CILSubFOp>(loc, lhs, rhs);
    }

    llvm_unreachable("Unknown types");
  }
  case BinaryOperatorKind::BO_Mul: {
    if (binOp->getType()->isIntegerType()) {
      return builder.create<CIL::CILMulIOp>(loc, lhs, rhs);
    }
    if (binOp->getType()->isFloatingType()) {
      return builder.create<CIL::CILMulFOp>(loc, lhs, rhs);
    }
  }
  case BinaryOperatorKind::BO_Rem: {
    if (binOp->getType()->isIntegerType()) {
      if (binOp->getType()->isUnsignedIntegerType())
        return builder.create<CIL::CILModUIOp>(loc, lhs, rhs);
      else
        return builder.create<CIL::CILModIOp>(loc, lhs, rhs);
    }
    if (binOp->getType()->isFloatingType()) {
      return builder.create<CIL::CILModFOp>(loc, lhs, rhs);
    }
  }
  case BinaryOperatorKind::BO_Div: {
    if (binOp->getType()->isIntegerType()) {
      return builder.create<CIL::CILDivIOp>(loc, lhs, rhs);
    }
    if (binOp->getType()->isFloatingType()) {
      return builder.create<CIL::CILDivFOp>(loc, lhs, rhs);
    }
  }
  case BinaryOperatorKind::BO_Shl:
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILShlOp>(loc, lhs, rhs);
  // TODO: Decide ashr or lshr
  case BinaryOperatorKind::BO_Shr:
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILAShrOp>(loc, lhs, rhs);
  case BinaryOperatorKind::BO_And:
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILAndOp>(loc, lhs, rhs);
  case BinaryOperatorKind::BO_Or:
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILOrOp>(loc, lhs, rhs);
  case BinaryOperatorKind::BO_Xor:
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILXOrOp>(loc, lhs, rhs);
  case BinaryOperatorKind::BO_LAnd: {
    lhs = getBoolTypeFor(lhs);
    rhs = getBoolTypeFor(rhs);
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILLogicalAndOp>(loc, lhs, rhs);
  }
  case BinaryOperatorKind::BO_LOr: {
    lhs = getBoolTypeFor(lhs);
    rhs = getBoolTypeFor(rhs);
    assert(binOp->getType()->isIntegerType());
    return builder.create<CIL::CILLogicalOrOp>(loc, lhs, rhs);
  }
  case BinaryOperatorKind::BO_Comma: {
    // RHS is emitted as lvalue for comma operator
    return rhs;
  }
  default:
    binOp->dump();
    llvm_unreachable("unknown binary kind operator");
  }
  return {};
}

mlir::Value MLIRCodeGen::emitSizeOf(mlir::Location loc,
                                    const clang::Type *type) {
  auto charSize = context.getTypeSizeInChars(type);
  auto resTy = builder.getCILIntType();
  auto valueAttr = builder.getI32IntegerAttr(charSize.getQuantity());
  return builder.create<CIL::CILConstantOp>(loc, resTy, valueAttr);
}

mlir::Value MLIRCodeGen::getBoolTypeFor(mlir::Value v) {
  if (auto zext = dyn_cast<CIL::ZeroExtendOp>(v.getDefiningOp())) {
    v = zext.getValue();
    // FIXME: Bug. Make sure the zext isn't generated in the first place.
    zext.erase();
    return v;
  }

  if (auto ptrTy = v.getType().dyn_cast_or_null<CIL::PointerType>()) {
    auto nullPointer = builder.create<CIL::NullPointerOp>(v.getLoc(), ptrTy);
    auto pointerEq = builder.create<CIL::CmpPointerEqualOp>(
        v.getLoc(), builder.getCILBoolType(), v, nullPointer);

    v = builder.create<CIL::LNotOp>(v.getLoc(), builder.getCILBoolType(),
                                    pointerEq);
    return v;
  }

  return emitCast(v, builder.getCILBoolType());
  ;
}

void MLIRCodeGen::emitCondExprIf(Expr *cond, Block *trueBlock,
                                 Block *falseBlock) {

  if (auto parenExpr = dyn_cast<clang::ParenExpr>(cond))
    cond = parenExpr->getSubExpr();

  if (BinaryOperator *condBOp = dyn_cast<BinaryOperator>(cond)) {
    // if (a && b)
    if (condBOp->getOpcode() == BinaryOperatorKind::BO_LAnd) {
      auto lhsTrue = getNewBlock(falseBlock);
      emitCondExprIf(condBOp->getLHS(), lhsTrue, falseBlock);
      builder.setInsertionPointToStart(lhsTrue);
      emitCondExprIf(condBOp->getRHS(), trueBlock, falseBlock);
      return;
    }

    if (condBOp->getOpcode() == BinaryOperatorKind::BO_LOr) {
      auto lhsFalse = getNewBlock(falseBlock);
      emitCondExprIf(condBOp->getLHS(), trueBlock, lhsFalse);
      builder.setInsertionPointToStart(lhsFalse);
      emitCondExprIf(condBOp->getRHS(), trueBlock, falseBlock);
      return;
    }
  }

  SmallVector<mlir::Value, 2> args;
  auto condValue = emitExpression(cond);
  condValue = getBoolTypeFor(condValue);
  auto loc = condValue.getLoc();
  builder.create<CIL::CILIfOp>(loc, condValue, trueBlock, args, falseBlock,
                               args);
  return;
}

void MLIRCodeGen::emitCondExprFor(Expr *cond, Block *trueBlock,
                                  Block *falseBlock) {

  if (auto parenExpr = dyn_cast<clang::ParenExpr>(cond))
    cond = parenExpr->getSubExpr();

  if (BinaryOperator *condBOp = dyn_cast<BinaryOperator>(cond)) {
    // if (a && b)
    if (condBOp->getOpcode() == BinaryOperatorKind::BO_LAnd) {
      auto lhsTrue = getNewBlock(falseBlock);
      emitCondExprFor(condBOp->getLHS(), lhsTrue, falseBlock);
      builder.setInsertionPointToStart(lhsTrue);
      emitCondExprFor(condBOp->getRHS(), trueBlock, falseBlock);
      return;
    }

    if (condBOp->getOpcode() == BinaryOperatorKind::BO_LOr) {
      auto lhsFalse = getNewBlock(falseBlock);
      emitCondExprFor(condBOp->getLHS(), trueBlock, lhsFalse);
      builder.setInsertionPointToStart(lhsFalse);
      emitCondExprFor(condBOp->getRHS(), trueBlock, falseBlock);
      return;
    }
  }

  SmallVector<mlir::Value, 2> args;
  auto condValue = emitExpression(cond);
  condValue = getBoolTypeFor(condValue);
  auto loc = condValue.getLoc();
  builder.create<CIL::CILForOp>(loc, condValue, trueBlock, args, falseBlock,
                                args);
  return;
}

void MLIRCodeGen::emitCondExprWhile(Expr *cond, Block *trueBlock,
                                    Block *falseBlock) {

  if (auto parenExpr = dyn_cast<clang::ParenExpr>(cond))
    cond = parenExpr->getSubExpr();

  if (BinaryOperator *condBOp = dyn_cast<BinaryOperator>(cond)) {
    // if (a && b)
    if (condBOp->getOpcode() == BinaryOperatorKind::BO_LAnd) {
      auto lhsTrue = getNewBlock(falseBlock);
      emitCondExprWhile(condBOp->getLHS(), lhsTrue, falseBlock);
      builder.setInsertionPointToStart(lhsTrue);
      emitCondExprWhile(condBOp->getRHS(), trueBlock, falseBlock);
      return;
    }

    if (condBOp->getOpcode() == BinaryOperatorKind::BO_LOr) {
      auto lhsFalse = getNewBlock(falseBlock);
      emitCondExprWhile(condBOp->getLHS(), trueBlock, lhsFalse);
      builder.setInsertionPointToStart(lhsFalse);
      emitCondExprWhile(condBOp->getRHS(), trueBlock, falseBlock);
      return;
    }
  }

  SmallVector<mlir::Value, 2> args;
  auto condValue = emitExpression(cond);
  condValue = getBoolTypeFor(condValue);
  auto loc = condValue.getLoc();
  builder.create<CIL::CILWhileOp>(loc, condValue, trueBlock, args, falseBlock,
                                  args);
  return;
}

mlir::Value MLIRCodeGen::emitConditionalOperator(ConditionalOperator *condOp) {
  auto cond = emitExpression(condOp->getCond());
  auto loc = cond.getLoc();
  cond = getBoolTypeFor(cond);

  auto exitBlock = getNewBlock();
  auto thenBlock = getNewBlock(exitBlock);
  auto elseBlock = getNewBlock(exitBlock);

  SmallVector<mlir::Value, 2> args;
  auto boolTy = builder.getCILBoolType();
  cond = emitCast(cond, boolTy);
  builder.create<CIL::CILIfOp>(loc, cond, thenBlock, args, elseBlock, args);

  builder.setInsertionPointToStart(thenBlock);
  auto value = emitExpression(condOp->getTrueExpr());
  if (value)
    builder.create<mlir::BranchOp>(loc, exitBlock, value);
  else
    builder.create<mlir::BranchOp>(loc, exitBlock);

  builder.setInsertionPointToStart(elseBlock);
  auto falseVal = emitExpression(condOp->getFalseExpr());
  if (falseVal) {
    builder.create<mlir::BranchOp>(loc, exitBlock, falseVal);

    exitBlock->addArgument(value.getType());
    builder.setInsertionPointToStart(exitBlock);
    return exitBlock->getArgument(0);
  } else {
    builder.create<mlir::BranchOp>(loc, exitBlock);
    builder.setInsertionPointToStart(exitBlock);
    return {};
  }
}

mlir::Value MLIRCodeGen::emitPredefinedExpr(PredefinedExpr *expr) {
  return emitStringLiteral(expr->getFunctionName());
}
