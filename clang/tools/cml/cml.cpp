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

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include <fstream>
#include <streambuf>
#include <string>

#include "clang/cil/dialect/CIL/CILDialect.h"
#include "clang/cil/pass/Pass.h"

#define DEBUG_TYPE "cml"

using namespace llvm;

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory MLIRCGCategory("MLIR Codegen category ");

enum Action {
  EmitRawAST,
  EmitAST,
  EmitIR,
  EmitMLIR,
  EmitLoopOpt,
  EmitLLVM,
  EmitBC,
  EmitASM,
  EmitExe,
};

cl::opt<Action> ActionOpt(
    cl::desc("Choose IR Type:"), cl::cat(MLIRCGCategory),
    cl::values(clEnumValN(EmitAST, "emit-ast",
                          "Emit AST (after semantic checks)"),
               clEnumValN(EmitRawAST, "emit-raw-ast",
                          "Emit AST (before semantic checks)"),
               clEnumValN(EmitMLIR, "emit-mlir", "Emit MLIR"),
               clEnumValN(EmitIR, "emit-ir", "Emit CIL"),
               clEnumValN(EmitLoopOpt, "loop-opt", "Emit MLIR after Loop Opts"),
               clEnumValN(EmitLLVM, "emit-llvm", "Emit LLVM IR"),
               clEnumValN(EmitBC, "emit-bc", "Emit LLVM BC"),
               clEnumValN(EmitASM, "emit-asm", "Emit ASM")),
    cl::init(EmitExe));
enum OptLevel {
  O0 = 0,
  O1,
  O2,
  O3,
};

cl::opt<OptLevel> OptimizationLevel(
    cl::desc("Choose optimization level:"), cl::cat(MLIRCGCategory),
    cl::values(clEnumVal(O0, "No optimization"),
               clEnumVal(O1, "Enable trivial optimizations"),
               clEnumVal(O2, "Enable default optimizations"),
               clEnumVal(O3, "Enable expensive optimizations")),
    cl::init(O1));

cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::cat(MLIRCGCategory),
                                    cl::value_desc("filename"), cl::init(""));

cl::opt<std::string> RuntimePath("L", cl::desc("Specify CIL runtime path"),
                                 cl::cat(MLIRCGCategory),
                                 cl::value_desc("<path-to-CIL-runtime>"),
                                 cl::init(""));

cl::opt<std::string> IncludePath("I", cl::desc("Specify Include path"),
                                 cl::cat(MLIRCGCategory),
                                 cl::value_desc("<include-dir-path>"),
                                 cl::Prefix, cl::init(""));

static cl::list<std::string>
    MacroNames("D", cl::desc("Name of the macro to be defined"),
               cl::value_desc("macro name"), cl::Prefix);

llvm::cl::opt<bool> StopAtCompile("c", llvm::cl::desc("stop at compilation"),
                                  cl::cat(MLIRCGCategory),
                                  llvm::cl::init(false));

llvm::cl::opt<bool> EnableDebug("g", llvm::cl::desc("Enable debugging symbols"),
                                cl::cat(MLIRCGCategory), llvm::cl::init(false));

llvm::cl::opt<bool> EnableCPP("cpp", llvm::cl::desc("Enable C++ 11"),
                              cl::cat(MLIRCGCategory), llvm::cl::init(false));

llvm::cl::opt<bool>
    EnableSTLOpt("optStl", llvm::cl::desc("Apply STL related optimizations"),
                 cl::cat(MLIRCGCategory), llvm::cl::init(false));

llvm::cl::opt<bool> PrintMLIRPasses("print-mlir",
                                    llvm::cl::desc("Print after all"),
                                    cl::cat(MLIRCGCategory),
                                    llvm::cl::init(false));

llvm::cl::opt<bool> EnableLNO("lno",
                              llvm::cl::desc("Enable Loop Nest Optimizations"),
                              cl::cat(MLIRCGCategory), llvm::cl::init(false));

llvm::cl::opt<bool> DumpVersion("v", llvm::cl::desc("Version check"),
                                cl::cat(MLIRCGCategory), llvm::cl::init(false));

llvm::cl::opt<bool> PrepareForLTO("flto", llvm::cl::desc("Prepare for LTO"),
                                  cl::cat(MLIRCGCategory),
                                  llvm::cl::init(false));

llvm::cl::opt<std::string> MArchName("march",
                                     cl::desc("Specify target architecture"),
                                     cl::cat(MLIRCGCategory),
                                     cl::value_desc("marchname"), cl::init(""));

static bool prepareLLVMTarget(std::unique_ptr<llvm::TargetMachine> &TM,
                              std::unique_ptr<llvm::Module> &llvmModule) {
  // Initialize targets.
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();

  llvmModule->setSourceFileName("");

  // set LLVM target triple.
  // Default to x86_64 for now.
  auto TargetTriple = sys::getDefaultTargetTriple();
  llvmModule->setTargetTriple(TargetTriple);

  std::string Error;
  std::string Triple = llvmModule->getTargetTriple();
  const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget) {
    llvm::errs() << "\n could not find target for triple " << Triple;
    return false;
  }

  llvm::Optional<CodeModel::Model> CM = llvm::CodeModel::Small;
  llvm::Optional<Reloc::Model> RM = llvm::Reloc::Static;
  auto OptLevel = CodeGenOpt::Default;
  switch (OptimizationLevel) {
  case OptLevel::O1:
    OptLevel = CodeGenOpt::Less;
    break;
  case OptLevel::O2:
  case OptLevel::O0:
    OptLevel = CodeGenOpt::Default;
    break;
  case OptLevel::O3:
    OptLevel = CodeGenOpt::Aggressive;
    break;
  };

  llvm::TargetOptions Options;
  Options.ThreadModel = llvm::ThreadModel::POSIX;
  Options.FloatABIType = llvm::FloatABI::Default;
  Options.AllowFPOpFusion = llvm::FPOpFusion::Standard;
  Options.UnsafeFPMath = true;

  const char *CPU;
  if (MArchName.empty())
    CPU = sys::getHostCPUName().data();
  else
    CPU = MArchName.c_str();
  StringMap<bool> HostFeatures;
  auto status = sys::getHostCPUFeatures(HostFeatures);
  SubtargetFeatures features;
  if (status) {
    for (auto &F : HostFeatures) {
      features.AddFeature(F.first(), F.second);
    }
  }
  TM.reset(TheTarget->createTargetMachine(Triple, CPU, features.getString(),
                                          Options, RM, CM, OptLevel));

  llvmModule->setDataLayout(TM->createDataLayout());
  return true;
}

namespace clang {
namespace mlir_codegen {
std::unique_ptr<ASTConsumer>
makeUniqueMLIRCodeGen(mlir::MLIRContext &context,
                      mlir::OwningModuleRef &theModule,
                      clang::ASTContext &astContext);
}
} // namespace clang

class BackendAction : public ASTFrontendAction {
public:
  BackendAction(mlir::MLIRContext &mlirContext,
                mlir::OwningModuleRef &TheModule)
      : mlirContext(mlirContext), TheModule(TheModule) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {

    return clang::mlir_codegen::makeUniqueMLIRCodeGen(mlirContext, TheModule,
                                                      CI.getASTContext());
  }

private:
  mlir::MLIRContext &mlirContext;
  mlir::OwningModuleRef &TheModule;
};

static void setInvocation(clang::CompilerInvocation *Invocation) {

  auto ppOpts = &Invocation->getPreprocessorOpts();
  for (auto macro : MacroNames) {
    ppOpts->addMacroDef(macro);
  }

  clang::LangOptions *langOpts = Invocation->getLangOpts();
  if (EnableCPP) {
    langOpts->C11 = 1;
    langOpts->C99 = 0;
    langOpts->CPlusPlus = 1;
    langOpts->CPlusPlus11 = 1;
    langOpts->ZVector = 1;
    // langOpts->CPlusPlus14 = 1;
    // langOpts->GNUMode = 1;
    langOpts->CXXExceptions = 1;
    Invocation->setLangDefaults(
        *langOpts, clang::Language::CXX,
        llvm::Triple(Invocation->getTargetOpts().Triple),
        *Invocation->PreprocessorOpts.get(), clang::LangStandard::lang_cxx11);
  } else {
    langOpts->GNUMode = 1;
    langOpts->CXXExceptions = 0;
    langOpts->RTTI = 0;
    langOpts->Bool = 1;
    langOpts->CPlusPlus = 0;
    langOpts->C99 = 1;
    Invocation->setLangDefaults(
        *langOpts, clang::Language::C,
        llvm::Triple(Invocation->getTargetOpts().Triple),
        *Invocation->PreprocessorOpts.get(), clang::LangStandard::lang_c99);
  }
}

class CMLToolChain : public ToolAction {
public:
  CMLToolChain(mlir::MLIRContext &mlirContext, mlir::OwningModuleRef &TheModule)
      : mlirContext(mlirContext), TheModule(TheModule) {}
  /// Invokes the compiler with a FrontendAction created by create().
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Create a compiler instance to handle the actual work.
    setInvocation(Invocation.get());
    CompilerInstance Compiler(std::move(PCHContainerOps));
    Compiler.setInvocation(std::move(Invocation));

    // The FrontendAction can have lifetime requirements for Compiler or its
    // members, and we need to ensure it's deleted earlier than Compiler. So we
    // pass it to an std::unique_ptr declared after the Compiler variable.
    std::unique_ptr<FrontendAction> ScopedToolAction(create());

    // Create the compiler's actual diagnostics engine.
    Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!Compiler.hasDiagnostics())
      return false;

    Compiler.createSourceManager(*Files);

    if (IncludePath != "") {
      auto &headerSearchOpts = Compiler.getHeaderSearchOpts();
      headerSearchOpts.AddPath(IncludePath, frontend::Angled, false, true);
      // headerSearchOpts.Verbose = true;
    }

    const bool Success = Compiler.ExecuteAction(*ScopedToolAction);

    Files->clearStatCache();
    return Success;
  }

  /// Returns a new clang::FrontendAction.
  virtual std::unique_ptr<FrontendAction> create() {
    return std::make_unique<BackendAction>(mlirContext, TheModule);
  }

private:
  mlir::MLIRContext &mlirContext;
  mlir::OwningModuleRef &TheModule;
};

class CustomASTDumpAction : public ASTFrontendAction {
public:
  CustomASTDumpAction() {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    std::error_code EC;
    setInvocation(&CI.getInvocation());
    CI.getFrontendOpts().ProgramAction = clang::frontend::ASTDump;
    CI.getFrontendOpts().ASTDumpDecls = 1;
    CI.getFrontendOpts().ASTDumpAll = 1;
    return clang::CreateASTDumper(nullptr, "", true, false, false,
                                  clang::ADOF_Default);
  }
};

static bool runLLVMPasses(std::unique_ptr<llvm::Module> &llvmModule,
                          std::unique_ptr<llvm::TargetMachine> &TM,
                          llvm::raw_fd_ostream &OS) {
  prepareLLVMTarget(TM, llvmModule);
  if (ActionOpt == EmitBC && OptimizationLevel == O0) {
    LLVM_DEBUG(llvm::dbgs() << "Emitting LLVM BC before optimizations\n");
    llvm::WriteBitcodeToFile(*llvmModule.get(), OS);
    OS.flush();
    return true;
  }

  if (ActionOpt == EmitLLVM && OptimizationLevel == O0) {
    LLVM_DEBUG(llvm::dbgs() << "Emitting LLVM IR\n");
    llvmModule->print(OS, nullptr);
    OS.flush();
    return true;
  }

  llvm::Triple TargetTriple(llvmModule->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TargetTriple));

  PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptimizationLevel;
  PMBuilder.SizeLevel = 0;
  PMBuilder.LoopVectorize = OptimizationLevel > 1;
  PMBuilder.SLPVectorize = OptimizationLevel > 1;
  PMBuilder.PrepareForLTO = PrepareForLTO;
  PMBuilder.DisableUnrollLoops = !(OptimizationLevel > 1);
  PMBuilder.Inliner = createFunctionInliningPass(PMBuilder.OptLevel,
                                                 PMBuilder.SizeLevel, false);

  legacy::FunctionPassManager FPM(llvmModule.get());
  FPM.add(new TargetLibraryInfoWrapperPass(*TLII));
  FPM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  legacy::PassManager MPM;
  MPM.add(new TargetLibraryInfoWrapperPass(*TLII));
  MPM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  PMBuilder.populateFunctionPassManager(FPM);
  PMBuilder.populateModulePassManager(MPM);

  // Run all the function passes.
  FPM.doInitialization();
  for (llvm::Function &F : *llvmModule)
    if (!F.isDeclaration())
      FPM.run(F);
  FPM.doFinalization();

  // Run all the module passes.
  MPM.run(*llvmModule.get());

  if (PrepareForLTO || ActionOpt == EmitBC) {
    LLVM_DEBUG(llvm::dbgs() << "Emitting LLVM BC after optimizations\n");
    llvm::WriteBitcodeToFile(*llvmModule, OS);
    OS.flush();
    return true;
  }

  if (ActionOpt == EmitLLVM) {
    LLVM_DEBUG(llvm::dbgs() << "Emitting LLVM IR after optimizations\n");
    llvmModule->print(OS, nullptr);
    OS.flush();
    return true;
  }
  return true;
}

// High level MLIR passes.
static void addCILDialectPasses(mlir::PassManager &mlirPM) {
  if (EnableCPP && EnableSTLOpt) {
    mlirPM.addPass(CIL::createVectorPushBackOptimizerPass());
    mlirPM.addPass(CIL::createVectorReserveOptimizerPass());
    mlirPM.addPass(mlir::createCanonicalizerPass());
  }
}

// High level MLIR lowering.
static void addLowerCILDialectPasses(mlir::PassManager &mlirPM) {
  if (EnableCPP) {
    mlirPM.addPass(CIL::createClassInheritanceLoweringPass());
    mlirPM.addPass(CIL::createConstructorLoweringPass());
    mlirPM.addPass(CIL::createClassLoweringPass());
  }
}

static void addLowerLevelMLIROptPasses(mlir::PassManager &mlirPM) {}

static void addLLVMLoweringPasses(mlir::PassManager &mlirPM) {
  mlirPM.addPass(CIL::createLoopStructureLoweringPass());
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(CIL::lowering::createCILToLLVMLoweringPass());
}

static bool runMLIRPasses(mlir::OwningModuleRef &theModule,
                          llvm::raw_fd_ostream &OS) {

  mlir::PassManager mlirPM(theModule->getContext());
  mlirPM.disableMultithreading();
  mlir::applyPassManagerCLOptions(mlirPM);

  // Enable print after all.
  if (PrintMLIRPasses) {
    // mlir::IRPrinterConfig config;
    // mlirPM.enableIRPrinting([](mlir::Pass *p) { return false; },
    //                         [](mlir::Pass *p) { return true; }, true,
    //                         llvm::errs());
  }

  switch (ActionOpt) {
  case EmitIR: {
    if (OptimizationLevel > 0) {
      addCILDialectPasses(mlirPM);
    }
    break;
  }
  case EmitMLIR:
  case EmitLoopOpt: {
    if (OptimizationLevel > 0 || EnableLNO) {
      addCILDialectPasses(mlirPM);
    }
    addLowerCILDialectPasses(mlirPM);
    if (OptimizationLevel > 0 || EnableLNO) {
      addLowerLevelMLIROptPasses(mlirPM);
    }
    break;
  }
  default: {
    if (OptimizationLevel > 0 || EnableLNO) {
      addCILDialectPasses(mlirPM);
    }
    addLowerCILDialectPasses(mlirPM);
    if (OptimizationLevel > 0 || EnableLNO) {
      addLowerLevelMLIROptPasses(mlirPM);
    }
    addLLVMLoweringPasses(mlirPM);
  }
  };

  auto result = mlirPM.run(theModule.get());
  if (failed(result)) {
    llvm::errs() << "Failed to run MLIR Pass manager\n";
    return false;
  }
  return true;
}

static void fixFlags(StringRef InputFile) {
  if (PrepareForLTO) {
    ActionOpt = EmitBC;
  }
  if (ActionOpt == EmitLoopOpt) {
    EnableLNO = true;
  }

  if (OutputFilename == "") {
    std::string extension = "";
    switch (ActionOpt) {
    case EmitRawAST:
    case EmitAST:
      extension = "ast.c";
      break;
    case EmitBC:
      if (PrepareForLTO) {
        extension = "o";
      } else {
        extension = "bc";
      }
      break;
    case EmitMLIR:
    case EmitIR:
    case EmitLoopOpt:
      extension = "mlir";
      break;
    case EmitLLVM:
      extension = "ll";
      break;
    case EmitASM:
      extension = "s";
      break;
    case EmitExe:
      if (StopAtCompile) {
        extension = "o";
      } else {
        OutputFilename = "a.out";
      }
      break;
    default:
      llvm_unreachable("Unhandled action type");
    };
    if (ActionOpt != EmitExe || (ActionOpt == EmitExe && StopAtCompile)) {
      // Replace the existing extension in the input file to the new one.
      assert(!extension.empty());
      OutputFilename = InputFile.str();
      llvm::SmallString<128> outputFile(OutputFilename);
      llvm::sys::path::replace_extension(outputFile, extension);
      OutputFilename = outputFile.str().str();
    }
  }
}

static bool compileFile(ClangTool &Tool) {

  auto sourceFiles = Tool.getSourcePaths();
  assert(sourceFiles.size() == 1 && "only one file for now");
  auto InputFile = sourceFiles.front();
  fixFlags(InputFile);

  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-stdlib=libc++", ArgumentInsertPosition::BEGIN));

  LLVM_DEBUG(llvm::dbgs() << "Started parsing input file " << InputFile
                          << "\n");

  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFilename, EC, llvm::sys::fs::F_None);

  if (ActionOpt == EmitAST) {
    LLVM_DEBUG(llvm::dbgs() << "Emitting the AST after SEMA\n");
    // Simple code to dump parse tree.
    auto file = std::make_unique<raw_fd_ostream>(OutputFilename, EC,
                                                 llvm::sys::fs::F_None);

    // Tool.appendArgumentsAdjuster(
    // getInsertArgumentAdjuster("-stdlib=libc++",
    // ArgumentInsertPosition::BEGIN));

    auto result =
        Tool.run(newFrontendActionFactory<CustomASTDumpAction>().get());
    return result == 0;
  }

  mlir::registerDialect<CIL::CILDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext mlirContext;
  mlir::OwningModuleRef theModule;
  mlir::registerPassManagerCLOptions();

  theModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext));
  auto newToolChain = new CMLToolChain(mlirContext, theModule);
  auto result = Tool.run(newToolChain);
  // MLIR Codegen.
  if (result != 0) {
    llvm::errs() << "\n Error during MLIR emission";
    return false;
  }

  if (failed(mlir::verify(theModule.get()))) {
    theModule->dump();
    theModule->emitError("module verification error");
    return false;
  }

  // Run transforms on MLIR.
  if (!runMLIRPasses(theModule, OS)) {
    llvm::errs() << "Failed to emit MLIR\n";
    return false;
  }

  bool emitMLIRCode = false;
  switch (ActionOpt) {
  case EmitIR:
  case EmitMLIR:
  case EmitLoopOpt:
    emitMLIRCode = true;
    break;
  default:
    emitMLIRCode = false;
  };

  if (emitMLIRCode) {
    theModule->print(OS);
    OS.flush();
    return true;
  }

  // Emit LLVM IR from MLIR.
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule.get());
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return false;
  }

  // Prepare and run LLVM IR passes.
  std::unique_ptr<llvm::TargetMachine> TM;
  if (!runLLVMPasses(llvmModule, TM, OS)) {
    llvm::errs() << "\n Failed to run LLVM Passes";
    return false;
  }

  // runLLVMPasses already dumped ir
  if (ActionOpt == EmitBC || ActionOpt == EmitLLVM || PrepareForLTO)
    return true;

  // Create CodeGen Passes.
  legacy::PassManager CGPasses;
  CGPasses.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  if (ActionOpt == EmitASM) {
    if (TM->addPassesToEmitFile(CGPasses, OS, nullptr,
                                llvm::CGFT_AssemblyFile)) {
      llvm::errs() << "\n Failed to emit Assembly file.";
      return false;
    }

    // Run all codegen passes.
    CGPasses.run(*llvmModule.get());
    LLVM_DEBUG(llvm::dbgs() << "Emitting ASM file\n");
    return true;
  }

  // Generate binary action. Emit object file first and then create exe.
  assert(ActionOpt == EmitExe);
  LLVMTargetMachine &LTM = static_cast<LLVMTargetMachine &>(*TM);
  llvm::MachineModuleInfo MMI(&LTM);
  auto MCContext = &MMI.getContext();

  std::string objFile = "";

  // Create temporary object file.
  auto tempFile = llvm::sys::path::filename(StringRef(InputFile));
  std::string tempFilename = "/tmp/" + tempFile.str();
  auto TmpFile = llvm::sys::fs::TempFile::create(tempFilename + "-%%%%%.o");
  if (!TmpFile) {
    llvm::errs() << "\n Failed to create temporary file!";
    return false;
  }

  objFile = StopAtCompile ? OutputFilename : TmpFile.get().TmpName;
  llvm::raw_fd_ostream TOS(objFile, EC, llvm::sys::fs::F_None);

  LLVM_DEBUG(llvm::dbgs() << "Emitting Temp file " << objFile);
  if (TM->addPassesToEmitMC(CGPasses, MCContext, TOS, false)) {
    llvm::errs() << "\n Failed to generate object code";
    return false;
  }

  // Run all codegen passes.
  CGPasses.run(*llvmModule.get());

  if (StopAtCompile)
    return true;
  // Create ld command

  // FIXME: Expects clang binary for linking.
  StringRef ldCommand = getenv("CLANG_BINARY");
  if (ldCommand.empty()) {
    llvm::errs() << "\n CLANG_BINARY env variable not set!";
    return false;
  }

  if (RuntimePath.size() > 0)
    RuntimePath = "-L" + RuntimePath;
  const char *args[10] = {ldCommand.data(), "-stdlib=libc++", objFile.c_str(),
                          "-o", OutputFilename.c_str(), "-lm",
                          /*"-lomp",*/
                          "-lstdc++", RuntimePath.c_str(), NULL};

  std::string errorStr;
  bool ExecFailed = false;
  std::vector<llvm::Optional<StringRef>> Redirects;
  Redirects = {llvm::NoneType::None, llvm::NoneType::None,
               llvm::NoneType::None};

  llvm::SmallVector<llvm::StringRef, 16> argsArr(args, args + 9);
  llvm::sys::ExecuteAndWait(ldCommand, argsArr, llvm::None, Redirects, 0, 0,
                            &errorStr, &ExecFailed);
  if (ExecFailed) {
    llvm::errs() << "\n ld tool execution failed : " << errorStr;
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "Emitting binary file" << OutputFilename);
  // Delete the temp file created.
  if (auto E = TmpFile->discard()) {
    return false;
  }
  return true;
}

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MLIRCGCategory);
  // cl::ParseCommandLineOptions(argc, argv);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());
  if (!compileFile(Tool)) {
    return 1;
  }
  return 0;
}

