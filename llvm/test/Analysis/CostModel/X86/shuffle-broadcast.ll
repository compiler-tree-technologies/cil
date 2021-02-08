; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+sse2 | FileCheck %s -check-prefixes=CHECK,SSE,SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+ssse3 | FileCheck %s -check-prefixes=CHECK,SSE,SSSE3
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+sse4.2 | FileCheck %s -check-prefixes=CHECK,SSE,SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+avx | FileCheck %s -check-prefixes=CHECK,AVX,AVX1
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+avx2 | FileCheck %s -check-prefixes=CHECK,AVX,AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+avx512f | FileCheck %s --check-prefixes=CHECK,AVX512,AVX512F
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+avx512f,+avx512bw | FileCheck %s --check-prefixes=CHECK,AVX512
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mattr=+avx512f,+avx512bw,+avx512vbmi | FileCheck %s --check-prefixes=CHECK,AVX512
;
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mcpu=slm | FileCheck %s --check-prefixes=CHECK,SSE,SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mcpu=goldmont | FileCheck %s --check-prefixes=CHECK,SSE,SSE42
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -cost-model -analyze -mcpu=btver2 | FileCheck %s --check-prefixes=CHECK,AVX,BTVER2

;
; Verify the cost model for broadcast shuffles.
;

define void @test_vXf64(<2 x double> %src128, <4 x double> %src256, <8 x double> %src512) {
; SSE-LABEL: 'test_vXf64'
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_vXf64'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_vXf64'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_vXf64'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_vXf64'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V128 = shufflevector <2 x double> %src128, <2 x double> undef, <2 x i32> zeroinitializer
  %V256 = shufflevector <4 x double> %src256, <4 x double> undef, <4 x i32> zeroinitializer
  %V512 = shufflevector <8 x double> %src512, <8 x double> undef, <8 x i32> zeroinitializer
  ret void
}

define void @test_vXi64(<2 x i64> %src128, <4 x i64> %src256, <8 x i64> %src512) {
; SSE-LABEL: 'test_vXi64'
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_vXi64'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_vXi64'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_vXi64'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_vXi64'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V128 = shufflevector <2 x i64> %src128, <2 x i64> undef, <2 x i32> zeroinitializer
  %V256 = shufflevector <4 x i64> %src256, <4 x i64> undef, <4 x i32> zeroinitializer
  %V512 = shufflevector <8 x i64> %src512, <8 x i64> undef, <8 x i32> zeroinitializer
  ret void
}

define void @test_vXf32(<2 x float> %src64, <4 x float> %src128, <8 x float> %src256, <16 x float> %src512) {
; SSE-LABEL: 'test_vXf32'
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_vXf32'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_vXf32'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_vXf32'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_vXf32'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V64 = shufflevector <2 x float> %src64, <2 x float> undef, <2 x i32> zeroinitializer
  %V128 = shufflevector <4 x float> %src128, <4 x float> undef, <4 x i32> zeroinitializer
  %V256 = shufflevector <8 x float> %src256, <8 x float> undef, <8 x i32> zeroinitializer
  %V512 = shufflevector <16 x float> %src512, <16 x float> undef, <16 x i32> zeroinitializer
  ret void
}

define void @test_vXi32(<2 x i32> %src64, <4 x i32> %src128, <8 x i32> %src256, <16 x i32> %src512) {
; SSE-LABEL: 'test_vXi32'
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer
; SSE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_vXi32'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_vXi32'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_vXi32'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_vXi32'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V64 = shufflevector <2 x i32> %src64, <2 x i32> undef, <2 x i32> zeroinitializer
  %V128 = shufflevector <4 x i32> %src128, <4 x i32> undef, <4 x i32> zeroinitializer
  %V256 = shufflevector <8 x i32> %src256, <8 x i32> undef, <8 x i32> zeroinitializer
  %V512 = shufflevector <16 x i32> %src512, <16 x i32> undef, <16 x i32> zeroinitializer
  ret void
}

define void @test_vXi16(<8 x i16> %src128, <16 x i16> %src256, <32 x i16> %src512) {
; SSE2-LABEL: 'test_vXi16'
; SSE2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; SSE2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; SSE2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; SSE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; SSSE3-LABEL: 'test_vXi16'
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; SSE42-LABEL: 'test_vXi16'
; SSE42-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; SSE42-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; SSE42-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; SSE42-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_vXi16'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_vXi16'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_vXi16'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_vXi16'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V128 = shufflevector <8 x i16> %src128, <8 x i16> undef, <8 x i32> zeroinitializer
  %V256 = shufflevector <16 x i16> %src256, <16 x i16> undef, <16 x i32> zeroinitializer
  %V512 = shufflevector <32 x i16> %src512, <32 x i16> undef, <32 x i32> zeroinitializer
  ret void
}

define void @test_vXi8(<16 x i8> %src128, <32 x i8> %src256, <64 x i8> %src512) {
; SSE2-LABEL: 'test_vXi8'
; SSE2-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; SSE2-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; SSE2-NEXT:  Cost Model: Found an estimated cost of 3 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; SSE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; SSSE3-LABEL: 'test_vXi8'
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; SSSE3-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; SSE42-LABEL: 'test_vXi8'
; SSE42-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; SSE42-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; SSE42-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; SSE42-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_vXi8'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_vXi8'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_vXi8'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_vXi8'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V128 = shufflevector <16 x i8> %src128, <16 x i8> undef, <16 x i32> zeroinitializer
  %V256 = shufflevector <32 x i8> %src256, <32 x i8> undef, <32 x i32> zeroinitializer
  %V512 = shufflevector <64 x i8> %src512, <64 x i8> undef, <64 x i32> zeroinitializer
  ret void
}

;
; Tests the cost model for broadcast shuffles of second operand.
;

define void @test_upper_vXf32(<2 x float> %a64, <2 x float> %b64, <4 x float> %a128, <4 x float> %b128, <8 x float> %a256, <8 x float> %b256, <16 x float> %a512, <16 x float> %b512) {
; SSE-LABEL: 'test_upper_vXf32'
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %a64, <2 x float> %b64, <2 x i32> <i32 2, i32 2>
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %a128, <4 x float> %b128, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x float> %a256, <8 x float> %b256, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; SSE-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x float> %a512, <16 x float> %b512, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; SSE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX1-LABEL: 'test_upper_vXf32'
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %a64, <2 x float> %b64, <2 x i32> <i32 2, i32 2>
; AVX1-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %a128, <4 x float> %b128, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <8 x float> %a256, <8 x float> %b256, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; AVX1-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <16 x float> %a512, <16 x float> %b512, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; AVX1-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX2-LABEL: 'test_upper_vXf32'
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %a64, <2 x float> %b64, <2 x i32> <i32 2, i32 2>
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %a128, <4 x float> %b128, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x float> %a256, <8 x float> %b256, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; AVX2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x float> %a512, <16 x float> %b512, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; AVX2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; AVX512-LABEL: 'test_upper_vXf32'
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %a64, <2 x float> %b64, <2 x i32> <i32 2, i32 2>
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %a128, <4 x float> %b128, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V256 = shufflevector <8 x float> %a256, <8 x float> %b256, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; AVX512-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V512 = shufflevector <16 x float> %a512, <16 x float> %b512, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; AVX512-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
; BTVER2-LABEL: 'test_upper_vXf32'
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V64 = shufflevector <2 x float> %a64, <2 x float> %b64, <2 x i32> <i32 2, i32 2>
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 1 for instruction: %V128 = shufflevector <4 x float> %a128, <4 x float> %b128, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V256 = shufflevector <8 x float> %a256, <8 x float> %b256, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %V512 = shufflevector <16 x float> %a512, <16 x float> %b512, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
; BTVER2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %V64 = shufflevector <2 x float> %a64, <2 x float> %b64, <2 x i32> <i32 2, i32 2>
  %V128 = shufflevector <4 x float> %a128, <4 x float> %b128, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %V256 = shufflevector <8 x float> %a256, <8 x float> %b256, <8 x i32> <i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>
  %V512 = shufflevector <16 x float> %a512, <16 x float> %b512, <16 x i32> <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  ret void
}