; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,HAWAII %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,FIJI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

define void @local_store_i56(i56 addrspace(3)* %ptr, i56 %arg) #0 {
; CIVI-LABEL: local_store_i56:
; CIVI:       ; %bb.0:
; CIVI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIVI-NEXT:    s_mov_b32 m0, -1
; CIVI-NEXT:    v_lshrrev_b32_e32 v3, 16, v2
; CIVI-NEXT:    ds_write_b16 v0, v2 offset:4
; CIVI-NEXT:    ds_write_b32 v0, v1
; CIVI-NEXT:    ds_write_b8 v0, v3 offset:6
; CIVI-NEXT:    s_waitcnt lgkmcnt(0)
; CIVI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: local_store_i56:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_write_b8_d16_hi v0, v2 offset:6
; GFX9-NEXT:    ds_write_b16 v0, v2 offset:4
; GFX9-NEXT:    ds_write_b32 v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  store i56 %arg, i56 addrspace(3)* %ptr, align 8
  ret void
}

define amdgpu_kernel void @local_store_i55(i55 addrspace(3)* %ptr, i55 %arg) #0 {
; HAWAII-LABEL: local_store_i55:
; HAWAII:       ; %bb.0:
; HAWAII-NEXT:    s_add_u32 s0, s4, 14
; HAWAII-NEXT:    s_addc_u32 s1, s5, 0
; HAWAII-NEXT:    v_mov_b32_e32 v0, s0
; HAWAII-NEXT:    v_mov_b32_e32 v1, s1
; HAWAII-NEXT:    flat_load_ubyte v0, v[0:1]
; HAWAII-NEXT:    s_load_dword s0, s[4:5], 0x0
; HAWAII-NEXT:    s_load_dword s1, s[4:5], 0x2
; HAWAII-NEXT:    s_load_dword s2, s[4:5], 0x3
; HAWAII-NEXT:    s_mov_b32 m0, -1
; HAWAII-NEXT:    s_waitcnt lgkmcnt(0)
; HAWAII-NEXT:    v_mov_b32_e32 v1, s0
; HAWAII-NEXT:    v_mov_b32_e32 v3, s1
; HAWAII-NEXT:    v_mov_b32_e32 v2, s2
; HAWAII-NEXT:    ds_write_b16 v1, v2 offset:4
; HAWAII-NEXT:    s_waitcnt vmcnt(0)
; HAWAII-NEXT:    v_and_b32_e32 v0, 0x7f, v0
; HAWAII-NEXT:    ds_write_b8 v1, v0 offset:6
; HAWAII-NEXT:    ds_write_b32 v1, v3
; HAWAII-NEXT:    s_endpgm
;
; FIJI-LABEL: local_store_i55:
; FIJI:       ; %bb.0:
; FIJI-NEXT:    s_load_dword s0, s[4:5], 0x0
; FIJI-NEXT:    s_load_dword s1, s[4:5], 0x8
; FIJI-NEXT:    s_load_dword s2, s[4:5], 0xc
; FIJI-NEXT:    s_mov_b32 m0, -1
; FIJI-NEXT:    s_waitcnt lgkmcnt(0)
; FIJI-NEXT:    v_mov_b32_e32 v2, s0
; FIJI-NEXT:    v_mov_b32_e32 v3, s1
; FIJI-NEXT:    s_and_b32 s3, s2, 0xffff
; FIJI-NEXT:    s_add_u32 s0, s4, 14
; FIJI-NEXT:    s_addc_u32 s1, s5, 0
; FIJI-NEXT:    v_mov_b32_e32 v0, s0
; FIJI-NEXT:    v_mov_b32_e32 v1, s1
; FIJI-NEXT:    flat_load_ubyte v0, v[0:1]
; FIJI-NEXT:    v_mov_b32_e32 v1, s2
; FIJI-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; FIJI-NEXT:    v_lshlrev_b32_e32 v0, 16, v0
; FIJI-NEXT:    v_or_b32_e32 v0, s3, v0
; FIJI-NEXT:    v_bfe_u32 v0, v0, 16, 7
; FIJI-NEXT:    ds_write_b16 v2, v1 offset:4
; FIJI-NEXT:    ds_write_b8 v2, v0 offset:6
; FIJI-NEXT:    ds_write_b32 v2, v3
; FIJI-NEXT:    s_endpgm
;
; GFX9-LABEL: local_store_i55:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    v_mov_b32_e32 v1, s5
; GFX9-NEXT:    v_mov_b32_e32 v2, 0
; GFX9-NEXT:    global_load_ubyte_d16_hi v2, v[0:1], off offset:14
; GFX9-NEXT:    s_load_dword s0, s[4:5], 0x0
; GFX9-NEXT:    s_load_dword s1, s[4:5], 0x8
; GFX9-NEXT:    s_load_dword s2, s[4:5], 0xc
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v3, s1
; GFX9-NEXT:    v_mov_b32_e32 v1, s2
; GFX9-NEXT:    s_and_b32 s3, s2, 0xffff
; GFX9-NEXT:    ds_write_b16 v0, v1 offset:4
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_or_b32_e32 v1, s3, v2
; GFX9-NEXT:    v_and_b32_e32 v1, 0x7fffff, v1
; GFX9-NEXT:    ds_write_b8_d16_hi v0, v1 offset:6
; GFX9-NEXT:    ds_write_b32 v0, v3
; GFX9-NEXT:    s_endpgm
  store i55 %arg, i55 addrspace(3)* %ptr, align 8
  ret void
}

define amdgpu_kernel void @local_store_i48(i48 addrspace(3)* %ptr, i48 %arg) #0 {
; HAWAII-LABEL: local_store_i48:
; HAWAII:       ; %bb.0:
; HAWAII-NEXT:    s_load_dword s0, s[4:5], 0x0
; HAWAII-NEXT:    s_load_dword s1, s[4:5], 0x2
; HAWAII-NEXT:    s_load_dword s2, s[4:5], 0x3
; HAWAII-NEXT:    s_mov_b32 m0, -1
; HAWAII-NEXT:    s_waitcnt lgkmcnt(0)
; HAWAII-NEXT:    v_mov_b32_e32 v0, s0
; HAWAII-NEXT:    v_mov_b32_e32 v1, s1
; HAWAII-NEXT:    v_mov_b32_e32 v2, s2
; HAWAII-NEXT:    ds_write_b16 v0, v2 offset:4
; HAWAII-NEXT:    ds_write_b32 v0, v1
; HAWAII-NEXT:    s_endpgm
;
; FIJI-LABEL: local_store_i48:
; FIJI:       ; %bb.0:
; FIJI-NEXT:    s_load_dword s0, s[4:5], 0x0
; FIJI-NEXT:    s_load_dword s1, s[4:5], 0x8
; FIJI-NEXT:    s_load_dword s2, s[4:5], 0xc
; FIJI-NEXT:    s_mov_b32 m0, -1
; FIJI-NEXT:    s_waitcnt lgkmcnt(0)
; FIJI-NEXT:    v_mov_b32_e32 v0, s0
; FIJI-NEXT:    v_mov_b32_e32 v1, s1
; FIJI-NEXT:    v_mov_b32_e32 v2, s2
; FIJI-NEXT:    ds_write_b16 v0, v2 offset:4
; FIJI-NEXT:    ds_write_b32 v0, v1
; FIJI-NEXT:    s_endpgm
;
; GFX9-LABEL: local_store_i48:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dword s0, s[4:5], 0x0
; GFX9-NEXT:    s_load_dword s1, s[4:5], 0x8
; GFX9-NEXT:    s_load_dword s2, s[4:5], 0xc
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s1
; GFX9-NEXT:    v_mov_b32_e32 v2, s2
; GFX9-NEXT:    ds_write_b16 v0, v2 offset:4
; GFX9-NEXT:    ds_write_b32 v0, v1
; GFX9-NEXT:    s_endpgm
  store i48 %arg, i48 addrspace(3)* %ptr, align 8
  ret void
}

define amdgpu_kernel void @local_store_i65(i65 addrspace(3)* %ptr, i65 %arg) #0 {
; HAWAII-LABEL: local_store_i65:
; HAWAII:       ; %bb.0:
; HAWAII-NEXT:    s_load_dword s2, s[4:5], 0x0
; HAWAII-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x2
; HAWAII-NEXT:    s_load_dword s3, s[4:5], 0x4
; HAWAII-NEXT:    s_mov_b32 m0, -1
; HAWAII-NEXT:    s_waitcnt lgkmcnt(0)
; HAWAII-NEXT:    v_mov_b32_e32 v2, s2
; HAWAII-NEXT:    v_mov_b32_e32 v0, s0
; HAWAII-NEXT:    v_mov_b32_e32 v1, s1
; HAWAII-NEXT:    s_and_b32 s0, s3, 1
; HAWAII-NEXT:    v_mov_b32_e32 v3, s0
; HAWAII-NEXT:    ds_write_b8 v2, v3 offset:8
; HAWAII-NEXT:    ds_write_b64 v2, v[0:1]
; HAWAII-NEXT:    s_endpgm
;
; FIJI-LABEL: local_store_i65:
; FIJI:       ; %bb.0:
; FIJI-NEXT:    s_load_dword s2, s[4:5], 0x0
; FIJI-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x8
; FIJI-NEXT:    s_load_dword s3, s[4:5], 0x10
; FIJI-NEXT:    s_mov_b32 m0, -1
; FIJI-NEXT:    s_waitcnt lgkmcnt(0)
; FIJI-NEXT:    v_mov_b32_e32 v2, s2
; FIJI-NEXT:    v_mov_b32_e32 v0, s0
; FIJI-NEXT:    v_mov_b32_e32 v1, s1
; FIJI-NEXT:    s_and_b32 s0, s3, 1
; FIJI-NEXT:    v_mov_b32_e32 v3, s0
; FIJI-NEXT:    ds_write_b8 v2, v3 offset:8
; FIJI-NEXT:    ds_write_b64 v2, v[0:1]
; FIJI-NEXT:    s_endpgm
;
; GFX9-LABEL: local_store_i65:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dword s2, s[4:5], 0x0
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[4:5], 0x8
; GFX9-NEXT:    s_load_dword s3, s[4:5], 0x10
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v2, s2
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s1
; GFX9-NEXT:    s_and_b32 s0, s3, 1
; GFX9-NEXT:    v_mov_b32_e32 v3, s0
; GFX9-NEXT:    ds_write_b8 v2, v3 offset:8
; GFX9-NEXT:    ds_write_b64 v2, v[0:1]
; GFX9-NEXT:    s_endpgm
  store i65 %arg, i65 addrspace(3)* %ptr, align 8
  ret void
}

define void @local_store_i13(i13 addrspace(3)* %ptr, i13 %arg) #0 {
; CIVI-LABEL: local_store_i13:
; CIVI:       ; %bb.0:
; CIVI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIVI-NEXT:    v_and_b32_e32 v1, 0x1fff, v1
; CIVI-NEXT:    s_mov_b32 m0, -1
; CIVI-NEXT:    ds_write_b16 v0, v1
; CIVI-NEXT:    s_waitcnt lgkmcnt(0)
; CIVI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: local_store_i13:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_and_b32_e32 v1, 0x1fff, v1
; GFX9-NEXT:    ds_write_b16 v0, v1
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  store i13 %arg, i13 addrspace(3)* %ptr, align 8
  ret void
}

define void @local_store_i17(i17 addrspace(3)* %ptr, i17 %arg) #0 {
; CIVI-LABEL: local_store_i17:
; CIVI:       ; %bb.0:
; CIVI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIVI-NEXT:    s_mov_b32 m0, -1
; CIVI-NEXT:    v_bfe_u32 v2, v1, 16, 1
; CIVI-NEXT:    ds_write_b16 v0, v1
; CIVI-NEXT:    ds_write_b8 v0, v2 offset:2
; CIVI-NEXT:    s_waitcnt lgkmcnt(0)
; CIVI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: local_store_i17:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    ds_write_b16 v0, v1
; GFX9-NEXT:    v_and_b32_e32 v1, 0x1ffff, v1
; GFX9-NEXT:    ds_write_b8_d16_hi v0, v1 offset:2
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  store i17 %arg, i17 addrspace(3)* %ptr, align 8
  ret void
}

attributes #0 = { nounwind }