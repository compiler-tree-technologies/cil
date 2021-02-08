; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py
; RUN: opt -S -analyze -scalar-evolution < %s | FileCheck %s

define void @umin_unsigned_check(i64 %n) {
; CHECK-LABEL: 'umin_unsigned_check'
; CHECK-NEXT:  Classifying expressions for: @umin_unsigned_check
; CHECK-NEXT:    %min.n = select i1 %min.cmp, i64 4096, i64 %n
; CHECK-NEXT:    --> (4096 umin %n) U: [0,4097) S: [0,4097)
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    --> {0,+,1}<%loop> U: [0,4098) S: [0,4098) Exits: (1 + (4096 umin %n))<nuw><nsw> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %iv.next = add i64 %iv, 1
; CHECK-NEXT:    --> {1,+,1}<%loop> U: [1,4099) S: [1,4099) Exits: (2 + (4096 umin %n)) LoopDispositions: { %loop: Computable }
; CHECK-NEXT:  Determining loop execution counts for: @umin_unsigned_check
; CHECK-NEXT:  Loop %loop: backedge-taken count is (1 + (4096 umin %n))<nuw><nsw>
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 4097
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (1 + (4096 umin %n))<nuw><nsw>
; CHECK-NEXT:   Predicates:
; CHECK:       Loop %loop: Trip multiple is 1
;
entry:
  %min.cmp = icmp ult i64 4096, %n
  %min.n = select i1 %min.cmp, i64 4096, i64 %n
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 1
  %exit = icmp ugt i64 %iv, %min.n
  br i1 %exit, label %loop_exit, label %loop

loop_exit:
  ret void
}

define void @umin_signed_check(i64 %n) {
; CHECK-LABEL: 'umin_signed_check'
; CHECK-NEXT:  Classifying expressions for: @umin_signed_check
; CHECK-NEXT:    %min.n = select i1 %min.cmp, i64 4096, i64 %n
; CHECK-NEXT:    --> (4096 umin %n) U: [0,4097) S: [0,4097)
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    --> {0,+,1}<%loop> U: [0,4098) S: [0,4098) Exits: (1 + (4096 umin %n))<nuw><nsw> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %iv.next = add i64 %iv, 1
; CHECK-NEXT:    --> {1,+,1}<%loop> U: [1,4099) S: [1,4099) Exits: (2 + (4096 umin %n)) LoopDispositions: { %loop: Computable }
; CHECK-NEXT:  Determining loop execution counts for: @umin_signed_check
; CHECK-NEXT:  Loop %loop: backedge-taken count is (1 + (4096 umin %n))<nuw><nsw>
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 4097
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (1 + (4096 umin %n))<nuw><nsw>
; CHECK-NEXT:   Predicates:
; CHECK:       Loop %loop: Trip multiple is 1
;
entry:
  %min.cmp = icmp ult i64 4096, %n
  %min.n = select i1 %min.cmp, i64 4096, i64 %n
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 1
  %exit = icmp sgt i64 %iv, %min.n
  br i1 %exit, label %loop_exit, label %loop

loop_exit:
  ret void
}

define void @smin_signed_check(i64 %n) {
; CHECK-LABEL: 'smin_signed_check'
; CHECK-NEXT:  Classifying expressions for: @smin_signed_check
; CHECK-NEXT:    %min.n = select i1 %min.cmp, i64 4096, i64 %n
; CHECK-NEXT:    --> (4096 smin %n) U: [-9223372036854775808,4097) S: [-9223372036854775808,4097)
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    --> {0,+,1}<%loop> U: [0,4098) S: [0,4098) Exits: (0 smax (1 + (4096 smin %n))<nsw>) LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %iv.next = add i64 %iv, 1
; CHECK-NEXT:    --> {1,+,1}<%loop> U: [1,4099) S: [1,4099) Exits: (1 + (0 smax (1 + (4096 smin %n))<nsw>))<nuw><nsw> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:  Determining loop execution counts for: @smin_signed_check
; CHECK-NEXT:  Loop %loop: backedge-taken count is (0 smax (1 + (4096 smin %n))<nsw>)
; CHECK-NEXT:  Loop %loop: max backedge-taken count is 4097
; CHECK-NEXT:  Loop %loop: Predicated backedge-taken count is (0 smax (1 + (4096 smin %n))<nsw>)
; CHECK-NEXT:   Predicates:
; CHECK:       Loop %loop: Trip multiple is 1
;
entry:
  %min.cmp = icmp slt i64 4096, %n
  %min.n = select i1 %min.cmp, i64 4096, i64 %n
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 1
  %exit = icmp sgt i64 %iv, %min.n
  br i1 %exit, label %loop_exit, label %loop

loop_exit:
  ret void
}

define void @smin_unsigned_check(i64 %n) {
; CHECK-LABEL: 'smin_unsigned_check'
; CHECK-NEXT:  Classifying expressions for: @smin_unsigned_check
; CHECK-NEXT:    %min.n = select i1 %min.cmp, i64 4096, i64 %n
; CHECK-NEXT:    --> (4096 smin %n) U: [-9223372036854775808,4097) S: [-9223372036854775808,4097)
; CHECK-NEXT:    %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    --> {0,+,1}<%loop> U: full-set S: full-set Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:    %iv.next = add i64 %iv, 1
; CHECK-NEXT:    --> {1,+,1}<%loop> U: full-set S: full-set Exits: <<Unknown>> LoopDispositions: { %loop: Computable }
; CHECK-NEXT:  Determining loop execution counts for: @smin_unsigned_check
; CHECK-NEXT:  Loop %loop: Unpredictable backedge-taken count.
; CHECK-NEXT:  Loop %loop: Unpredictable max backedge-taken count.
; CHECK-NEXT:  Loop %loop: Unpredictable predicated backedge-taken count.
;
entry:
  %min.cmp = icmp slt i64 4096, %n
  %min.n = select i1 %min.cmp, i64 4096, i64 %n
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 1
  %exit = icmp ugt i64 %iv, %min.n
  br i1 %exit, label %loop_exit, label %loop

loop_exit:
  ret void
}
