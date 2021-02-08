// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class C {
  int c;

public:
  void set_c(int val) { c = val; }
  int get_c() { return c; }
};

class B {
  C *o3;

public:
  void set_C(C *val) { o3 = val; }
  C *get_C() { return o3; }
};

class A {
public:
  B *o2;
  A(B *o2) : o2(o2) {}
  int get_c() { return o2->get_C()->get_c(); };
};

int main() {
  C o3;
  o3.set_c(11);
  B o2;
  o2.set_C(&o3);
  A o(&o2);
  printf("%d\n", o.get_c()); // CHECK: 11
  return 0;
}
