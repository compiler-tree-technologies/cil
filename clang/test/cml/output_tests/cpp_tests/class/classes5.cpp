// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class B {
  int b;

public:
  void set_b(int val) { b = val; }
  int get_b() { return b; }
};

class C {
  int c;

public:
  void set_c(int val) { c = val; }
  int get_c() { return c; }
};

class A {
public:
  B *o2;
  C *o3;
  A(B *o2, C *o3) : o2(o2), o3(o3) {}
};

int main() {
  B o2;
  C o3;
  A o(&o2, &o3);
  o.o2->set_b(10);
  o.o3->set_c(11);
  printf("%d %d\n", o2.get_b(), o3.get_c()); // CHECK: 10 11
  return 0;
}
