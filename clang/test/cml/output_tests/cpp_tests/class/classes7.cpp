// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class C {
  int c;

public:
  void set_c(int val) { c = val; }
  int get_c() { return c; }
};

class B {
public:
  C o3;
};

class A {
public:
  B o2;
  int get_c() { return o2.o3.get_c(); };
  void set_c(int val) { o2.o3.set_c(val); };
};

int main() {

  A o;
  o.set_c(11);
  printf("%d\n", o.get_c()); // CHECK: 11
  return 0;
}
