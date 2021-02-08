// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
public:
  class B {
  public:
    B() { printf("\n this needs to be called\n"); } // CHECK: this needs to be called
    int b;
  };
};

static A::B stat_init;

int main() {
  printf("\n inside main\n"); // CHECK: inside main 
  return 0;
}
