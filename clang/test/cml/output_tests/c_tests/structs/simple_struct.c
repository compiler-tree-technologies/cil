// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
struct st {
  int a;
  int b;
};


int main() {

  struct st A;
  A.b = 10;
  A.a = 20;
  printf ("%d %d\n", A.a, A.b);
  return 0;
}
// CHECK: 20 10
