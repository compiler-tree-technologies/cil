// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
struct st {
  int a;
  int b;
};


int main() {

  struct st A[2];
  A[0].b = 10;
  A[0].a = 20;
  A[1].b = 30;
  A[1].a = 40;
  printf ("%d %d\n", A[1].a, A[1].b);
  printf ("%d %d\n", A[0].a, A[0].b);
  return 0;
}
// CHECK: 40 30
// CHECK: 20 10
