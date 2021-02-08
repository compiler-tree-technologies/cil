// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
#include <stdlib.h>
struct st {
  int a;
  int b;
};

int main() {

  struct st *A;
  unsigned long size = 8;
  A = malloc(size);
  A->b = 10;
  A->a = 20;
  // CHECK:  20 10
  printf("%d %d\n", A->a, A->b);
  free(A);
  return 0;
}
