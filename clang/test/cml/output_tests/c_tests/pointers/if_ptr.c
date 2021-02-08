// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
#include <stdlib.h>

int main() {
  int *p = NULL;
  p = (int *) malloc(sizeof(int));
  if (p)
    free(p);
  // CHECK: SUCCESS
  printf ("SUCCESS\n");
  return 0;
}
