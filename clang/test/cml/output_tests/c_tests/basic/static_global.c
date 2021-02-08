// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

static int val;

int main() {

  val = 10;

  val++;

  printf("%d\n",val); // CHECK: 11
  return 0;
}
