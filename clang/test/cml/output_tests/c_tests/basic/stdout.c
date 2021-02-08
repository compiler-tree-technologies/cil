// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  // CHECK: Printing to stdout
  fprintf(stdout, "Printing to stdout \n");
  return 0;
}
