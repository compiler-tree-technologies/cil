// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

enum ENUM {
  A = 1,
  B = 2,
  C = 3,
  D = 4
};

int main() {
  enum ENUM E;
  E = 10;

  if (E == A)
    printf ("A\n");
  else if (E == B)
    printf ("B\n");
  else if (E == C)
    printf ("C\n");
  else
    // CHECK: Unknown
    printf ("Unknown\n");
  return 0;
}
