// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

namespace nm1 {
class D {
public:
  int d;
};
} // namespace nm1

namespace nm2 {
class D {
public:
  float d;
};
} // namespace nm2
int main() {
  nm1::D o;
  o.d = 11;
  printf("%d\n", o.d); // CHECK: 11
  nm2::D o2;
  o2.d = 13.32;
  printf("%f\n", o2.d); // CHECK: 13.32
  return 0;
}
