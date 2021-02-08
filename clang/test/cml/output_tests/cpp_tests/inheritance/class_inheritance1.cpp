// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class C {
private:
  int c;

public:
  void set_c(int val) { c = val; }
  int get_c() { return c; }
};

class B : public C {
public:
  int e;
};

int main() {

  B o;
  o.e = 12;
  o.set_c(11);
  printf("%d %d\n", o.get_c(), o.e); // CHECK: 11 12
  return 0;
}
