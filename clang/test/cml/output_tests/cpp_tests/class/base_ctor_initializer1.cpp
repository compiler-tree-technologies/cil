// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
// XFAIL: *
#include <iostream>

class A {
public:
  A() {
    std::cout << "\nbase class"; // CHECK: base class
  }
};

class B : public A {
public:
  B() : A() {
    std::cout << "\nderived class"; // CHECK: derived class
  }
};

int main() {
  B b;
  return 0;
}
