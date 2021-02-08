#include <stdio.h>
// RUN: %cml %s -o %t && %t | FileCheck %s
typedef long long int ll;

// Utility function to find
// GCD of 'a' and 'b'
int gcd(int a, int b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}

// Returns LCM of array elements
ll findlcm(int arr[], int n) {
  // Initialize result
  ll ans = arr[0];

  // ans contains LCM of arr[0], ..arr[i]
  // after i'th iteration,
  for (int i = 1; i < n; i++)
    ans = (((arr[i] * ans)) / (gcd(arr[i], ans)));

  return ans;
}

// Driver Code
int main() {
  int arr[] = {2, 7, 3, 9, 4};
  int n = 5;
  printf("%lld", findlcm(arr, n)); // CHECK: 252
  return 0;
}
