/*
 * Input validator for AtCoder ABC214 E - Packing Under Range Regulations.
 * https://atcoder.jp/contests/abc214/tasks/abc214_e
 *
 * Format:
 *     T
 *     then T test cases, each:
 *     N
 *     L_1 R_1
 *     ...
 *     L_N R_N
 *
 * Constraints:
 *     1 <= T <= 2 * 10^5
 *     1 <= N <= 2 * 10^5
 *     1 <= L_i <= R_i <= 10^9
 *     the sum of N over the test cases in one input is at most 2 * 10^5
 *
 * Build (Polygon supplies testlib.h itself):
 *     g++ -O2 -o validator validator.cpp
 * Run:
 *     ./validator < input.txt
 */

#include "testlib.h"

const int T_MIN = 1, T_MAX = 200000;
const int N_MIN = 1, N_MAX = 200000;
const int V_MIN = 1, V_MAX = 1000000000;
const int SUM_N_MAX = 200000;

int main(int argc, char *argv[]) {
    registerValidation(argc, argv);

    int t = inf.readInt(T_MIN, T_MAX, "T");
    inf.readEoln();

    long long sumN = 0;
    for (int tc = 1; tc <= t; tc++) {
        setTestCase(tc);

        int n = inf.readInt(N_MIN, N_MAX, "N");
        inf.readEoln();

        sumN += n;
        ensuref(sumN <= SUM_N_MAX,
                "the sum of N over all test cases must be at most %d, but it already reached %lld",
                SUM_N_MAX, sumN);

        for (int i = 1; i <= n; i++) {
            int l = inf.readInt(V_MIN, V_MAX, format("L_%d", i));
            inf.readSpace();
            int r = inf.readInt(V_MIN, V_MAX, format("R_%d", i));
            inf.readEoln();

            ensuref(l <= r, "L_%d must not exceed R_%d, but %d > %d", i, i, l, r);
        }
    }
    unsetTestCase();

    inf.readEof();
    return 0;
}
