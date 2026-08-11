#!/usr/bin/env python3
"""Input validator for AtCoder ABC214 E - Packing Under Range Regulations.

https://atcoder.jp/contests/abc214/tasks/abc214_e

Format
------
    T
    followed by T test cases, each:
    N
    L_1 R_1
    ...
    L_N R_N

Constraints
-----------
    1 <= T <= 2 * 10^5
    1 <= N <= 2 * 10^5
    1 <= L_i <= R_i <= 10^9
    the sum of N over the test cases in one input is at most 2 * 10^5
    (integrality is not spelled out in the statement, but every value is a
    count or a box index, so it is enforced here)

The check is byte-exact, not token-based: separators must be a single space
between L_i and R_i and a single '\\n' at the end of every line, integers may
not carry a sign or a leading zero, and the file must end right after the last
newline. Sloppier input would still parse under a whitespace-splitting reader,
which is exactly the kind of drift a validator exists to catch.

Usage
-----
    python3 validator.py FILE [FILE ...]   # validate each file
    python3 validator.py < FILE            # validate stdin

Exits 0 if every input is valid, 1 otherwise.
"""

from __future__ import annotations

import sys

T_MIN, T_MAX = 1, 2 * 10**5
N_MIN, N_MAX = 1, 2 * 10**5
V_MIN, V_MAX = 1, 10**9
SUM_N_MAX = 2 * 10**5


class ValidationError(Exception):
    pass


class Scanner:
    """A strict byte scanner: every separator has to be asked for explicitly."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def _where(self) -> str:
        line = self.data.count(b"\n", 0, self.pos) + 1
        col = self.pos - (self.data.rfind(b"\n", 0, self.pos) + 1) + 1
        return f"line {line}, column {col}"

    def _fail(self, msg: str) -> None:
        raise ValidationError(f"{self._where()}: {msg}")

    def _peek(self) -> bytes:
        return self.data[self.pos : self.pos + 1]

    def _describe(self, ch: bytes) -> str:
        if not ch:
            return "end of file"
        if ch == b"\n":
            return "'\\n'"
        if ch == b"\r":
            return "'\\r'"
        return repr(ch.decode("latin-1"))

    def read_int(self, lo: int, hi: int, name: str) -> int:
        start = self.pos
        if self._peek() in (b"+", b"-"):
            self._fail(f"{name}: a sign is not allowed")
        while self._peek().isdigit():
            self.pos += 1
        if self.pos == start:
            self._fail(f"{name}: expected a digit, found {self._describe(self._peek())}")
        raw = self.data[start : self.pos]
        if len(raw) > 1 and raw[0:1] == b"0":
            self._fail(f"{name}: leading zero in {raw.decode()!r}")
        value = int(raw)
        if not lo <= value <= hi:
            self.pos = start
            self._fail(f"{name} = {value} is out of range [{lo}, {hi}]")
        return value

    def read_space(self) -> None:
        if self._peek() != b" ":
            self._fail(f"expected a single space, found {self._describe(self._peek())}")
        self.pos += 1
        if self._peek() == b" ":
            self._fail("expected a single space, found more than one")

    def read_newline(self) -> None:
        if self._peek() == b"\r":
            self._fail("expected '\\n', found '\\r' (CRLF line ending)")
        if self._peek() != b"\n":
            self._fail(f"expected '\\n', found {self._describe(self._peek())}")
        self.pos += 1

    def read_eof(self) -> None:
        if self.pos != len(self.data):
            self._fail(f"expected end of file, found {self._describe(self._peek())}")


def validate(data: bytes) -> dict:
    """Validate one whole input. Raises ValidationError, else returns stats."""
    sc = Scanner(data)

    t = sc.read_int(T_MIN, T_MAX, "T")
    sc.read_newline()

    total_n = 0
    max_n = 0
    for case in range(1, t + 1):
        n = sc.read_int(N_MIN, N_MAX, f"case {case}: N")
        sc.read_newline()

        total_n += n
        max_n = max(max_n, n)
        if total_n > SUM_N_MAX:
            sc._fail(
                f"case {case}: the sum of N reached {total_n}, "
                f"which exceeds {SUM_N_MAX}"
            )

        for i in range(1, n + 1):
            left = sc.read_int(V_MIN, V_MAX, f"case {case}: L_{i}")
            sc.read_space()
            right_at = sc.pos
            right = sc.read_int(V_MIN, V_MAX, f"case {case}: R_{i}")
            sc.read_newline()
            if left > right:
                sc.pos = right_at
                sc._fail(f"case {case}: L_{i} = {left} > R_{i} = {right}")

    sc.read_eof()
    return {"T": t, "sum_N": total_n, "max_N": max_n, "bytes": len(data)}


def main(argv: list[str]) -> int:
    paths = argv[1:]
    if not paths:
        try:
            validate(sys.stdin.buffer.read())
        except ValidationError as exc:
            print(f"<stdin>: INVALID: {exc}", file=sys.stderr)
            return 1
        print("<stdin>: OK")
        return 0

    failed = 0
    for path in paths:
        try:
            with open(path, "rb") as handle:
                stats = validate(handle.read())
        except ValidationError as exc:
            failed += 1
            print(f"{path}: INVALID: {exc}", file=sys.stderr)
        except OSError as exc:
            failed += 1
            print(f"{path}: UNREADABLE: {exc}", file=sys.stderr)
        else:
            print(
                f"{path}: OK  T={stats['T']} sum_N={stats['sum_N']} "
                f"max_N={stats['max_N']}"
            )

    print(f"\n{len(paths) - failed}/{len(paths)} valid", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
