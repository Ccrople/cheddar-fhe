#pragma once

#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

namespace cheddar {

/**
 * @brief `body(begin, end)` over `[0, n)`, the range cut into one contiguous
 * piece per hardware thread, the first piece run on the caller.
 *
 * For the host-side matrix work behind the compiled transforms --
 * `StripedMatrix::Mult` and the converters' folds -- which is plain
 * arithmetic over tens of millions of entries and ran on one core: the
 * inverse converter's fold alone was ~200 s of a 303 s cold build (Doing.md
 * 3.3). Every caller partitions on an index its writes are disjoint over, so
 * nothing is locked, and each keeps the serial order of accumulation into
 * every entry, so the result is the serial result bit for bit.
 *
 * Nothing thrown inside `body` is caught: a `std::thread` that throws ends
 * the process, so callers check their inputs before entering.
 */
template <typename F>
void ParallelFor(int n, F &&body) {
  if (n <= 0) return;
  int threads = static_cast<int>(std::thread::hardware_concurrency());
  if (threads <= 0) threads = 1;
  threads = std::min(threads, n);
  const int chunk = (n + threads - 1) / threads;
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  for (int t = 1; t < threads; t++) {
    const int begin = t * chunk;
    const int end = std::min(n, begin + chunk);
    if (begin >= end) break;
    pool.emplace_back([&body, begin, end] { body(begin, end); });
  }
  body(0, std::min(n, chunk));
  for (auto &th : pool) th.join();
}

}  // namespace cheddar
