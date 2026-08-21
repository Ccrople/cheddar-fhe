#include "extension/Profile.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace cheddar {

namespace {

struct Entry {
  std::string label;
  double seconds = 0.0;
  int count = 0;
};

// A vector rather than a map: there are a few dozen labels, the lookup is a
// linear scan of short strings once per step, and the order the layer runs in
// is the order the report should read in.
std::vector<Entry> &Entries() {
  static std::vector<Entry> entries;
  return entries;
}

}  // namespace

void Profile::Add(const std::string &label, double seconds) {
  auto &entries = Entries();
  for (auto &e : entries) {
    if (e.label == label) {
      e.seconds += seconds;
      e.count++;
      return;
    }
  }
  entries.push_back({label, seconds, 1});
}

void Profile::Reset() { Entries().clear(); }

void Profile::Report(const char *title) {
  auto &entries = Entries();
  if (entries.empty()) return;

  double total = 0.0;
  size_t width = 0;
  for (const auto &e : entries) {
    total += e.seconds;
    width = std::max(width, e.label.size());
  }

  // The steps nest -- turn A contains its projections -- so the column does
  // not sum to the total and is not presented as if it did. Each row is
  // against the whole, which is what "where does the time go" asks.
  std::cout << "\n" << title << " (host wall time, device synchronised at "
            << "every step boundary, so each row includes its own drain)\n";
  std::cout << std::string(width + 34, '-') << "\n";
  for (const auto &e : entries) {
    std::cout << "  " << std::left << std::setw(static_cast<int>(width))
              << e.label << std::right << std::fixed << std::setprecision(1)
              << std::setw(10) << e.seconds * 1e3 << " ms" << std::setw(6)
              << e.count << "x" << std::setw(8) << std::setprecision(2)
              << (total > 0.0 ? 100.0 * e.seconds / total : 0.0) << "%\n";
  }
  std::cout << std::string(width + 34, '-') << "\n";
  std::cout << "  " << std::left << std::setw(static_cast<int>(width))
            << "sum of the rows above" << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << total * 1e3 << " ms\n"
            << std::flush;
}

}  // namespace cheddar
