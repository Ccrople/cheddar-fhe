#include "extension/Profile.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

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

  // Formatted into a buffer and written in one go, because `std::fixed` and
  // `setprecision` are sticky stream state: setting them on std::cout here
  // silently reformats whatever the caller prints next, and what the caller
  // prints next is an accuracy ledger.
  //
  // The rows partition the layer -- `ProfileScope` marks leaf steps only, and
  // anything nested inside one gets `NvtxScope`, which does not time -- so the
  // sum is the layer and the percentage column is against it.
  std::ostringstream out;
  out << "\n"
      << title << " (host wall time, device synchronised at every step "
      << "boundary, so each row includes its own drain)\n";
  out << std::string(width + 34, '-') << "\n";
  for (const auto &e : entries) {
    out << "  " << std::left << std::setw(static_cast<int>(width)) << e.label
        << std::right << std::fixed << std::setprecision(1) << std::setw(10)
        << e.seconds * 1e3 << " ms" << std::setw(6) << e.count << "x"
        << std::setw(8) << std::setprecision(2)
        << (total > 0.0 ? 100.0 * e.seconds / total : 0.0) << "%\n";
  }
  out << std::string(width + 34, '-') << "\n";
  out << "  " << std::left << std::setw(static_cast<int>(width))
      << "sum of the rows above" << std::right << std::fixed
      << std::setprecision(1) << std::setw(10) << total * 1e3 << " ms\n";
  std::cout << out.str() << std::flush;
}

}  // namespace cheddar
