#pragma once

namespace cheddar {

/**
 * @brief How many primes of each kind a container holds, and at which ring
 * degree.
 *
 * The degree lives here rather than in a static on Container because a
 * container's buffers are sized `GetNumTotal() * degree`, and two Contexts at
 * different degrees have to be able to coexist in one process. A static made
 * that impossible: whichever Context was constructed last decided the size of
 * every buffer allocated afterwards, silently and for both rings.
 *
 * NPInfo is the natural home because it already travels with every container
 * and every operation, and because `Parameter::LevelToNP` is the one factory
 * that produces the real ones -- so stamping it there reaches everything
 * derived from a parameter. A default-constructed NPInfo carries degree 0,
 * which is consistent: it also carries no primes, so every size it implies is
 * already zero.
 *
 * Note that `operator==` compares the degree too, which turns a cross-ring
 * mix-up into a failure at the existing NP assertions rather than a silently
 * misinterpreted buffer.
 */
struct NPInfo {
  int num_main_ = 0;
  int num_ter_ = 0;
  int num_aux_ = 0;
  int degree_ = 0;

  /**
   * @brief Get the number of Q primes
   *
   * @return int num_q = num_main + num_ter
   */
  int GetNumQ() const;

  /**
   * @brief Get the number of total primes
   *
   * @return int num_total = num_main + num_ter + num_aux
   */
  int GetNumTotal() const;

  /**
   * @brief Construct a new NPInfo object
   *
   * @param num_main number of main primes
   * @param num_ter number of terminal primes
   * @param num_aux number of auxiliary primes
   * @param degree ring degree; 0 only for a placeholder that ModifyNP will
   * overwrite before any buffer is sized from it
   */
  explicit NPInfo(int num_main = 0, int num_ter = 0, int num_aux = 0,
                  int degree = 0);

  // custom copy constructor and assignment operator
  NPInfo(const NPInfo &other);
  NPInfo &operator=(const NPInfo &other);

  // custom comparison operators
  bool operator==(const NPInfo &other) const;
  bool IsSubsetOf(const NPInfo &other) const;
  bool IsSupersetOf(const NPInfo &other) const;
};

}  // namespace cheddar
