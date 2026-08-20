#pragma once

#include <vector>

namespace cheddar {

/**
 * @brief A class for storing bootstrapping parameters.
 *
 */
struct BootParameter {
  /**
   * @brief Construct a new BootParameter object
   *
   * @param max_level the maximum level
   * @param num_cts_levels the number of levels for CoeffToSlot (CtS)
   * @param num_stc_levels the number of levels for SlotToCoeff (StC)
   * @param log_message_ratio approximately log2(q0 / scale to target at the
   * lowest level). Do not modify if you do not know what you are doing.
   * @param num_slack_levels levels left free between EvalMod and StC, for work
   * done in the slot domain before the encoding conversion.
   *
   * Zero -- the default, and what Boot alone needs -- makes StC start exactly
   * where EvalMod ends, which is where every shipped preset puts it and is why
   * `GetStCStartLevel()` equals `default_encryption_level` there.
   *
   * [SYLPH] figure 2's Private Prefill path needs it non-zero: the non-linear
   * operators run at high level, then StC, then the matrix product at the
   * smallest modulus. With zero slack there is nowhere to put them -- HalfBoot
   * is Boot minus StC, so it lands on precisely StC's compiled level, and even
   * a one-level operator drops the ciphertext below it.
   *
   * The levels are not new ones. They come out of the range below StC's
   * output, which the bootstrap cycle otherwise leaves idle: Boot's input has
   * to be at level 0 regardless, so on `bootparam_35` levels 16 down to 0 are
   * the linear phase's working range and only a couple of them are used.
   * Slack D moves the whole conversion down by D and the linear phase with it.
   */
  BootParameter(int max_level, int num_cts_levels, int num_stc_levels,
                int log_message_ratio = 5, int num_slack_levels = 0);

  const int max_level_;
  const int num_cts_levels_;
  const int num_stc_levels_;
  const int num_slack_levels_;

  const int log_message_ratio_;

  /**
   * @brief log2 of the ratio between the modulus and the message bootstrapping
   * assumes.
   *
   * A caller crossing the encoding boundary outside Boot needs it: measured on
   * a slot -> SlotToCoeff -> HalfBoot round trip, the output came back exactly
   * 2^-log_message_ratio of the input, and it also cost that many bits of
   * precision, because feeding EvalMod an argument 32x smaller than it is built
   * for loses five bits.
   */
  int GetLogMessageRatio() const { return log_message_ratio_; }

  /** @brief Levels left free between EvalMod and StC. */
  int GetNumSlackLevels() const { return num_slack_levels_; }

  // The following three parameters are inter-related, so changing
  // one of them requires changing the others.
  // TODO (jongmin.kim): Allow changing these parameters
  const std::vector<double> mod_coefficients_;
  const int num_double_angle_;
  const int initial_K_;

  /**
   * @brief Get the level consumption for EvalMod
   *
   * @return int the number of levels consumed by EvalMod
   */
  int GetNumEvalModLevels() const;

  /**
   * @brief Get the maximum level
   *
   * @return int the maximum level
   */
  int GetMaxLevel() const;

  /**
   * @brief Get the starting level for CoeffToSlot (CtS)
   *
   * @return int the starting level for CoeffToSlot (CtS)
   */
  int GetCtSStartLevel() const;

  /**
   * @brief Get the starting level for EvalMod
   *
   * @return int the starting level for EvalMod
   */
  int GetEvalModStartLevel() const;

  /**
   * @brief Get the starting level for SlotToCoeff (StC), which is where
   * EvalMod ends less `num_slack_levels_`.
   *
   * @return int the starting level for SlotToCoeff (StC)
   */
  int GetStCStartLevel() const;

  /**
   * @brief Get the level EvalMod ends on, which is where HalfBoot lands. Equal
   * to GetStCStartLevel() only when there is no slack.
   *
   * @return int the level EvalMod ends on
   */
  int GetEvalModEndLevel() const;

  /**
   * @brief Get the starting level for bootstrapping. Should be the same as
   * GetMaxLevel() and GetCtSStartLevel().
   *
   * @return int the starting level for bootstrapping
   */
  int GetStartLevel() const;

  /**
   * @brief Get the ending level for bootstrapping. Should be the same as
   * default_encryption_level_.
   *
   * @return int the ending level for bootstrapping
   */
  int GetEndLevel() const;
};

}  // namespace cheddar
