#pragma once

#include <vector>

#include "core/DeviceVector.h"
#include "core/ElementWise.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

template <typename word>
class ModSwitchHandler {
 private:
  using Dv = DeviceVector<word>;

  // The order mattters here
  const int level_;
  const int num_aux_;
  const int beta_;

  const Parameter<word> &param_;
  const ElementWiseHandler<word> &elem_handler_;
  const NTTHandler<word> &ntt_handler_;

 public:
  /**
   * @param num_aux how many auxiliary primes the extended basis carries. 0
   * takes the parameter set's own `alpha_`, which is what every level had
   * until now. A smaller basis is legal and is what a key switch low in the
   * chain wants: `alpha_` is sized for the deepest switch in the set, and
   * raising a three-prime level-1 ciphertext into a fifteen-prime basis is
   * most of what ModPack costs. `PrepareEvk` already builds keys against
   * `np.num_aux_` rather than `alpha_` -- it says so -- so only the
   * evaluation side had to learn this.
   */
  ModSwitchHandler(const Parameter<word> &param, int level,
                   const ElementWiseHandler<word> &elem_handler,
                   const NTTHandler<word> &ntt_handler, int num_aux = 0);

  /** @brief The auxiliary prime count this handler was built for. */
  int GetNumAux() const { return num_aux_; }

  // diable copying (or moving also)
  ModSwitchHandler(const ModSwitchHandler &) = delete;
  ModSwitchHandler &operator=(const ModSwitchHandler &) = delete;

  // for forwarding purposes
  ModSwitchHandler(ModSwitchHandler &&) = default;

  void PseudoModUp(DvView<word> &dst, const DvConstView<word> &src,
                   const DvConstView<word> &p_prod) const;
  void ModUp(std::vector<DvView<word>> &dst,
             const DvConstView<word> &src) const;
  /**
   * @brief ModUp for a caller that still holds the coefficient-domain input.
   *
   * The ordinary entry point opens by undoing the NTT it was just handed, so a
   * caller that transformed the polynomial itself immediately beforehand --
   * ModPack does, once per module component -- should not transform it at all.
   * Given the coefficients this pays one constant multiply where the INTT was,
   * carries the limbs that pass through unchanged in the coefficient domain
   * too, and lets the mod-up's own forward transform cover them along with
   * everything else. Two launches per switch disappear with the caller's NTT.
   *
   * The constant is not `mod_up1_`. That one folds in an N^{-1} to normalise
   * an INTT which, here, does not run.
   */
  void ModUpFromCoeff(std::vector<DvView<word>> &dst,
                      const DvConstView<word> &src_coeff) const;
  void ModDown(DvView<word> &dst, const DvConstView<word> &src) const;
  void Rescale(DvView<word> &dst, const DvConstView<word> &src) const;
  void ModDownAndRescale(DvView<word> &dst, const DvConstView<word> &src) const;

 private:
  // ModUp constants
  Dv mod_up1_;
  // The same constant for ModUpFromCoeff: mod_up1_ * N, in Montgomery form.
  // The N is there because the INTT that mod_up1_ normalises does not run on
  // that path, and the Montgomery form because the kernel consuming it
  // multiplies with MultMontgomery rather than through INTTPhase2.
  Dv mod_up1_coeff_;
  // R^2 per q prime: what turns a plain residue into its Montgomery form. The
  // pass-through limbs arrive already transformed and already in Montgomery
  // form on the ordinary path, and NTTForModUp converts nothing, so on the
  // coefficient path they have to be put in that form before it runs.
  Dv mont_r2_;
  std::vector<DeviceVector<make_signed_t<word>>> mod_up2_;

  // ModDown constants
  Dv mod_down1_;
  DeviceVector<make_signed_t<word>> mod_down2_;
  Dv inv_prime_prod_;

  // Rescale constants
  int rescale_pad_start_;
  int rescale_pad_end_;
  int rescale_restore_start_;
  int rescale_restore_end_;

  Dv rescale1_;
  DeviceVector<make_signed_t<word>> rescale2_;
  Dv rescale_inv_prime_prod_;
  Dv rescale_padding_;

  // ModDownAndRescale constants
  Dv mod_down_rescale1_;
  DeviceVector<make_signed_t<word>> mod_down_rescale2_;
  Dv mod_down_rescale_inv_prime_prod_;
  Dv mod_down_rescale_padding_;

  Dv entire_padding_;

  // heuristic CUDA kernel block number;
  static constexpr int block_dim_ = 256;

  void PopulateModSwitchConstants(Dv &const1,
                                  DeviceVector<make_signed_t<word>> &const2,
                                  const std::vector<word> &src_primes,
                                  const std::vector<word> &dst_primes,
                                  int restore_start, int restore_end);
  void PopulateModDownEpilogueConstants(Dv &inv_p_prod, Dv &padding,
                                        const std::vector<word> &src_primes,
                                        const std::vector<word> &dst_primes,
                                        int restore_start, int restore_end);

  // The body behind ModUp and ModUpFromCoeff. Exactly one of the two sources
  // is given: `src` is an NTT-domain input, `src_coeff` a coefficient-domain
  // one.
  void ModUpWorker(std::vector<DvView<word>> &dst, const DvConstView<word> *src,
                   const DvConstView<word> *src_coeff) const;

  enum class ModDownType { ModDown, Rescale, ModDownAndRescale };
  void ModDownWorker(DvView<word> &dst, const DvConstView<word> &src,
                     ModDownType type) const;
};

}  // namespace cheddar
