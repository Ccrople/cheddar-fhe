#include "Testbed.h"

// Full slot packing, which is MaxNumSlots() and not a constant: degree / 2 on
// the ordinary ring and degree on the conjugate-invariant one. Every preset
// here is logN 16, so this is the 1 << 15 it used to say for all of them
// except ci16_35, where it is 1 << 16 real slots.
static constexpr int warm_up = 5;

TEST_P(Testbed32, Bootstrap) {
  using word = uint32_t;
  const int num_slots = param_->MaxNumSlots();
  std::cout << "Preparing for bootstrapping (num_slots: " << num_slots << ")"
            << std::endl;
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg1;
  // The real subring has real slots and Encode refuses an imaginary part.
  GenerateRandomMessage(msg1, num_slots, -1.0, 1.0,
                        /*complex=*/!param_->conjugate_invariant_);
  Ciphertext<word> ct1;

  Ciphertext<word> ct_res;
  std::vector<Complex> res;

  __ProfileStart("Boot-Basic", warm_up, EncodeAndEncrypt(ct1, msg1, 0));
  boot_context->Boot(ct_res, ct1, interface_->GetEvkMap());
  __ProfileEnd("Boot-Basic");

  // check correctness
  DecryptAndDecode(res, ct_res);
  CompareMessages(msg1, res);

  __ProfileStart("Boot-MinKS", warm_up, EncodeAndEncrypt(ct1, msg1, 0));
  boot_context->Boot(ct_res, ct1, interface_->GetEvkMap(), true);
  __ProfileEnd("Boot-MinKS");

  // check correctness
  DecryptAndDecode(res, ct_res);
  CompareMessages(msg1, res);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json", "sylphflow16_35.json",
                    "ci16_35.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });

TEST_P(Testbed64, Bootstrap) {
  using word = uint64_t;
  const int num_slots = param_->MaxNumSlots();
  std::cout << "Preparing for bootstrapping (num_slots: " << num_slots << ")"
            << std::endl;
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg1;
  // The real subring has real slots and Encode refuses an imaginary part.
  GenerateRandomMessage(msg1, num_slots, -1.0, 1.0,
                        /*complex=*/!param_->conjugate_invariant_);
  Ciphertext<word> ct1;

  Ciphertext<word> ct_res;
  std::vector<Complex> res;

  __ProfileStart("Boot-Basic", warm_up, EncodeAndEncrypt(ct1, msg1, 0));
  boot_context->Boot(ct_res, ct1, interface_->GetEvkMap());
  __ProfileEnd("Boot-Basic");

  // check correctness
  DecryptAndDecode(res, ct_res);
  CompareMessages(msg1, res);

  __ProfileStart("Boot-MinKS", warm_up, EncodeAndEncrypt(ct1, msg1, 0));
  boot_context->Boot(ct_res, ct1, interface_->GetEvkMap(), true);
  __ProfileEnd("Boot-MinKS");

  // check correctness
  DecryptAndDecode(res, ct_res);
  CompareMessages(msg1, res);
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed64, testing::Values("bootparam_40_64bit.json"),
    [](const testing::TestParamInfo<Testbed64::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });