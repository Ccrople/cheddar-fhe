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

// The hoisted transform's giant steps as one batched key switch against the
// per-rotation loop: the same kernels, and a modular sum in either order, so
// StC's words must agree exactly.
TEST_P(Testbed32, TheBatchedGiantStepsAreTheSerialOnesWordForWord) {
  using word = uint32_t;
  const int num_slots = param_->MaxNumSlots();
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots, -1.0, 1.0,
                        /*complex=*/!param_->conjugate_invariant_);
  Ciphertext<word> ct;
  EncodeAndEncrypt(ct, msg,
                   boot_context->GetBootParameter().GetStCStartLevel());

  Ciphertext<word> serial, batched;
  HoistHandler<word>::SetGiantStepSerial(true);
  boot_context->SlotToCoeff(serial, num_slots, ct, interface_->GetEvkMap());
  HoistHandler<word>::SetGiantStepSerial(false);
  boot_context->SlotToCoeff(batched, num_slots, ct, interface_->GetEvkMap());

  size_t differ = 0, total = 0;
  const DeviceVector<word> *got[2] = {&batched.bx_, &batched.ax_};
  const DeviceVector<word> *want[2] = {&serial.bx_, &serial.ax_};
  for (int p = 0; p < 2; p++) {
    HostVector<word> a, b;
    CopyDeviceToHost(a, *got[p]);
    CopyDeviceToHost(b, *want[p]);
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); i++) differ += (a[i] != b[i]);
    total += a.size();
  }
  std::cout << "StC giant steps: " << differ << " of " << total
            << " words differ between the batched and the serial key switches"
            << std::endl;
  ASSERT_EQ(differ, 0u);
}

// The batched EvalMod against the per-ciphertext loop: the same compiled
// tree, every key switch through MultKeyBatch with the one key and every
// elementwise pass with the batch on gridDim.z, so the words must agree
// exactly (Doing.md 3.23's lever).
TEST_P(Testbed32, TheBatchedEvalModIsTheSerialOneWordForWord) {
  using word = uint32_t;
  const int num_slots = param_->MaxNumSlots();
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  const int start_level =
      boot_context->GetBootParameter().GetEvalModStartLevel();
  const int group = 5;
  std::vector<Ciphertext<word>> serial(group), batched(group);
  for (int b = 0; b < group; b++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg, num_slots, -1.0, 1.0,
                          /*complex=*/!param_->conjugate_invariant_);
    Ciphertext<word> ct;
    EncodeAndEncrypt(ct, msg, start_level);
    boot_context->Copy(serial[b], ct);
    boot_context->Copy(batched[b], ct);
  }

  std::vector<Ciphertext<word> *> ptrs(group);
  BootContext<word>::SetEvalModSerial(true);
  for (int b = 0; b < group; b++) {
    // The serial loop reads the declared scale EvalMod was compiled for.
    serial[b].SetScale(boot_context->GetEvalModStartScale());
    ptrs[b] = &serial[b];
  }
  boot_context->EvaluateModBatch(ptrs, interface_->GetEvkMap().GetMultiplicationKey());
  BootContext<word>::SetEvalModSerial(false);
  for (int b = 0; b < group; b++) {
    batched[b].SetScale(boot_context->GetEvalModStartScale());
    ptrs[b] = &batched[b];
  }
  boot_context->EvaluateModBatch(ptrs, interface_->GetEvkMap().GetMultiplicationKey());

  size_t differ = 0, total = 0;
  for (int b = 0; b < group; b++) {
    const DeviceVector<word> *got[2] = {&batched[b].bx_, &batched[b].ax_};
    const DeviceVector<word> *want[2] = {&serial[b].bx_, &serial[b].ax_};
    for (int p = 0; p < 2; p++) {
      HostVector<word> a, w;
      CopyDeviceToHost(a, *got[p]);
      CopyDeviceToHost(w, *want[p]);
      ASSERT_EQ(a.size(), w.size());
      for (size_t i = 0; i < a.size(); i++) differ += (a[i] != w[i]);
      total += a.size();
    }
  }
  std::cout << "EvalMod: " << differ << " of " << total
            << " words differ between the batched and the serial evaluations"
            << std::endl;
  ASSERT_EQ(differ, 0u);
}

// [SYLPH] section 3.1.3 states the bootstrap requirement as one number and a
// rule, and neither is the SNR this suite has always printed:
//
//   "given a p-bit bootstrapping procedure and an input bound B, we scale a
//    ciphertext ct by 1/B so that all slot values lie within [-1, 1]. This
//    scaling reduces the effective precision by log2 B bits, yielding
//    p - log2 B bits of precision after bootstrapping. For Llama-3-8B, we use
//    a 20-bit bootstrapping procedure with B = 128, resulting in 13-bit
//    effective precision."
//
// So p is the MAX ABSOLUTE error over a message filling [-1, 1] -- the
// worst slot, not the average and not a ratio -- and the layer's budget is
//
//     effective bits = p - log2(B)   >=  12    ([SYLPH] 3.1.2 / table 7)
//
// with B the largest magnitude any tensor carries into a bootstrap. Sylph's
// calibration (tables 2 and 3) is what puts B at 128: RMSNorm's input falls
// from 2243.97 to 7.65, SoftMax's from 39.24 to 32.78, down-proj's output
// from 310.56 to 1.92, and 128 covers all of them with margin.
//
// This test reports p for whichever preset it runs on, and the two numbers
// that follow from it: the effective precision at Sylph's own B = 128, and
// the largest B this preset could afford while still clearing 12 bits. It
// asserts nothing about p -- a preset that misses 20 bits is a budget fact to
// design around, not a broken bootstrap -- but it does check that the message
// survived at all, so a silent failure cannot masquerade as a small p.
TEST_P(Testbed32, BootstrapPrecisionAgainstSylph) {
  using word = uint32_t;
  const int num_slots = param_->MaxNumSlots();
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);
  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  // Fill [-1, 1], which is the interval the rule is stated on.
  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots, -1.0, 1.0,
                        /*complex=*/!param_->conjugate_invariant_);
  Ciphertext<word> ct, ct_res;
  EncodeAndEncrypt(ct, msg, 0);
  boot_context->Boot(ct_res, ct, interface_->GetEvkMap());
  std::vector<Complex> res;
  DecryptAndDecode(res, ct_res);

  ASSERT_EQ(msg.size(), res.size());
  double max_abs = 0.0, sq_sum = 0.0;
  for (size_t i = 0; i < msg.size(); i++) {
    const double d = std::abs(msg[i] - res[i]);
    max_abs = std::max(max_abs, d);
    sq_sum += d * d;
  }
  const double rms = std::sqrt(sq_sum / static_cast<double>(msg.size()));
  const double p = -std::log2(max_abs);
  const double p_rms = -std::log2(rms);

  // Default cout precision rounds 2.9e-06 to "0.00000", which loses the one
  // number this test exists to report.
  const std::streamsize saved = std::cout.precision(6);
  std::cout << "[SYLPH 3.1.3] bootstrap precision on " << num_slots
            << (param_->conjugate_invariant_ ? " REAL" : " complex")
            << " slots filling [-1, 1]:" << std::endl;
  std::cout << "  max abs err " << std::scientific << max_abs
            << std::defaultfloat << "  ->  p = " << p << " bits" << std::endl;
  std::cout << "  rms abs err " << std::scientific << rms
            << std::defaultfloat << "  ->  " << p_rms
            << " bits (NOT the paper's convention; for comparison only)"
            << std::endl;
  std::cout << "  at Sylph's B = 128: effective " << (p - 7.0)
            << " bits, and the target is 12" << std::endl;
  std::cout << "  largest B this preset affords at 12 bits: 2^" << (p - 12.0)
            << " = " << std::exp2(p - 12.0) << std::endl;
  std::cout << "  ([SYLPH] has p = 20, B = 128, effective 13)" << std::endl;
  std::cout.precision(saved);

  // Not a precision assertion -- only that the message is still there. A
  // bootstrap that lost the payload would otherwise report a small p as
  // though it were a budget number.
  EXPECT_LT(max_abs, 0.05) << "the payload did not survive the bootstrap";
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json", "sylphflow16_35.json",
                    "sylphflow16_40.json", "ci16_35.json", "ci16_40.json"),
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