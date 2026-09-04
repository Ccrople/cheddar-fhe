#include "extension/CiBatch.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

#include "common/Assert.h"
#include "common/CommonUtils.h"
#include "core/EncodeGpu.h"
#include "extension/Profile.h"

namespace cheddar {

// ---------------------------------------------------------------------------
// CiBatchLayout
// ---------------------------------------------------------------------------

CiBatchLayout::CiBatchLayout(int num_slots, int num_tokens)
    : num_slots{num_slots}, num_tokens{num_tokens} {
  AssertTrue(num_tokens > 0 && IsPowOfTwo(num_tokens),
             "CiBatchLayout: num_tokens must be a power of two");
  AssertTrue(num_slots > 0 && num_slots % num_tokens == 0,
             "CiBatchLayout: the tokens must divide the slot count");
  num_instances = num_slots / num_tokens;
}

CiBatchLayout::CiBatchLayout(int num_slots, int num_tokens, int lanes,
                             int rank)
    : CiBatchLayout(num_slots, num_tokens) {
  AssertTrue(lanes > 0 && rank > 0 && IsPowOfTwo(lanes) && IsPowOfTwo(rank),
             "CiBatchLayout: lanes and rank must be powers of two");
  AssertTrue(num_tokens * rank * lanes == num_slots,
             "CiBatchLayout: the chain addressing needs num_slots = "
             "num_tokens * rank * lanes");
  this->lanes = lanes;
  this->rank = rank;
}

int CiBatchLayout::BlockOf(int token, int group) const {
  AssertTrue(lanes > 0, "CiBatchLayout::BlockOf: not chain-addressed");
  const int num_blocks = num_tokens * rank;
  const int bits = Log2Ceil(num_blocks);
  const int flat = token * rank + group;
  int rev = 0;
  for (int i = 0; i < bits; i++) rev |= ((flat >> i) & 1) << (bits - 1 - i);
  return rev;
}

void CiBatchLayout::Pack(std::vector<Complex> &msg,
                         const std::vector<double> &values) const {
  AssertTrue(values.size() ==
                 static_cast<size_t>(num_instances) * num_tokens,
             "CiBatchLayout::Pack: values must be [instance][token]");
  msg.assign(num_slots, Complex(0.0, 0.0));
  for (int b = 0; b < num_instances; b++) {
    for (int t = 0; t < num_tokens; t++) {
      msg[Slot(t, b)] =
          Complex(values[static_cast<size_t>(b) * num_tokens + t], 0.0);
    }
  }
}

void CiBatchLayout::Unpack(std::vector<double> &values,
                           const std::vector<Complex> &msg) const {
  AssertTrue(static_cast<int>(msg.size()) == num_slots,
             "CiBatchLayout::Unpack: message size is not the slot count");
  values.assign(static_cast<size_t>(num_instances) * num_tokens, 0.0);
  for (int b = 0; b < num_instances; b++) {
    for (int t = 0; t < num_tokens; t++) {
      values[static_cast<size_t>(b) * num_tokens + t] = msg[Slot(t, b)].real();
    }
  }
}

void CiBatchLayout::PackPerToken(std::vector<Complex> &msg,
                                 const std::vector<double> &per_token) const {
  AssertTrue(static_cast<int>(per_token.size()) == num_tokens,
             "CiBatchLayout::PackPerToken: one value per token");
  msg.assign(num_slots, Complex(0.0, 0.0));
  for (int t = 0; t < num_tokens; t++) {
    for (int b = 0; b < num_instances; b++) {
      msg[Slot(t, b)] = Complex(per_token[t], 0.0);
    }
  }
}

#ifdef USE_CUBLAS

// ---------------------------------------------------------------------------
// CiBatchProjection
// ---------------------------------------------------------------------------

namespace kernel {
// `res[c][o] = gain[c] * w[c][o]`: the gain on the INPUT channel, which is
// the axis RMSNorm's weight lives on.
__global__ void FoldGainKernel(float *res, const float *w, const float *gain,
                               int in, int out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t n = static_cast<size_t>(in) * out;
  if (idx >= n) return;
  const int c = static_cast<int>(idx / out);
  res[idx] = w[idx] * gain[c];
}
}  // namespace kernel

template <typename word>
CiBatchProjection<word>::CiBatchProjection(ConstContextPtr<word> context,
                                           const Config &cfg)
    : context_{std::move(context)}, cfg_{cfg} {
  AssertTrue(cfg_.rows_per_tile > 0,
             "CiBatchProjection: rows_per_tile must be positive");
  blas_ = std::make_unique<PcmmBlasHandler<word>>(context_->param_);
}

template <typename word>
void CiBatchProjection<word>::Prepare(const std::string &name,
                                      const float *tensor, int in, int out,
                                      int level, double w_scale,
                                      double input_scale_ratio) {
  NvtxScope _nv("batch: Prepare");
  AssertTrue(tensor != nullptr, "CiBatchProjection::Prepare: null tensor");
  AssertTrue(in > 0 && out > 0, "CiBatchProjection::Prepare: bad shape");
  AssertTrue(level >= 1 && level <= context_->param_.max_level_,
             "CiBatchProjection::Prepare: the product needs a level to "
             "rescale from");
  AssertTrue(input_scale_ratio > 0.0,
             "CiBatchProjection::Prepare: input_scale_ratio");

  Operand op;
  op.in = in;
  op.out = out;
  op.level = level;
  op.w_scale = w_scale;
  // The one scale that lands the rescaled product canonical (class comment),
  // divided by whatever the inputs' recorded scale carries above canonical.
  op.weight_scale = context_->param_.GetScale(level) / input_scale_ratio;
  op.rows_per_tile = std::min(cfg_.rows_per_tile, out);

  const GpuEncoder<word> &encoder = context_->gpu_encoder_;
  const NPInfo np = context_->param_.LevelToNP(level);
  const int num_primes = np.GetNumTotal();

  // The column map is the identity on the input channel and does not move
  // between tiles; the row map is the tile's window onto the output axis.
  HostVector<int32_t> col_map(in);
  for (int c = 0; c < in; c++) col_map[c] = c;
  col_map_.resize(in);
  CopyHostToDevice(col_map_, col_map);
  row_map_.resize(op.rows_per_tile);
  {
    const size_t n =
        static_cast<size_t>(num_primes) * op.rows_per_tile * in;
    AssertTrue(n < (static_cast<size_t>(1) << 31),
               "CiBatchProjection::Prepare: a tile's residues do not fit an "
               "int index; lower rows_per_tile");
    if (static_cast<size_t>(residues_.size()) < n) {
      residues_.resize(static_cast<int>(n));
    }
  }

  for (int r0 = 0; r0 < out; r0 += op.rows_per_tile) {
    const int rows = std::min(op.rows_per_tile, out - r0);
    HostVector<int32_t> row_map(rows);
    for (int r = 0; r < rows; r++) row_map[r] = r0 + r;
    CopyHostToDevice(row_map_, row_map);
    // `EncodeResiduesGathered` reads `tensor[col_map[c] * out + row_map[r]]`
    // for the (r, c) entry of a `rows x cols` matrix -- rows are OUTPUT
    // channels, the GEMM's `u[o][c]`, and the tensor is `[in][out]`.
    encoder.template EncodeResiduesGathered<float>(
        residues_.data(), level, op.weight_scale, tensor, out, row_map_.data(),
        col_map_.data(), rows, in, w_scale);
    SplitMatrix tile;
    blas_->SplitMatrixFromResidues(tile, level, op.weight_scale,
                                   residues_.data(), rows, in);
    op.bytes += PcmmBlasHandler<word>::SplitBytes(tile);
    op.tiles.push_back(std::move(tile));
  }
  if (cfg_.verbose) {
    std::cout << "  [batch] " << name << ": " << in << " x " << out
              << " at level " << level << ", " << op.tiles.size()
              << " tile(s), "
              << static_cast<double>(op.bytes) / (1024.0 * 1024.0) << " MiB"
              << std::endl;
  }
  operands_[name] = std::move(op);
}

template <typename word>
size_t CiBatchProjection<word>::Bytes() const {
  size_t n = 0;
  for (const auto &kv : operands_) n += kv.second.bytes;
  return n;
}

template <typename word>
void CiBatchProjection<word>::Split(Source &src, const std::vector<Ct> &x,
                                    const std::string &name) const {
  NvtxScope _nv("batch: split source");
  auto it = operands_.find(name);
  AssertTrue(it != operands_.end(),
             "CiBatchProjection::Split: no operand named " + name);
  const Operand &op = it->second;
  AssertTrue(static_cast<int>(x.size()) == op.in,
             "CiBatchProjection::Split: " + name + " contracts " +
                 std::to_string(op.in) + " channels, given " +
                 std::to_string(x.size()));
  const NPInfo np = context_->param_.LevelToNP(op.level);
  for (const auto &c : x) {
    AssertTrue(c.GetNP() == np,
               "CiBatchProjection::Split: an input is not at the operand's "
               "level");
    AssertFalse(c.HasRx(), "CiBatchProjection::Split: input has rx");
  }
  // Cut for the largest tile any operand of this handler has, so that every
  // operand sharing the split's (in, level) can be projected from it.
  blas_->PrepareSource(src.split, op.tiles.front(), x,
                       std::min(cfg_.rows_per_tile, op.out) > op.rows_per_tile
                           ? cfg_.rows_per_tile
                           : op.rows_per_tile);
  src.in = op.in;
  src.level = op.level;
}

namespace kernel {
// dst_ptrs[2 * z + blockIdx.y][i] = src[z][poly blockIdx.y][i]: one tile's
// rescaled rows, `[row][b|a][words]` in `src`, into the rows' own buffers.
template <typename word>
__global__ void ScatterRows(word *const *dst_ptrs, int poly_words,
                            const word *src) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= poly_words) return;
  const int z = blockIdx.z;
  dst_ptrs[2 * z + blockIdx.y][i] =
      src[static_cast<size_t>(z) * 2 * poly_words +
          static_cast<size_t>(blockIdx.y) * poly_words + i];
}
}  // namespace kernel

template <typename word>
bool CiBatchProjection<word>::RescaleSerial() {
  static const bool serial = [] {
    const char *e = std::getenv("CHEDDAR_BATCH_RESCALE_SERIAL");
    return e != nullptr && e[0] == '1';
  }();
  return serial;
}

template <typename word>
void CiBatchProjection<word>::BeginSplit(Source &src, int in, int level,
                                         int num_slots, double scale) const {
  AssertTrue(in > 0 && level >= 1 && level <= context_->param_.max_level_,
             "CiBatchProjection::BeginSplit: bad shape or level");
  AssertTrue(scale >= 0.0, "CiBatchProjection::BeginSplit: negative scale");
  blas_->PrepareSourceBegin(
      src.split, level, in, cfg_.rows_per_tile,
      scale > 0.0 ? scale : context_->param_.GetScale(level), num_slots);
  src.in = in;
  src.level = level;
}

template <typename word>
void CiBatchProjection<word>::AddColumn(Source &src, int col,
                                        const Ct &x) const {
  blas_->SplitSourceColumn(src.split, col, x);
}

template <typename word>
void CiBatchProjection<word>::Project(std::vector<Ct> &res, const Source &src,
                                      const std::string &name,
                                      int tile) const {
  NvtxScope _nv("batch: Project tile");
  auto it = operands_.find(name);
  AssertTrue(it != operands_.end(),
             "CiBatchProjection::Project: no operand named " + name);
  const Operand &op = it->second;
  AssertTrue(src.in == op.in && src.level == op.level,
             "CiBatchProjection::Project: the split source is not " + name +
                 "'s shape");
  AssertTrue(tile >= 0 && tile < static_cast<int>(op.tiles.size()),
             "CiBatchProjection::Project: no tile " + std::to_string(tile) +
                 " in " + name);
  const SplitMatrix &u = op.tiles[tile];

  if (RescaleSerial()) {
    // The A/B: one `Rescale` per output ciphertext.
    std::vector<Ct> part;
    {
      NvtxScope _g("batch: gemm");
      blas_->Multiply(part, u, src.split);
    }
    NvtxScope _r("batch: rescale");
    res.clear();
    res.reserve(part.size());
    for (auto &p : part) {
      Ct r;
      context_->Rescale(r, p);
      res.push_back(std::move(r));
    }
    return;
  }

  // The tile's rows into one buffer, ONE rescale over its `2 * rows`
  // polynomials (`ModSwitchHandler::RescaleBatch`, word for word the
  // per-ciphertext one), and the rows scattered into their ciphertexts. At
  // the model's width a projection is 4096-14336 rescales, each a
  // launch-bound few hundred microseconds on a 4-limb ciphertext; batched
  // they are a handful of launches.
  const Parameter<word> &param = context_->param_;
  const int rows = u.rows;
  const int degree = param.degree_;
  const int level = op.level;
  const int q_words = src.split.np.GetNumTotal() * degree;
  const NPInfo next_np = param.LevelToNP(level - 1);
  const int next_words = next_np.GetNumTotal() * degree;
  // The two tile buffers are kept between calls (grown, never shrunk): a
  // projection at the model's width is 28-56 tiles, and a gigabyte
  // allocated and freed per tile beside thousands of live ciphertexts is
  // what fragmented the pool (Doing.md 7.4).
  const size_t prod_words = static_cast<size_t>(rows) * 2 * q_words;
  const size_t resc_words = static_cast<size_t>(rows) * 2 * next_words;
  if (static_cast<size_t>(prod_.size()) < prod_words) {
    prod_ = DeviceVector<word>();
    prod_.resize(static_cast<int>(prod_words));
  }
  if (static_cast<size_t>(rescaled_.size()) < resc_words) {
    rescaled_ = DeviceVector<word>();
    rescaled_.resize(static_cast<int>(resc_words));
  }
  DeviceVector<word> &prod = prod_;
  DeviceVector<word> &rescaled = rescaled_;
  {
    NvtxScope _g("batch: gemm");
    blas_->MultiplyInto(prod.data(), 2 * q_words, u, src.split);
  }
  {
    NvtxScope _r("batch: rescale batch");
    context_->mod_switch_handlers_.at(level).RescaleBatch(
        rescaled.data(), next_words, prod.data(), q_words, 2 * rows);
  }
  {
    NvtxScope _s("batch: scatter");
    res.clear();
    res.resize(rows);
    HostVector<word *> ptrs(2 * rows);
    for (int i = 0; i < rows; i++) {
      Ct &r = res[i];
      r.RemoveRx();
      r.ModifyNP(next_np);
      r.SetScale(u.scale * src.split.scale / param.GetRescalePrimeProd(level));
      r.SetNumSlots(src.split.num_slots);
      ptrs[2 * i] = r.bx_.data();
      ptrs[2 * i + 1] = r.ax_.data();
    }
    DeviceVector<word *> ptrs_dev(2 * rows);
    CopyHostToDevice(ptrs_dev, ptrs);
    constexpr int kBlock = 256;
    kernel::ScatterRows<word>
        <<<dim3((next_words + kBlock - 1) / kBlock, 2, rows), kBlock>>>(
            ptrs_dev.data(), next_words, rescaled.data());
  }
}

template <typename word>
void CiBatchProjection<word>::Project(std::vector<Ct> &res,
                                      const std::vector<Ct> &x,
                                      const std::string &name) const {
  NvtxScope _nv("batch: Project");
  Source src;
  Split(src, x, name);
  const Operand &op = operands_.at(name);
  res.clear();
  res.reserve(op.out);
  for (int t = 0; t < static_cast<int>(op.tiles.size()); t++) {
    std::vector<Ct> part;
    Project(part, src, name, t);
    for (auto &p : part) res.push_back(std::move(p));
  }
}

template <typename word>
void CiBatchProjection<word>::FoldGain(DeviceVector<float> &res,
                                       const float *tensor, int in, int out,
                                       const std::vector<double> &gain) {
  AssertTrue(static_cast<int>(gain.size()) == in,
             "CiBatchProjection::FoldGain: one gain per input channel");
  HostVector<float> h_gain(in);
  for (int c = 0; c < in; c++) h_gain[c] = static_cast<float>(gain[c]);
  DeviceVector<float> d_gain(in);
  CopyHostToDevice(d_gain, h_gain);
  const size_t n = static_cast<size_t>(in) * out;
  AssertTrue(n < (static_cast<size_t>(1) << 31),
             "CiBatchProjection::FoldGain: tensor too large for an int index");
  res.resize(static_cast<int>(n));
  constexpr int kBlock = 256;
  const int grid = static_cast<int>((n + kBlock - 1) / kBlock);
  kernel::FoldGainKernel<<<grid, kBlock>>>(res.data(), tensor, d_gain.data(),
                                          in, out);
  // The gain buffer is freed at scope exit on the same stream the kernel
  // runs on; frees are stream-ordered, so no synchronise is needed.
}

template class CiBatchProjection<uint32_t>;
template class CiBatchProjection<uint64_t>;

#endif  // USE_CUBLAS

}  // namespace cheddar
