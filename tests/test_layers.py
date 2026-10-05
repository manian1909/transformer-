import numpy as np

from scratchformer import Tensor
from scratchformer import functional as F
from scratchformer.attention import (
    MultiHeadAttention,
    PositionalEncoding,
    make_causal_mask,
    make_padding_mask,
)
from scratchformer.gradcheck import gradcheck
from scratchformer.model import DecoderLayer, EncoderLayer, Transformer
from scratchformer.nn import LayerNorm, Linear
from scratchformer.rng import manual_seed

rng = np.random.default_rng(7)


def x64(*shape):
    return Tensor(rng.standard_normal(shape), requires_grad=True, dtype=np.float64)


def test_linear_gradients():
    layer = Linear(4, 3).astype(np.float64)
    gradcheck(lambda x, w, b: layer(x), [x64(2, 5, 4), layer.weight, layer.bias])


def test_layernorm_normalises_and_has_correct_gradients():
    ln = LayerNorm(6).astype(np.float64)
    x = Tensor(rng.standard_normal((2, 3, 6)) * 4 + 2, requires_grad=True, dtype=np.float64)
    y = ln(x).data
    np.testing.assert_allclose(y.mean(-1), 0.0, atol=1e-7)
    np.testing.assert_allclose(y.std(-1), 1.0, atol=1e-3)
    ln.gamma.data = rng.standard_normal(6)
    gradcheck(lambda x, g, b: ln(x), [x, ln.gamma, ln.beta])


def test_positional_encoding_values():
    pe = PositionalEncoding(8, max_len=16).table
    np.testing.assert_allclose(pe[0, 0::2], 0.0)
    np.testing.assert_allclose(pe[0, 1::2], 1.0)
    np.testing.assert_allclose(pe[3, 2], np.sin(3 / 10000 ** (2 / 8)), rtol=1e-5)


def test_attention_gradients_with_mask():
    mha = MultiHeadAttention(8, 2).astype(np.float64)
    keys = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    mask = make_padding_mask(keys, 0)
    q, kv = x64(2, 3, 8), x64(2, 4, 8)
    params = [mha.w_q.weight, mha.w_k.weight, mha.w_v.weight, mha.w_o.weight, mha.w_o.bias]
    gradcheck(lambda q, kv, *p: mha(q, kv, kv, mask), [q, kv, *params])


def test_masked_positions_get_zero_attention():
    mha = MultiHeadAttention(8, 2)
    x = Tensor(rng.standard_normal((1, 5, 8)))
    mha(x, x, x, make_causal_mask(5))
    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert np.all(mha.last_attention[0, :, upper] == 0.0)
    np.testing.assert_allclose(mha.last_attention.sum(-1), 1.0, rtol=1e-5)


def test_encoder_and_decoder_layer_gradients():
    enc = EncoderLayer(8, 2, 16).astype(np.float64)
    dec = DecoderLayer(8, 2, 16).astype(np.float64)
    src, tgt = x64(2, 4, 8), x64(2, 3, 8)
    gradcheck(lambda s: enc(s), [src])
    gradcheck(lambda t, m: dec(t, m, make_causal_mask(3)), [tgt, src])


def test_full_transformer_loss_gradients():
    manual_seed(0)
    model = Transformer(7, 7, d_model=8, n_heads=2, n_encoder_layers=1,
                        n_decoder_layers=1, d_ff=16, dropout=0.0).astype(np.float64)
    src = np.array([[3, 4, 5, 2], [6, 3, 2, 0]])
    tgt_in = np.array([[1, 5, 4, 3], [1, 3, 6, 0]])
    tgt_out = np.array([[5, 4, 3, 2], [3, 6, 2, 0]])
    params = model.parameters()

    def loss_fn(*_):
        return F.cross_entropy(model(src, tgt_in), tgt_out, ignore_index=0)

    # probe a few entries of every parameter (embeddings, attention, norms, ...)
    gradcheck(loss_fn, params, max_checks=4)


def test_padding_does_not_change_predictions():
    manual_seed(0)
    model = Transformer(7, 7, d_model=16, n_heads=2, n_encoder_layers=1,
                        n_decoder_layers=1, d_ff=32, dropout=0.0)
    model.eval()
    src = np.array([[3, 4, 5, 2]])
    tgt = np.array([[1, 5, 4]])
    padded_src = np.array([[3, 4, 5, 2, 0, 0]])
    padded_tgt = np.array([[1, 5, 4, 0, 0]])
    a = model(src, tgt).data
    b = model(padded_src, padded_tgt).data[:, :3]
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_decoder_is_causal():
    manual_seed(0)
    model = Transformer(7, 7, d_model=16, n_heads=2, n_encoder_layers=1,
                        n_decoder_layers=1, d_ff=32, dropout=0.0)
    src = np.array([[3, 4, 5, 2]])
    a = model(src, np.array([[1, 5, 4, 3]])).data
    b = model(src, np.array([[1, 5, 6, 6]])).data
    # changing tokens at positions >= 2 must not affect logits at positions < 2
    np.testing.assert_allclose(a[:, :2], b[:, :2], rtol=1e-5)
