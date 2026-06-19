import math

import numpy as np
import torch

from vq_sigreg.data import BranchConfig, branch_centers, sample_dataset, true_neg_log_density
from vq_sigreg.models import EnumeratedVQ, LeJEPA2D, TransformerChunkDecoder, VQSigReg2D, VQSigRegOpenLoop
from vq_sigreg.official_lejepa import OfficialLeJEPASIGReg
from vq_sigreg.sigreg import sigreg_epps_pulley


def test_branch_density_is_lower_on_branch_centers_than_valley():
    cfg = BranchConfig(n_branches=2, separation=0.8, thickness=0.04, wiggle=0.0)
    x = np.array([0.25], dtype=np.float32)
    centers = branch_centers(x, cfg)[:, 0, :]
    valley = centers.mean(axis=0, keepdims=True)
    nll_centers = true_neg_log_density(np.repeat(x, 2), centers, cfg)
    nll_valley = true_neg_log_density(x, valley, cfg)[0]
    assert float(nll_centers.mean()) < float(nll_valley)


def test_sigreg_is_finite_and_differentiable():
    emb = torch.randn(64, 8, requires_grad=True)
    loss = sigreg_epps_pulley(emb, global_step=3, num_slices=8, num_knots=9)
    assert torch.isfinite(loss)
    loss.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()


def test_official_lejepa_sigreg_wrapper_is_finite_and_differentiable():
    emb = torch.randn(32, 6, requires_grad=True)
    loss_fn = OfficialLeJEPASIGReg(num_slices=4, n_points=9)
    loss = loss_fn(emb)
    assert torch.isfinite(loss)
    loss.backward()
    assert emb.grad is not None
    assert torch.isfinite(emb.grad).all()


def test_vq_enumeration_has_exact_rate_bits():
    vq = EnumeratedVQ(embedding_dim=4, codebook_size=8)
    z, rate = vq.enumerate_codes(torch.device("cpu"), torch.float32)
    assert z.shape == (8, 4)
    assert torch.allclose(rate, torch.full((8,), math.log2(8)))


def test_vq_sigreg_forward_shapes_and_finite_softmin():
    model = VQSigReg2D(hidden_dim=16, embedding_dim=3, codebook_size=5)
    x = torch.randn(7, 1)
    y = torch.randn(7, 2)
    out = model(x, y, select_temp=16.0)
    assert out["shat"].shape == (7, 5, 3)
    assert out["yhat"].shape == (7, 5, 2)
    assert out["softmin"].shape == (7,)
    assert torch.isfinite(out["softmin"]).all()
    assert torch.isfinite(out["code_perplexity"])


def test_transformer_chunk_decoder_preserves_leading_shape():
    decoder = TransformerChunkDecoder(
        embedding_dim=8,
        chunk_dim=12,
        action_dim=2,
        model_dim=16,
        num_layers=1,
        num_heads=4,
    )
    emb = torch.randn(3, 5, 8)
    chunk = decoder(emb)
    assert chunk.shape == (3, 5, 12)
    assert torch.isfinite(chunk).all()


def test_vq_sigreg_openloop_transformer_decoder_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        decoder_type="transformer",
        transformer_layers=1,
        transformer_heads=2,
        transformer_dim=8,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert out["prior_logits"].shape == (4, 6)
    assert out["reranker_logits"].shape == (4, 6)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    candidates = model.candidate_outputs(obs)
    assert candidates["chunk_hat"].shape == (4, 6, 16)
    assert candidates["reranker_logits"].shape == (4, 6)
    assert candidates["cycle_logits"].shape == (4, 6)
    assert torch.isfinite(candidates["cycle_logits"]).all()


def test_vq_sigreg_openloop_final_alignment_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        final_align_residual=True,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_delta_action_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        delta_action_residual=True,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    assert model.apply_delta_action_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_continuous_action_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        continuous_action_residual=True,
        continuous_action_steps=2,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    assert model.apply_continuous_action_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_local_continuous_action_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        local_continuous_action_residual=True,
        local_continuous_action_steps=2,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    assert model.apply_local_continuous_action_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_local_spline_action_residual_shapes_and_zero_start():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        local_spline_action_residual=True,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    assert model.apply_local_spline_action_residual(obs, selected).shape == (4, 16)
    controls = torch.randn(4, 3 * model.action_dim)
    residual = model._bezier_residual_from_controls(controls).reshape(4, model.horizon, model.action_dim)
    assert torch.allclose(residual[:, 0], torch.zeros_like(residual[:, 0]), atol=1e-6)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_contact_action_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        contact_action_residual=True,
        contact_action_steps=2,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    assert model.contact_features(obs).shape == (4, model.contact_feature_dim)
    assert model.apply_contact_action_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_multi_contact_action_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        multi_contact_action_residual=True,
        multi_contact_action_samples=4,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    samples = model.multi_contact_candidate_chunks(obs, selected)
    assert samples.shape == (4, 4, 16)
    assert model.apply_multi_contact_action_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_flow_action_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        flow_action_residual=True,
        flow_action_steps=2,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    residual = torch.zeros_like(selected)
    t = torch.zeros(selected.shape[0], 1)
    assert model.flow_velocity(obs, selected, residual, t).shape == (4, 16)
    assert model.apply_flow_action_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_hierarchical_flow_decoder_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        hierarchical_flow_decoder=True,
        hierarchical_flow_steps=2,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    candidates = model.candidate_outputs(obs)
    assert candidates["chunk_hat"].shape == (4, 6, 16)
    codes = model.code_predictions(model.obs_encoder(obs))
    velocity = model.hierarchical_flow_velocity(
        obs,
        model.obs_encoder(obs),
        codes["z"],
        codes["shat"],
        torch.zeros_like(codes["shat"].new_zeros(4, 6, 16)),
        torch.zeros(4, 6, 1),
    )
    assert velocity.shape == (4, 6, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_vq_sigreg_openloop_hierarchical_flow_relative_anchor():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        hierarchical_flow_decoder=True,
        hierarchical_flow_steps=3,
        hierarchical_flow_relative=True,
    )
    obs = torch.randn(4, 12)
    # Anchor tiles the latest agent xy across the horizon in [x, y, x, y, ...] order.
    anchor = model.hierarchical_flow_anchor(obs, (4,))
    agent_xy = obs.reshape(4, 2, 6)[:, -1, 0:2]
    assert anchor.shape == (4, 16)
    assert torch.allclose(anchor.reshape(4, 8, 2)[:, 0, :], agent_xy)
    assert torch.allclose(anchor.reshape(4, 8, 2)[:, 5, :], agent_xy)
    # With zeroed flow head, decoded relative output collapses to the anchor.
    for p in model.hierarchical_flow_head.parameters():
        torch.nn.init.zeros_(p)
    candidates = model.candidate_outputs(obs)
    assert candidates["chunk_hat"].shape == (4, 6, 16)
    expected = anchor[:, None, :].expand(4, 6, 16)
    assert torch.allclose(candidates["chunk_hat"], expected, atol=1e-5)


def test_vq_sigreg_openloop_fine_vq_residual_shapes():
    model = VQSigRegOpenLoop(
        obs_dim=12,
        chunk_dim=16,
        hidden_dim=16,
        embedding_dim=8,
        codebook_size=6,
        fine_vq_residual=True,
        fine_vq_codebook_size=5,
    )
    obs = torch.randn(4, 12)
    chunk = torch.randn(4, 16)
    out = model(obs, chunk, select_temp=16.0)
    assert out["chunk_hat"].shape == (4, 6, 16)
    assert model.predict_all_chunks(obs).shape == (4, 6, 16)
    selected = out["chunk_hat"][:, 0]
    fine = model.fine_vq_candidate_chunks(obs, selected)
    assert fine["fine_chunks"].shape == (4, 5, 16)
    assert fine["fine_logits"].shape == (4, 5)
    assert model.apply_fine_vq_residual(obs, selected).shape == (4, 16)
    assert torch.isfinite(out["chunk_hat"]).all()


def test_lejepa_and_vq_sigreg_use_matched_encoders_and_decoders():
    lejepa = LeJEPA2D(hidden_dim=32, embedding_dim=2)
    vq = VQSigReg2D(hidden_dim=32, embedding_dim=2, codebook_size=4)
    for a, b in [
        (lejepa.obs_encoder, vq.obs_encoder),
        (lejepa.target_encoder, vq.target_encoder),
        (lejepa.decoder, vq.decoder),
    ]:
        a_shapes = [tuple(p.shape) for p in a.parameters()]
        b_shapes = [tuple(p.shape) for p in b.parameters()]
        assert a_shapes == b_shapes


def test_sample_dataset_shapes():
    data = sample_dataset(11, BranchConfig(n_branches=3), seed=0)
    assert data["x"].shape == (11, 1)
    assert data["y"].shape == (11, 2)
    assert data["branch"].shape == (11,)
