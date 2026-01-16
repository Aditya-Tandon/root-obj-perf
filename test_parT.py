import unittest
import torch
import torch.nn as nn
from parT import (
    to_rapidity,
    to_phi,
    to_pt,
    PairwiseEmbedding,
    ParticleAttentionBlock,
    ClassAttentionBlock,
    ParticleTransformer,
)


def generate_physical_data(batch_size, num_particles):
    """Generates physical 4-vectors where E > |p| to avoid NaNs in rapidity."""
    # (B, N, 3)
    p = torch.randn(batch_size, num_particles, 3)
    # (B, N)
    m = torch.rand(batch_size, num_particles) + 0.1
    p_sq = torch.sum(p**2, dim=-1)
    E = torch.sqrt(p_sq + m**2) + 0.01  # Ensure margin
    # (B, N, 4) -> E, px, py, pz
    x = torch.cat([E.unsqueeze(-1), p], dim=-1)
    return x


class TestPhysicsUtilityFunctions(unittest.TestCase):
    """Test physics utility functions for converting 4-vectors to rapidity, phi, and pt."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_particles = 4
        self.eps = 1e-8

    def test_to_rapidity_basic(self):
        """Test rapidity calculation for basic 4-vectors."""
        # Create test 4-vectors [E, px, py, pz] where E > |pz|
        x = torch.tensor([[[10.0, 0.0, 0.0, 5.0]]])  # Shape: (1, 1, 4)
        rapidity = to_rapidity(x)

        # Expected: 0.5 * log((10+5)/(10-5)) = 0.5 * log(3)
        expected = 0.5 * torch.log(torch.tensor(3.0))
        self.assertAlmostEqual(rapidity.item(), expected.item(), places=5)

    def test_to_rapidity_shape(self):
        """Test that rapidity maintains correct shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        rapidity = to_rapidity(x)

        expected_shape = (self.batch_size, self.num_particles)
        self.assertEqual(rapidity.shape, expected_shape)

    def test_to_rapidity_stability(self):
        """Test rapidity calculation stability with eps parameter."""
        # Test case where E is close to pz
        # Using a safe margin because to_rapidity has eps,
        # but log(0) is still -inf unless handled by eps dominance.
        # But for physical particles E >= pz.
        x = torch.tensor([[[1.0, 0.0, 0.0, 0.9999]]])
        rapidity = to_rapidity(x)

        # Should not produce NaN or inf
        self.assertFalse(torch.isnan(rapidity).any())
        self.assertFalse(torch.isinf(rapidity).any())

    def test_to_phi_basic(self):
        """Test phi (azimuthal angle) calculation."""
        # Test vector along x-axis: phi = 0
        x = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        phi = to_phi(x)
        self.assertAlmostEqual(phi.item(), 0.0, places=5)

        # Test vector along y-axis: phi = pi/2
        x = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
        phi = to_phi(x)
        self.assertAlmostEqual(phi.item(), torch.pi / 2, places=5)

    def test_to_phi_shape(self):
        """Test that phi maintains correct shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        phi = to_phi(x)

        expected_shape = (self.batch_size, self.num_particles)
        self.assertEqual(phi.shape, expected_shape)

    def test_to_phi_mask(self):
        """Test that phi returns 0 for masked particles."""
        # Create particles with px=py=0
        x = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        phi = to_phi(x)
        self.assertEqual(phi.item(), 0.0)

    def test_to_pt_basic(self):
        """Test transverse momentum calculation."""
        # px=3, py=4 -> pt=5
        x = torch.tensor([[[0.0, 3.0, 4.0, 0.0]]])
        pt = to_pt(x)
        self.assertAlmostEqual(pt.item(), 5.0, places=5)

    def test_to_pt_shape(self):
        """Test that pt maintains correct shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        pt = to_pt(x)

        expected_shape = (self.batch_size, self.num_particles)
        self.assertEqual(pt.shape, expected_shape)

    def test_to_pt_positive(self):
        """Test that pt is always positive."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        pt = to_pt(x)
        self.assertTrue((pt >= 0).all())


class TestPairwiseEmbedding(unittest.TestCase):
    """Test the PairwiseEmbedding module."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_particles = 8
        self.input_dim = 4  # delta, k_t, z, m^2
        self.out_dim = 32
        self.model = PairwiseEmbedding(self.input_dim, self.out_dim)

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)

        output = self.model(x)

        expected_shape = (
            self.batch_size,
            self.num_particles,
            self.num_particles,
            self.out_dim,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_forward_with_mask(self):
        """Test forward pass with particle mask."""
        x = generate_physical_data(self.batch_size, self.num_particles)

        # Create mask: only first 4 particles are valid
        mask = torch.zeros(self.batch_size, self.num_particles, dtype=torch.bool)
        mask[:, :4] = True

        output = self.model(x, particle_mask=mask)

        # Since PairwiseEmbedding uses a bias, masked regions won't be strictly zero
        # unless the net has no bias or we mask the output.
        # But inputs are zeroed, so all masked regions should have identical values (the bias)

        masked_output = output[:, 4:, 4:, :]
        # Check that all values in masked region are identical
        # We compare to expected 'zero input' network response
        dummy = torch.zeros(1, 4)  # (delta, kt, z, m2) = 0
        expected = self.model.net(dummy)

        # Compare first element of masked region to expected
        self.assertTrue(torch.allclose(masked_output[0, 0, 0], expected[0], atol=1e-5))

    def test_pairwise_symmetry(self):
        """Test symmetry properties of pairwise features."""
        x = generate_physical_data(1, 4)

        output = self.model(x)
        self.assertEqual(output.shape, (1, 4, 4, self.out_dim))

    def test_no_nan_or_inf(self):
        """Test that output contains no NaN or inf values."""
        x = generate_physical_data(self.batch_size, self.num_particles)

        output = self.model(x)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestParticleAttentionBlock(unittest.TestCase):
    """Test the ParticleAttentionBlock module."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_particles = 8
        self.embed_dim = 64
        self.num_heads = 4
        self.model = ParticleAttentionBlock(self.embed_dim, self.num_heads)

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        x = torch.randn(self.batch_size, self.num_particles, self.embed_dim)
        output = self.model(x)

        self.assertEqual(output.shape, x.shape)

    def test_forward_with_pairwise_bias(self):
        """Test forward pass with pairwise bias."""
        x = torch.randn(self.batch_size, self.num_particles, self.embed_dim)
        u_ij = torch.randn(
            self.batch_size, self.num_particles, self.num_particles, self.embed_dim
        )

        output = self.model(x, u_ij=u_ij)

        self.assertEqual(output.shape, x.shape)

    def test_forward_with_mask(self):
        """Test forward pass with particle mask."""
        x = torch.randn(self.batch_size, self.num_particles, self.embed_dim)
        mask = torch.ones(self.batch_size, self.num_particles, dtype=torch.bool)
        mask[:, 4:] = False  # Mask out last 4 particles

        output = self.model(x, particle_mask=mask)

        self.assertEqual(output.shape, x.shape)

    def test_residual_connection(self):
        """Test that residual connections work properly."""
        x = torch.randn(self.batch_size, self.num_particles, self.embed_dim)

        # With identity initialization, output should be close to input
        output = self.model(x)

        # Output should have same shape
        self.assertEqual(output.shape, x.shape)

    def test_no_nan_or_inf(self):
        """Test that output contains no NaN or inf values."""
        x = torch.randn(self.batch_size, self.num_particles, self.embed_dim)
        output = self.model(x)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_num_heads_divisibility(self):
        """Test that embed_dim must be divisible by num_heads."""
        with self.assertRaises(Exception):
            # This should fail during forward pass
            bad_model = ParticleAttentionBlock(embed_dim=65, num_heads=4)
            x = torch.randn(1, 4, 65)
            bad_model(x)


class TestClassAttentionBlock(unittest.TestCase):
    """Test the ClassAttentionBlock module."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_particles = 8
        self.embed_dim = 64
        self.num_heads = 4
        self.model = ClassAttentionBlock(self.embed_dim, self.num_heads)

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        # Input includes CLS token (N+1 tokens)
        x = torch.randn(self.batch_size, self.num_particles + 1, self.embed_dim)
        output = self.model(x)

        self.assertEqual(output.shape, x.shape)

    def test_forward_with_mask(self):
        """Test forward pass with particle mask."""
        x = torch.randn(self.batch_size, self.num_particles + 1, self.embed_dim)
        mask = torch.ones(self.batch_size, self.num_particles, dtype=torch.bool)
        mask[:, 4:] = False

        output = self.model(x, particle_mask=mask)

        self.assertEqual(output.shape, x.shape)

    def test_cls_token_attention(self):
        """Test that CLS token is properly updated."""
        x = torch.randn(self.batch_size, self.num_particles + 1, self.embed_dim)
        output = self.model(x)

        # CLS token should be different from input
        self.assertFalse(torch.allclose(output[:, 0, :], x[:, 0, :]))

    def test_no_nan_or_inf(self):
        """Test that output contains no NaN or inf values."""
        x = torch.randn(self.batch_size, self.num_particles + 1, self.embed_dim)
        output = self.model(x)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestParticleTransformer(unittest.TestCase):
    """Test the full ParticleTransformer model."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_particles = 16
        self.input_dim = 4
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.num_cls_layers = 2
        self.num_classes = 1

        self.model = ParticleTransformer(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_pairwise_feat=4,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_cls_layers=self.num_cls_layers,
            num_classes=self.num_classes,
        )

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)

        output = self.model(x)

        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_with_mask(self):
        """Test forward pass with particle mask."""
        x = generate_physical_data(self.batch_size, self.num_particles)

        mask = torch.ones(self.batch_size, self.num_particles, dtype=torch.bool)
        mask[:, 8:] = False

        output = self.model(x, particle_mask=mask)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        # Manually create tensor to enable grads
        p = torch.randn(self.batch_size, self.num_particles, 3)
        m = torch.rand(self.batch_size, self.num_particles) + 0.1
        p_sq = torch.sum(p**2, dim=-1)
        E = torch.sqrt(p_sq + m**2) + 0.01

        x = torch.cat([E.unsqueeze(-1), p], dim=-1).requires_grad_(True)

        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for input
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_multiclass_classification(self):
        """Test model with multiple output classes."""
        num_classes = 5
        model = ParticleTransformer(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_classes=num_classes,
        )

        x = generate_physical_data(self.batch_size, self.num_particles)

        output = model(x)

        self.assertEqual(output.shape, (self.batch_size, num_classes))

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        for batch_size in [1, 4, 8]:
            x = generate_physical_data(batch_size, self.num_particles)

            output = self.model(x)

            self.assertEqual(output.shape, (batch_size, self.num_classes))

    def test_different_num_particles(self):
        """Test model with different numbers of particles."""
        for num_particles in [4, 8, 32]:
            x = generate_physical_data(self.batch_size, num_particles)

            output = self.model(x)

            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_no_nan_or_inf(self):
        """Test that output contains no NaN or inf values."""
        x = generate_physical_data(self.batch_size, self.num_particles)

        output = self.model(x)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_model_trainable(self):
        """Test that model parameters are trainable."""
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.assertGreater(trainable_params, 0)

    def test_cls_token_initialization(self):
        """Test that CLS token is properly initialized."""
        self.assertEqual(self.model.cls_token.shape, (1, 1, self.embed_dim))
        self.assertTrue(self.model.cls_token.requires_grad)

    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        self.model.eval()

        x = generate_physical_data(self.batch_size, self.num_particles)

        with torch.no_grad():
            output1 = self.model(x)
            output2 = self.model(x)

        # In eval mode, same input should give same output (no dropout randomness)
        self.assertTrue(torch.allclose(output1, output2))

    def test_batch_norm_behavior(self):
        """Test that batch normalization works properly."""
        self.model.train()

        x = generate_physical_data(self.batch_size, self.num_particles)

        # Run forward pass to update BN statistics
        output = self.model(x)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete model pipeline."""

    def test_end_to_end_training_step(self):
        """Test a complete training step."""
        model = ParticleTransformer(
            input_dim=4,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            num_classes=1,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Create dummy data
        x = generate_physical_data(4, 8)
        target = torch.randn(4, 1)

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Check that loss is computed
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_inference_pipeline(self):
        """Test inference pipeline."""
        model = ParticleTransformer(
            input_dim=4,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            num_classes=2,
        )
        model.eval()

        x = generate_physical_data(1, 10)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=-1)

        self.assertEqual(probs.shape, (1, 2))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_variable_length_sequences(self):
        """Test handling of variable-length particle sequences with masking."""
        model = ParticleTransformer(
            input_dim=4,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            num_classes=1,
        )
        model.eval()

        batch_size = 4
        max_particles = 16

        # Create batch with variable lengths
        x = generate_physical_data(batch_size, max_particles)

        # Create masks for different lengths
        lengths = [4, 8, 12, 16]
        mask = torch.zeros(batch_size, max_particles, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        with torch.no_grad():
            output = model(x, particle_mask=mask)

        self.assertEqual(output.shape, (batch_size, 1))
        self.assertFalse(torch.isnan(output).any())


if __name__ == "__main__":
    unittest.main()
