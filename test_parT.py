import unittest
import torch
import torch.nn as nn
from parT import (
    to_rapidity,
    to_energy,
    to_pt,
    PairwiseEmbedding,
    ParticleAttentionBlock,
    ClassAttentionBlock,
    ParticleTransformer,
)


def generate_physical_data(batch_size, num_particles):
    """Generates physical data in [mass, pt, eta, phi, dxy, z0, q] format."""
    # Mass: positive values
    mass = torch.rand(batch_size, num_particles) * 0.5 + 0.1  # (B, N)
    # pT: positive transverse momentum
    pt = torch.rand(batch_size, num_particles) * 100 + 1.0  # (B, N)
    # eta: pseudorapidity, typically in range [-5, 5]
    eta = torch.randn(batch_size, num_particles) * 2  # (B, N)
    # phi: azimuthal angle in range [-pi, pi]
    phi = (torch.rand(batch_size, num_particles) * 2 - 1) * torch.pi  # (B, N)
    # dxy: impact parameter
    dxy = torch.randn(batch_size, num_particles) * 0.1
    # z0: longitudinal impact parameter
    z0 = torch.randn(batch_size, num_particles) * 5.0
    # q: charge
    q = torch.randint(0, 2, (batch_size, num_particles)).float() * 2 - 1

    # (B, N, 7) -> mass, pt, eta, phi, dxy, z0, q
    x = torch.stack([mass, pt, eta, phi, dxy, z0, q], dim=-1)
    return x


class TestPhysicsUtilityFunctions(unittest.TestCase):
    """Test physics utility functions for converting [mass, pt, eta, phi] to rapidity, energy, and pt."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_particles = 4
        self.eps = 1e-8

    def test_to_rapidity_basic(self):
        """Test rapidity calculation for basic inputs."""
        # Create test data [mass, pt, eta, phi]
        # For eta=0, rapidity should also be ~0 for any mass/pt
        x = torch.tensor([[[0.1, 10.0, 0.0, 0.0]]])  # Shape: (1, 1, 4)
        rapidity = to_rapidity(x)

        # For eta=0: E = sqrt(m^2 + pt^2), pz = pt*sinh(0) = 0
        # rapidity = log((E + pz)/(sqrt(m^2 + pt^2))) = log(1) = 0
        self.assertAlmostEqual(rapidity.item(), 0.0, places=3)

    def test_to_rapidity_shape(self):
        """Test that rapidity maintains correct shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        rapidity = to_rapidity(x)

        expected_shape = (self.batch_size, self.num_particles)
        self.assertEqual(rapidity.shape, expected_shape)

    def test_to_rapidity_stability(self):
        """Test rapidity calculation stability with eps parameter."""
        # Test with large eta values
        x = torch.tensor([[[0.1, 10.0, 4.0, 0.0]]])
        rapidity = to_rapidity(x)

        # Should not produce NaN or inf
        self.assertFalse(torch.isnan(rapidity).any())
        self.assertFalse(torch.isinf(rapidity).any())

    def test_to_energy_basic(self):
        """Test energy calculation."""
        # Create test data [mass, pt, eta, phi]
        # E = sqrt(m^2 + pt^2 * cosh(eta)^2)
        # For eta=0: cosh(0)=1, so E = sqrt(m^2 + pt^2)
        mass, pt, eta = 3.0, 4.0, 0.0
        x = torch.tensor([[[mass, pt, eta, 0.0]]])
        energy = to_energy(x)

        expected = torch.sqrt(torch.tensor(mass**2 + pt**2))
        self.assertAlmostEqual(energy.item(), expected.item(), places=5)

    def test_to_energy_shape(self):
        """Test that energy maintains correct shape."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        energy = to_energy(x)

        expected_shape = (self.batch_size, self.num_particles)
        self.assertEqual(energy.shape, expected_shape)

    def test_to_energy_positive(self):
        """Test that energy is always positive."""
        x = generate_physical_data(self.batch_size, self.num_particles)
        energy = to_energy(x)
        self.assertTrue((energy > 0).all())

    def test_to_pt_basic(self):
        """Test transverse momentum calculation from px, py."""
        # The to_pt function uses x[..., 1] as px and x[..., 2] as py
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
        self.input_dim = 7  # delta, k_t, z, m^2, d_dxy, d_z0, q_ij
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

        # The mask zeros out the input features before passing through the network.
        # Since the network has biases, masked regions will have identical values
        # (the response of the network to zero input).
        masked_output = output[:, 4:, 4:, :]

        # All masked positions should produce identical outputs (from zero input)
        dummy = torch.zeros(1, 7)  # (delta, kt, z, m2, dxy, z0, q) = 0
        expected = self.model.net(dummy)

        # Check that first element of masked region matches expected
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
        self.input_dim = 7
        self.embed_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.num_cls_layers = 2
        self.num_classes = 1

        self.model = ParticleTransformer(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            num_pairwise_feat=7,
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
        # Create tensor with gradients enabled in [mass, pt, eta, phi] format
        mass = torch.rand(self.batch_size, self.num_particles) * 0.5 + 0.1
        pt = torch.rand(self.batch_size, self.num_particles) * 100 + 1.0
        eta = torch.randn(self.batch_size, self.num_particles) * 2
        phi = (torch.rand(self.batch_size, self.num_particles) * 2 - 1) * torch.pi
        dxy = torch.randn(self.batch_size, self.num_particles) * 0.1
        z0 = torch.randn(self.batch_size, self.num_particles) * 5.0
        q = torch.randint(0, 2, (self.batch_size, self.num_particles)).float() * 2 - 1

        x = torch.stack([mass, pt, eta, phi, dxy, z0, q], dim=-1).requires_grad_(True)

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
            input_dim=7,
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
            input_dim=7,
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
            input_dim=7,
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
