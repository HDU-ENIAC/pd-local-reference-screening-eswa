import unittest
import numpy as np
import torch
from pd_screening.model import LocalReferenceNet, LogPowerSTFT, normalize_reference
from pd_screening.calibration import threshold


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_and_gradients(self):
        torch.manual_seed(1)
        net = LocalReferenceNet()
        self.assertEqual(sum(p.numel() for p in net.parameters()), 163329)
        tf = LogPowerSTFT()
        x, r = tf(torch.randn(2,512)), tf(torch.randn(16,512)).reshape(2,8,32,128)
        self.assertEqual(tuple(net.encoder(x[:,None]).shape), (2,24,8,128))
        dense = net(x,r)
        self.assertEqual(tuple(dense.shape), (2,512))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(dense[:,32:448].amax(1),torch.tensor([0.,1.]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))
        self.assertFalse(any(m.training for m in net.modules() if isinstance(m,torch.nn.BatchNorm2d)))
        torch.testing.assert_close(net(x,r),net(x,r.flip(1)),atol=1e-5,rtol=1e-5)

    def test_scale_depends_only_on_refs(self):
        rng = np.random.default_rng(123)
        refs, x = rng.normal(size=(8,512)).astype('float32'), rng.normal(size=(3,512)).astype('float32')
        _, _, a = normalize_reference(x,refs)
        _, _, b = normalize_reference(x*1e3,refs)
        self.assertEqual(a,b)
        self.assertEqual(normalize_reference(x,np.zeros_like(refs))[2],1e-4)

    def test_stft_against_numpy(self):
        x = np.random.default_rng(3).normal(size=(2,512)).astype('float32')
        tf = LogPowerSTFT()
        padded = np.pad(x.astype(float),((0,0),(32,32)))
        frames = np.stack([padded[:,i:i+64] for i in range(0,512,4)],-1)
        wanted = np.log1p(abs(np.fft.rfft(frames*tf.window.numpy()[None,:,None],axis=1)[:,1:])**2)
        np.testing.assert_allclose(tf(torch.from_numpy(x)).numpy(),wanted,atol=2e-6,rtol=2e-6)

    def test_calibration(self):
        self.assertTrue(np.isinf(threshold(np.arange(32),.01)))
        self.assertEqual(threshold(np.arange(32),.05),31)
        self.assertEqual(threshold(np.arange(128),.05),122)
        self.assertEqual(threshold(np.arange(128),.01),127)
        self.assertFalse(np.any(np.ones(10)>threshold(np.ones(32),.05)))


if __name__ == '__main__':
    unittest.main()
