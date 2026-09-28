"""The independent rotary architecture obeys the same fixed causal contract."""
import torch
import test_contract
from student_rope import build_model


class RotaryContractTests(test_contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                      context=256, ffn_hidden=96,
                                      residual_dropout=.2, embedding_dropout=.1)).eval()

    def test_all_prefixes_match(self):
        x = torch.randint(2048, (2, 16))
        with torch.no_grad():
            full = self.model.predict_log_probs(x)
            for length in range(1, 17):
                torch.testing.assert_close(self.model.predict_log_probs(x[:, :length]),
                                           full[:, :length], atol=1e-5, rtol=1e-5)
