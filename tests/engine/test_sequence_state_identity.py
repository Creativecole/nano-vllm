import pickle

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_sequence_pickle_preserves_seq_id_for_worker_state_lookup():
    sequence = Sequence([1, 2, 3], SamplingParams(temperature=0.0))
    restored = pickle.loads(pickle.dumps(sequence))
    assert restored.seq_id == sequence.seq_id
    assert restored.last_token == sequence.last_token
    assert restored.num_tokens == sequence.num_tokens
