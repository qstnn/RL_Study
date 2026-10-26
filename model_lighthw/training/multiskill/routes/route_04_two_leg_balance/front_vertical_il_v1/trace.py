import torch
import numpy as np

class BufferedPhysicsTrace:
    """Bound GPU trace memory while retaining every 1 ms physical sample."""

    def __init__(self, capacity: int = 20_000):
        self.capacity = capacity
        self.buffer = None
        self.chunks = None
        self.used = 0
        self.samples = 0

    @torch.no_grad()
    def append(self, row: dict[str, torch.Tensor]):
        if self.buffer is None:
            self.buffer = {key: torch.empty((self.capacity, *value.shape),
                             dtype=value.dtype, device=value.device) for key, value in row.items()}
            self.chunks = {key: [] for key in row}
        if set(row) != set(self.buffer):
            raise RuntimeError("1 kHz physics metric fields changed during evaluation")
        if self.used == self.capacity:
            self.flush()
        for key, value in row.items():
            if value.shape != self.buffer[key].shape[1:] or value.dtype != self.buffer[key].dtype:
                raise RuntimeError("1 kHz physics metric shape/dtype changed: " + key)
            self.buffer[key][self.used].copy_(value)
        self.used += 1
        self.samples += 1

    def flush(self):
        if not self.used:
            return
        for key, value in self.buffer.items():
            self.chunks[key].append(value[:self.used].cpu().numpy().copy())
        self.used = 0

    def arrays(self):
        self.flush()
        if not self.chunks:
            raise RuntimeError("No physical samples captured")
        result = {key: np.concatenate(parts, axis=0) for key, parts in self.chunks.items()}
        if any(len(value) != self.samples for value in result.values()):
            raise RuntimeError("Buffered physical trace length mismatch")
        return result
