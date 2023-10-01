import paddle

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

# out = masked_fill(x, mask, 2)

def reshape_heads_to_batch_dim(self, *args):
    tensor = args[0]
    batch_size, seq_len, dim = tensor.shape
    head_size = self.heads
    tensor = tensor.reshape((batch_size, seq_len, head_size, dim // head_size))
    tensor = paddle.transpose(tensor, (0, 2, 1, 3))
    tensor = tensor.reshape((batch_size * head_size, seq_len, dim // head_size))
    return tensor

def reshape_batch_dim_to_heads(self, *args):
    tensor = args[0]
    batch_size, seq_len, dim = tensor.shape
    head_size = self.heads
    tensor = tensor.reshape((batch_size // head_size, head_size, seq_len, dim))
    tensor = paddle.transpose(tensor, (0, 2, 1, 3))
    tensor = tensor.reshape((batch_size // head_size, seq_len, dim * head_size))
    return tensor