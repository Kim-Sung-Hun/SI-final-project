import torch
import torch.nn as nn

# 입력 데이터 생성
batch_size = 32
sequence_length = 1
embed_dim = 512
num_heads = 8

q = torch.randn(batch_size, sequence_length, embed_dim)
k = torch.randn(batch_size, sequence_length, embed_dim)
v = torch.randn(batch_size, sequence_length, embed_dim)

# MultiheadAttention 모듈 생성
attn = nn.MultiheadAttention(embed_dim, num_heads)

# 어텐션 연산 수행
output, attn_output_weights = attn(q, k, v)

# 출력 확인
print(output.shape)  # torch.Size([32, 10, 512])
print(attn_output_weights.shape)  # torch.Size([8, 32, 10, 10])


