import sentencepiece_model_pb2 as model

m = model.ModelProto()
m.ParseFromString(open('tokenizer.model', 'rb').read())

tokens = ["<|SYSTEM|>", "<|USER|>", "<|ASSISTANT|>"]

for token in tokens:
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = token
    new_token.score = 0
    new_token.type = 4
    m.pieces.append(new_token)

with open("cc-tokenizer.model", "wb") as f:
    f.write(m.SerializeToString())

from sentencepiece import SentencePieceProcessor

sp = SentencePieceProcessor("cc-tokenizer.model")

print(sp.encode_as_pieces("<|SYSTEM|>"))
print(sp.encode("<|SYSTEM|>"))
print(sp.decode(sp.encode("<|SYSTEM|>")))
print(sp.decode(sp.encode("Hello <|SYSTEM|><|USER|><|ASSISTANT|>")))
