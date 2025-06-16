import torch
from dataloader import tokenize
from model import SentimentModel

# Hàm dự đoán sentiment
def predict_sentiment(text, model, vocab, max_len=100):
    model.eval()
    tokens = tokenize(text)
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    encoded = encoded[:max_len] + [vocab["<PAD>"]] * (max_len - len(encoded))
    input_tensor = torch.tensor([encoded])

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    sentiments = ["Negative", "Neutral", "Positive"]
    return sentiments[predicted.item()]

# Hàm lấy từ khóa dựa trên attention
# def get_keywords(text, model, vocab, max_len=100, top_k=3):
#     model.eval()
#     tokens = tokenize(text)
#     encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
#     encoded = encoded[:max_len] + [vocab["<PAD>"]] * (max_len - len(encoded))
#     input_tensor = torch.tensor([encoded])
#
#     with torch.no_grad():
#         _, attn_weights = model(input_tensor)  # Lấy attention weights
#         attn_weights = attn_weights.squeeze(0).squeeze(-1)  # [seq_len]
#
#     # Xác định từ khóa dựa trên trọng số lớn nhất
#     token_weights = list(zip(tokens, attn_weights.tolist()))
#     keywords = sorted(token_weights, key=lambda x: x[1], reverse=True)[:top_k]
#     return [kw[0] for kw in keywords]  # Chỉ trả về từ khóa

# def get_keywords(text, model, vocab, max_len=100, top_k=3):
#     tokens = tokenize(text)
#     encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
#     encoded = encoded[:max_len] + [vocab["<PAD>"]] * (max_len - len(encoded))
#     input_tensor = torch.tensor([encoded], dtype=torch.long)
#
#     # Đảm bảo mô hình ở chế độ đánh giá
#     model.eval()
#
#     # Lấy embedding và kích hoạt gradient
#     embedding_output = model.embedding(input_tensor)  # Lấy embedding (FloatTensor)
#     embedding_output.retain_grad()  # Đảm bảo gradient được lưu trên embedding_output
#
#     # Chạy qua LSTM và Fully Connected Layer
#     lstm_out, _ = model.lstm(embedding_output)  # Chạy qua LSTM
#     lstm_out = lstm_out.mean(dim=1)  # Mean pooling
#     output = model.fc(lstm_out)  # Chạy qua Fully Connected Layer
#     pred_label = output.argmax(dim=1)  # Lấy nhãn dự đoán
#
#     # Tính gradient của đầu ra với embedding
#     model.zero_grad()
#     output[0, pred_label].backward(retain_graph=True)  # Backpropagation
#     grads = embedding_output.grad[0]  # Gradient của embedding
#
#     # Tính mức độ ảnh hưởng của từng từ
#     token_importance = [(tokens[i], grads[i].abs().sum().item()) for i in range(len(tokens))]
#     keywords = sorted(token_importance, key=lambda x: x[1], reverse=True)[:top_k]
#     return [kw[0] for kw in keywords]

def get_keywords(text, model, vocab, max_len=500, top_k=3):
    tokens = tokenize(text)
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    encoded = encoded[:max_len] + [vocab["<PAD>"]] * (max_len - len(encoded))
    input_tensor = torch.tensor([encoded], dtype=torch.long)

    # Đảm bảo mô hình ở chế độ đánh giá
    model.eval()

    with torch.no_grad():
        # Lấy Attention từ Transformer Encoder
        seq_len = input_tensor.size(1)
        position_ids = torch.arange(0, seq_len, device=input_tensor.device).unsqueeze(0)
        embedded = model.embedding(input_tensor) + model.position_embedding(position_ids)

        transformer_out = model.transformer(embedded.permute(1, 0, 2))  # [seq_len, batch_size, embed_dim]
        attention_weights = torch.softmax(transformer_out.mean(dim=2), dim=0)  # Attention weights

    # Chọn các từ quan trọng
    token_importance = [(tokens[i], attention_weights[i].item()) for i in range(len(tokens))]
    keywords = sorted(token_importance, key=lambda x: x[1], reverse=True)[:top_k]
    return [kw[0] for kw in keywords]


# Hàm tải mô hình đã lưu
def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = SentimentModel(
        checkpoint["vocab_size"],
        checkpoint["embed_dim"],
        checkpoint["hidden_dim"],
        checkpoint["output_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    vocab = checkpoint["vocab"]
    return model, vocab

# Giao diện console
if __name__ == "__main__":
    # Tải mô hình và từ vựng
    model_path = "sentiment_model.pth"
    model, vocab = load_model(model_path)

    print("Model has been loaded!")
    print("Enter your review to predict: (or Enter 'exit' to quit).")

    while True:
        review = input("Enter your revieư: ")
        if review.lower() == "exit":
            print("End of program.")
            break

        # Dự đoán sentiment
        sentiment = predict_sentiment(review, model, vocab)

        # Lấy từ khóa
        keywords = get_keywords(review, model, vocab)

        # In kết quả
        print(f"Sentiment: {sentiment}")
        print(f"Keywords: {', '.join(keywords)}")
        print("-" * 50)
