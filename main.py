# from torch.optim import Adam
# from torch.nn import CrossEntropyLoss
# from dataloader import build_vocab, create_dataloader, load_data
# from model import SentimentModel
#
# file_path = "./data/yelp_academic_dataset_review.json"
#
# # Chia dữ liệu train/test
# texts_train, texts_test, labels_train, labels_test = load_data(file_path)
#
# # Xây dựng từ vựng từ tập train
# vocab = build_vocab(texts_train, max_vocab_size=10000)
#
# # Tạo DataLoader
# train_loader = create_dataloader(texts_train, labels_train, vocab, max_len=100, batch_size=32)
# test_loader = create_dataloader(texts_test, labels_test, vocab, max_len=100, batch_size=32)
#
# # Khởi tạo mô hình
# vocab_size = len(vocab) + 1
# embed_dim = 50
# hidden_dim = 128
# output_dim = 3
# model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)
#
# # Cấu hình huấn luyện
# criterion = CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=0.001)
#
# # Hàm đánh giá mô hình
# def evaluate(model, dataloader, criterion):
#     model.eval()  # Chuyển sang chế độ đánh giá
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():  # Không tính gradient trong chế độ đánh giá
#         for batch in dataloader:
#             inputs, targets = batch
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#
#             # Tính toán độ chính xác
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == targets).sum().item()
#             total += targets.size(0)
#
#     accuracy = correct / total
#     return total_loss / len(dataloader), accuracy
#
# # Huấn luyện
# epochs = 10
# for epoch in range(epochs):
#     # Chế độ huấn luyện
#     model.train()
#     total_loss = 0
#
#     for batch in train_loader:
#         inputs, targets = batch
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     # Đánh giá trên tập test
#     test_loss, test_accuracy = evaluate(model, test_loader, criterion)
#
#     # Hiển thị kết quả
#     print(f"Epoch {epoch + 1}/{epochs}")
#     print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
#     print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
#
# # Lưu mô hình và từ vựng
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "vocab": vocab,
#     "vocab_size": vocab_size,
#     "embed_dim": embed_dim,
#     "hidden_dim": hidden_dim,
#     "output_dim": output_dim
# }, "sentiment_model.pth")
#
#
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from dataloader import build_vocab, create_dataloader, load_data
from model import SentimentModel  # Cập nhật mô hình để dùng Transformer

file_path = "/Users/nhh6801/Documents/CDNC1-2-3/CD1/CDNC/data/yelp_academic_dataset_review.json"

# Chia dữ liệu train/test
texts_train, texts_test, labels_train, labels_test = load_data(file_path)

# Xây dựng từ vựng từ tập train
vocab = build_vocab(texts_train, max_vocab_size=10000)

# Tạo DataLoader
train_loader = create_dataloader(texts_train, labels_train, vocab, max_len=500, batch_size=32)
test_loader = create_dataloader(texts_test, labels_test, vocab, max_len=500, batch_size=32)

# Khởi tạo mô hình Transformer
vocab_size = len(vocab) + 1
embed_dim = 64  # Chia hết cho 4 (num_heads)
hidden_dim = 128
output_dim = 3
num_heads = 4  # Số lượng heads trong Multi-head Attention
num_layers = 2  # Số lượng Transformer Encoder Layers

model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim, num_heads=num_heads, num_layers=num_layers)

# Cấu hình huấn luyện
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Hàm đánh giá mô hình
def evaluate(model, dataloader, criterion):
    model.eval()  # Chuyển sang chế độ đánh giá
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Không tính gradient trong chế độ đánh giá
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Tính toán độ chính xác
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# Huấn luyện
epochs = 50
for epoch in range(epochs):
    # Chế độ huấn luyện
    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Đánh giá trên tập test
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    # Hiển thị kết quả
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Lưu mô hình và từ vựng
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab,
    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
    "num_heads": num_heads,
    "num_layers": num_layers
}, "sentiment_model.pth")
print("Mô hình đã được lưu!")
