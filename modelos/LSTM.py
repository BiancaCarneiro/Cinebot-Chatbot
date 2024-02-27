import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from common import *



class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_size = output_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torch.log_softmax(tag_space, dim=1)
        return tag_scores


def train_lstm(
        num_epochs:int,
        training_data:list,
        val_data:list,
        word_to_ix:dict,
        tag_to_ix:dict,
        embedding_dim:int,
        hidden_dim:int,
        patience:int = 5
    ) -> LSTMModel:
    best_val_loss = float('inf')
    counter = 0
    vocab_size = len(word_to_ix)
    output_dim = len(tag_to_ix)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMModel(embedding_dim, hidden_dim, vocab_size, output_dim).to(device)
    
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    print(f"Treinando LSTM no device {device}")
    
    for epoch in tqdm(range(num_epochs)):  
        model.train()  # Define o modelo em modo de treinamento
        
        for sentence, tags in training_data:
            # Preparando os dados de entrada e saída
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix, device)
            targets = prepare_sequence(tags, tag_to_ix, device)  # Mapeamento das tags para índices
            targets = targets.to(device)  # Movendo para GPU, se disponível
            
            # Passo forward
            tag_scores = model(sentence_in)
            
            # Calculando a perda e retropropagando o erro
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

        # Validação
        model.eval()  # Define o modelo em modo de avaliação
        val_loss = 0
        with torch.no_grad():
            for sentence, tags in val_data:
                sentence_in = prepare_sequence(sentence, word_to_ix, device)
                targets = prepare_sequence(tags, tag_to_ix, device)
                targets = targets.to(device)
                
                # Passo forward
                tag_scores = model(sentence_in)
                val_loss += loss_function(tag_scores, targets).item()

        val_loss /= len(val_data)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break
            
    return model

def test_LSTM(
        model: LSTMModel,
        word_to_ix:dict,
        tag_to_ix:dict,
        test_data: list
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Define o modelo em modo de avaliação
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence, tags in test_data:
            inputs = prepare_sequence(sentence, word_to_ix, device)
            targets = prepare_sequence(tags, tag_to_ix, device)
            tag_scores = model(inputs)
            _, predicted_tags = torch.max(tag_scores, dim=1)
            predicted_tags = predicted_tags.cpu().numpy()
            total += targets.size(0)
            correct += sum([1 for i in range(len(predicted_tags)) if predicted_tags[i] == targets[i]])
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')