from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

dataset = load_dataset("roneneldan/TinyStories")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.model_max_length = 2048

all_token_ids = []

# Some Variables to control the model
context_length = 256
embedding_dimension = 256
number_of_blocks = 2
epochs = 0

training_stories = dataset["train"].select(range(2500))
validation_stories = dataset["validation"].select(range(100)) # Not using this cuz /shrug

# Dont understand this very well
class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.query = torch.nn.Linear(embedding_dimension, embedding_dimension)
        self.key = torch.nn.Linear(embedding_dimension, embedding_dimension)
        self.value = torch.nn.Linear(embedding_dimension, embedding_dimension)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, embedding_dimension * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dimension * 4, embedding_dimension)
        )

        self.norm1 = torch.nn.LayerNorm(embedding_dimension)
        self.norm2 = torch.nn.LayerNorm(embedding_dimension)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)
        scores = scores / torch.sqrt(torch.tensor(embedding_dimension))

        sequence_length = x.shape[1]

        mask = torch.tril(torch.ones( sequence_length, sequence_length, device=device)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention_output = attention_weights @ value

        # Im'ma be honest i have no idea how this works
        # Residual connection + normalization
        x = x + attention_output
        x = self.norm1(x)

        # Feed-forward
        ffn_output = self.ffn(x)

        # Residual connection + normalization
        x = x + ffn_output
        x = self.norm2(x)

        return x

class TinyStoriesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TinyStoriesModel"

        self.token_embedding = torch.nn.Embedding(num_embeddings=len(tokenizer), embedding_dim=embedding_dimension).to(device)

        self.position_embedding = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dimension).to(device)

        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock() for _ in range(number_of_blocks)]).to(device)

        self.output_layer = torch.nn.Linear(embedding_dimension, len(tokenizer)).to(device)

    def forward(self, input_tokens):
        embedded = self.token_embedding(input_tokens)

        positions = torch.arange(input_tokens.shape[1], device=input_tokens.device)

        embedded = embedded + self.position_embedding(positions)

        transformer_output = embedded

        for block in self.transformer_blocks:
            transformer_output = block(transformer_output)

        logits = self.output_layer(transformer_output)

        return logits
    
for story in training_stories:
    story_token_ids = tokenizer(story["text"])
    all_token_ids.extend(story_token_ids["input_ids"])
    all_token_ids.append(tokenizer.eos_token_id)

inputs = []
targets = []

# Gets positions in a list starting from 0 to the length of the all_tokens_list
# in batches of context_length
for position in range(0, len(all_token_ids), context_length):

    # gets a set of tokens starting at list position and going to list position + context_length 
    input_sequence = all_token_ids[position:position + context_length]
    # Does the same thing as the input_sequence except it adds 1 to the starting and ending positions
    target_sequence = all_token_ids[position + 1:position + context_length + 1]

    # Makes sure that the input and target lists are the same length
    if len(input_sequence) == context_length and len(target_sequence) == context_length:
        inputs.append(input_sequence)
        targets.append(target_sequence)

# Creating the tensors after the loop
inputs_tensor = torch.tensor(inputs)
targets_tensor = torch.tensor(targets)

# Creating a dataset
training_dataset = TensorDataset(inputs_tensor, targets_tensor)

loader = DataLoader(
    training_dataset,
    batch_size=8,
    shuffle=True
)

model = TinyStoriesModel().to(device)

transformer_blocks = model.transformer_blocks
token_embedding = model.token_embedding
position_embedding = model.position_embedding
output_layer = model.output_layer
transformer = model.transformer_blocks

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
try:
    for epoch in range(epochs):
        total_loss = 0

        for training, target in loader:

            # Move batch to GPU
            training = training.to(device)
            target = target.to(device)

            logits = model(training)
            logits = logits.transpose(1, 2)

            loss = loss_function(logits, target)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss:.10f}")

        average_loss = total_loss / len(loader)

        print(
            f"Epoch: {epoch} "
            f"Average loss: {average_loss:.4f}"
        )

        torch.save(model.state_dict(), f"models/{model.name}.pt")
        print("Saving Model to: /models/" + model.name + ".pt" )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Model Parameter Count: ", parameter_count)

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current model...")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, "models/TinyStoriesModel_checkpoint.pt")

    print("Checkpoint saved.")

# This is for prompting the model, this is TEMPORARY!
# all of this will get moved to a separate script.
model.load_state_dict(torch.load("models/TinyStoriesModel.pt", map_location=device))
model.to(device)
model.eval()

prompt = "Lily was a little girl"

generated_tokens = tokenizer(prompt)["input_ids"]

# Loop through response generation to get our response tokens
for _ in range(256):

    model_input = torch.tensor([generated_tokens[-context_length:]], device=device)

    # Same stuff as the training loop except instead of training on the training data 
    # we simply give it out prompt then take the most likely next word. 
    with torch.no_grad():

        logits = model(model_input)

        # Get the scores for the final token
        next_token_logits = logits[:, -1, :]

        # Pick the token with the highest score
        next_token = torch.argmax(next_token_logits, dim=-1)

        next_token_id = next_token.item()

    generated_tokens.append(next_token_id)

    if next_token_id == tokenizer.eos_token_id:
        break

# Print the most response by decoding the response tokens
print(tokenizer.decode(generated_tokens))