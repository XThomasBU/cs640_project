

class CNNTransformer(nn.Module):
    def _init_(self, num_patches, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(CNNTransformer, self)._init_()
        # ... (other initializations remain the same)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(128, d_model, kernel_size=3, padding=1),  # Final convolution to get to d_model channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        )
        # Positional embeddings for the patches
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches, d_model))
        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        # Attention-based pooling layer
        self.attention_pool = nn.Linear(d_model, 1)
        self.fc = nn.Linear(d_model, 5)  # Assuming 10 is the number of classes

    
    def forward(self, batch_of_patches):
        # Process each image's patches separately
        all_image_outputs = []
        for image_patches in batch_of_patches:  # image_patches is a list of patches for one image
            # Process 4 patches at a time to reduce memory usage
            batch_size = 8
            conv_outputs = []
            for i in range(0, len(image_patches), batch_size):
                batch_patches = torch.stack(image_patches[i:i+batch_size])
                conv_output = self.shared_conv(batch_patches)
                conv_output = conv_output.view(batch_patches.size(0), -1)
                conv_outputs.append(conv_output)
                del batch_patches
                del conv_output
                gc.collect()

            # Concatenate the outputs for all patches of this image
            conv_seq = torch.cat(conv_outputs, dim=0)
            conv_seq = conv_seq.unsqueeze(0) 
            del conv_outputs
            # Add positional embeddings
            if conv_seq.size(0) > self.positional_embeddings.size(1):
                raise ValueError("Number of patches exceeds positional embeddings size.")
            conv_seq += self.positional_embeddings[:, :conv_seq.size(0), :]

            # Pass through transformer encoder
            transformer_output = self.transformer_encoder(conv_seq)
            del conv_seq
            gc.collect()
            # Attention pooling
            attn_weights = torch.softmax(self.attention_pool(transformer_output).squeeze(-1), dim=1)
            pooled_output = einsum('bn,bnd->bd', attn_weights, transformer_output)

            all_image_outputs.append(pooled_output)

        # Combine the outputs for all images in the batch
        batch_output = torch.stack(all_image_outputs, dim=0)

        # Apply fully connected layer to the pooled output for each image
        out = self.fc(batch_output)

        return out.squeeze(1)