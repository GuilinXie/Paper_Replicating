import torch
import torch.nn as nn
from torchsummary import summary


PATCH_SIZE = 16
IMG_SIZE = 224
HIDDEN_D = 768
NUM_HEADS = 12
MLP_SIZE = 3072
NUM_LAYERS = 12


# 1.Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector

    Args:
        in_channels (int): Number of color channels for the input image
        patch_size (int): Size of patches to convert input image into.
        embedding_dim (int): Size of embedding to turn image into. 
    """
    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels=3,
                 patch_size=16,
                 embedding_dim=768):
        super().__init__()
        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=embedding_dim, 
                                 kernel_size=patch_size, 
                                 stride=patch_size, 
                                 padding=0)    # (batch_size, 768, 14, 14)
        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)  # (batch_size, 768, 196)
        self.patch_size = patch_size
    
    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input Image size must be divisible by patch_size"
        
        # Perform the forward pass
        x_patched = self.patcher(x)                # (batch_size, emb_n, H, W)
        x_flatten = self.flatten(x_patched)        # (batch_size, emb_n, N)   <-- N is the number of the patches, same as the length of the sequence in NLP
        x_transposed = x_flatten.permute(0, 2, 1)   # (batch_size, N, emb_n)
        return x_transposed
    
    
# 1.Create a class which subclasses nn.Module
class MultiheadsSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block for short)"""
    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim=HIDDEN_D,
                 num_heads=NUM_HEADS,
                 attn_dropout=0):
        super().__init__()
        # 3. Create the Norm Layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # 4. Create the Multihead Self-Attention(MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True
                                                    ) 
    # 5. Create a forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,   # query embeddings
                                             key=x,     # key embedding
                                             value=x,   # value embedding
                                             need_weights=False)    # do we need the weights or just the layer outputs?
        return attn_output
    

class MLPBlock(nn.Module):
    def __init__(self,
                  embedding_dim=HIDDEN_D,
                  mlp_size=MLP_SIZE,
                  dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
        
      
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim=HIDDEN_D,
                 num_heads=NUM_HEADS,
                 mlp_size=MLP_SIZE,
                 mlp_dropout=0.1,
                 attn_dropout=0):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa_block = MultiheadsSelfAttentionBlock(embedding_dim=embedding_dim,
                                                      num_heads=num_heads,
                                                      attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)      
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

# 1. Create a ViT class subclassing nn.Module        
class ViT(nn.Module):
    """Create a Vision Transformer architecture with ViT-Base hyperparameters"""
    # 2. Initialize the class with hyperparameters
    def __init__(self,
                 img_size=IMG_SIZE,                     # Training resolution (224, 224)
                 in_channels=3,                         # Number of channels in input image
                 patch_size=PATCH_SIZE,                 # Patch size
                 num_transformer_layers=NUM_LAYERS,     # Layers for ViT-Base
                 embedding_dim=HIDDEN_D,                # Hidden size D for ViT-Base
                 mlp_size=MLP_SIZE,                     # MLP size for ViT-Base
                 num_heads=NUM_HEADS,                   # Heads for ViT-Base
                 attn_dropout=0,                        # Dropout for attention projection
                 mlp_dropout=0.1,                       # Dropout for dense/MLP layers
                 embedding_dropout=0.1,                 # Dropout for patch and position embedding
                 num_classes=1000):                        # Default for ImageNet, but can be customized
        super().__init__()
        
        # 3. Make the image size is divisible by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by the patch_size"
        
        # 4. Calculate number of patches (height * width / patch ** 2)
        self.num_patches = (img_size * img_size) // patch_size ** 2
        
        # 5. Create learnable class embedding
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)    # (1, 1, embedding_dim)
        
        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim))
        
        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        # 9. Create Transformer Encoder ( We can stack Transformer Encoder blocks using nn.Sequential())
        # self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                        #    num_heads=num_heads,
                                                                        #    mlp_size=mlp_size,
                                                                        #    mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        
        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)             # (batch_size, 1, num_classes)
        )
    
    # 11. Create a forward() method
    def forward(self, x):
        # 12. Get batch size
        batch_size = x.shape[0]
        # 13. Create class token embedding and expand it to match the batch_size
        
        class_token = self.class_embedding.expand(batch_size, -1, -1)   # "-1" means to infer the dimension
        # 14. Create patch embedding
        x = self.patch_embedding(x)
        # 15. Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim=1)
        # 16. Add position embedding to patch embedding
       
        x = self.position_embedding + x         # (batch_size, 1 + self.num_patches, embedding_dim)
        # 17. Run embedding dropout
        x = self.embedding_dropout(x)
        # 18. Pass through transformer_encoder
        x = self.transformer_encoder(x)         # (batch_size, 1 + self.num_patches, embedding_dim)
        
        # 19. Put 0 index logit through classifier
        x = x[:, 0]               # (batch_size, embedding_dim)
        x = self.classifier(x)    # (batch_size, num_classes)
         
        return x

# # Create the same as above with torch.nn.TransformerEncoderLayer()
# torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_D,
#                                                             nhead=NUM_HEADS,
#                                                             dim_feedforward=MLP_SIZE,
#                                                             dropout=0.1,        # Amount of dropout for dense layer for Vit-Base
#                                                             activation="gelu",
#                                                             batch_first=True,
#                                                             norm_first=True)     # Normalize first or after MSA/MLP layers?

# # Create the same using torch built-ins
# torch_transformer_encoder = nn.TransformerEncoder(encoder_lay=torch_transformer_encoder_layer,
#                             num_layers=NUM_LAYERS)

