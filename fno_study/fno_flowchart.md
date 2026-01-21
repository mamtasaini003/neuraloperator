# SimpleFNO Flowchart

```mermaid
graph TD
    %% Define styles
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef important fill:#fff3e0,stroke:#e65100,stroke-width:3px
    
    %% Input
    INPUT["Input: x\n(B, C_in, D1, ..., D)"]
    class INPUT input
    
    %% SimpleFNO Main Flow
    POS_EMB[Positional Embedding<br/>Purpose: Add spatial coordinates<br/>Input: (B, C_in, D1, ..., DN)<br/>Output: (B, C_in+n_dim, D1, ..., DN)]
    class POS_EMB process
    
    LIFT[Lifting Layer<br/>Purpose: Project to hidden dimension<br/>Input: (B, C_in+n_dim, D1, ..., DN)<br/>Output: (B, hidden_channels, D1, ..., DN)]
    class LIFT process
    
    PAD[Domain Padding<br/>Purpose: Handle boundary conditions<br/>Input: (B, hidden_channels, D1, ..., DN)<br/>Output: (B, hidden_channels, D1+pad, ..., DN+pad)]
    class PAD process
    
    %% FNO Blocks
    FNO_BLOCKS[FNO Blocks × n_layers<br/>Purpose: Core spectral processing<br/>Input: (B, hidden_channels, D1, ..., DN)<br/>Output: (B, hidden_channels, D1_out, ..., DN_out)]
    class FNO_BLOCKS important
    
    %% FNOBlock detailed flow
    subgraph FNO_BLOCK_DETAIL [FNOBlock Internal Flow]
        direction LR
        
        PRE_NORM[Preactivation Norm<br/>Purpose: Stabilize training<br/>Input: (B, C, D1, ..., DN)<br/>Output: Same, normalized]
        
        SPECTRAL[Spectral Convolution ⭐⭐⭐⭐⭐<br/>Paurpose: Global FFT processing<br/>Input: (B, C, D1, ..., DN)<br/>Output: (B, C, D1_out, ..., DN_out)]
        class SPECTRAL important
        
        SKIP[Skip Connection ⭐⭐⭐⭐⭐<br/>Purpose: Local information<br/>Input: (B, C, D1, ..., DN)<br/>Output: (B, C, D1_out, ..., DN_out)]
        class SKIP important
        
        COMBINE[Combine<br/>Purpose: Merge paths<br/>Input: Two tensors<br/>Output: (B, C, D1_out, ..., DN_out)]
        
        POST_NORM[Post-activation Norm<br/>Purpose: Stabilize training<br/>Input: (B, C, D1_out, ..., DN_out)<br/>Output: Same, normalized]
        
        ACTIVATION[Activation ⭐⭐⭐⭐<br/>Purpose: Non-linearity<br/>Input: (B, C, D1_out, ..., DN_out)<br/>Output: Same, activated]
        
        CHANNEL_MLP[Channel MLP ⭐⭐⭐<br/>Purpose: Channel mixing<br/>Input: (B, C, D1_out, ..., DN_out)<br/>Output: Same, mixed]
        
        PRE_NORM --> SPECTRAL
        PRE_NORM --> SKIP
        SPECTRAL --> COMBINE
        SKIP --> COMBINE
        COMBINE --> POST_NORM
        POST_NORM --> ACTIVATION
        ACTIVATION --> CHANNEL_MLP
    end
    
    UNPAD[Unpadding<br/>Purpose: Remove padding<br/>Input: (B, hidden_channels, D1+pad, ..., DN+pad)<br/>Output: (B, hidden_channels, D1, ..., DN)]
    class UNPAD process
    
    PROJ[Projection Layer ⭐⭐⭐⭐<br/>Purpose: Project to output<br/>Input: (B, hidden_channels, D1, ..., DN)<br/>Output: (B, C_out, D1, ..., DN)]
    class PROJ process
    
    FINAL_OUTPUT[Final Output<br/>(B, C_out, D1, ..., DN)]
    class FINAL_OUTPUT output
    
    %% Connections
    INPUT --> POS_EMB
    POS_EMB --> LIFT
    LIFT --> PAD
    PAD --> FNO_BLOCKS
    FNO_BLOCKS --> FNO_BLOCK_DETAIL
    FNO_BLOCK_DETAIL --> UNPAD
    UNPAD --> PROJ
    PROJ --> FINAL_OUTPUT
    
    %% Legend
    subgraph LEGEND [Legend]
        direction LR
        INPUT_EX[Input Data]
        class INPUT_EX input
        PROCESS_EX[Processing Step]
        class PROCESS_EX process
        OUTPUT_EX[Output Data]
        class OUTPUT_EX output
        IMPORTANT_EX[Critical Step ⭐⭐⭐⭐⭐]
        class IMPORTANT_EX important
    end
```

## SimpleFNO Architecture Flow

### Main Data Pipeline

1. **Input**: `(B, C_in, D1, ..., DN)`
   - Batch size `B`, input channels `C_in`, spatial dimensions `D1..DN`

2. **Positional Embedding** ⭐⭐⭐⭐
   - **Purpose**: Add spatial coordinate information
   - **Input**: `(B, C_in, D1, ..., DN)`
   - **Output**: `(B, C_in + n_dim, D1, ..., DN)`

3. **Lifting Layer** ⭐⭐⭐⭐
   - **Purpose**: Project to hidden dimension
   - **Input**: `(B, C_in + n_dim, D1, ..., DN)`
   - **Output**: `(B, hidden_channels, D1, ..., DN)`

4. **Domain Padding** ⭐⭐⭐
   - **Purpose**: Handle boundary conditions
   - **Input**: `(B, hidden_channels, D1, ..., DN)`
   - **Output**: `(B, hidden_channels, D1+pad, ..., DN+pad)`

5. **FNO Blocks** ⭐⭐⭐⭐⭐
   - **Purpose**: Core spectral processing (repeated `n_layers` times)
   - **Input**: `(B, hidden_channels, D1, ..., DN)`
   - **Output**: `(B, hidden_channels, D1_out, ..., DN_out)`

6. **Unpadding** ⭐⭐
   - **Purpose**: Remove padding
   - **Input**: `(B, hidden_channels, D1+pad, ..., DN+pad)`
   - **Output**: `(B, hidden_channels, D1, ..., DN)`

7. **Projection Layer** ⭐⭐⭐⭐
   - **Purpose**: Project to output channels
   - **Input**: `(B, hidden_channels, D1, ..., DN)`
   - **Output**: `(B, C_out, D1, ..., DN)`

8. **Final Output**: `(B, C_out, D1, ..., DN)`
   - Batch size `B`, output channels `C_out`, spatial dimensions `D1..DN`

### FNOBlock Internal Flow

Each FNOBlock processes data through these steps:

1. **Preactivation Norm** ⭐⭐⭐
   - **Purpose**: Stabilize training
   - **Input/Output**: Same dimensions, normalized

2. **Spectral Convolution** ⭐⭐⭐⭐⭐
   - **Purpose**: Global FFT-based processing
   - **Input**: `(B, C, D1, ..., DN)`
   - **Output**: `(B, C, D1_out, ..., DN_out)`

3. **Skip Connection** ⭐⭐⭐⭐⭐
   - **Purpose**: Preserve local information
   - **Input**: `(B, C, D1, ..., DN)`
   - **Output**: `(B, C, D1_out, ..., DN_out)`

4. **Combine**
   - **Purpose**: Merge spectral and local paths
   - **Input**: Two tensors of same shape
   - **Output**: `(B, C, D1_out, ..., DN_out)`

5. **Post-activation Norm** ⭐⭐⭐
   - **Purpose**: Stabilize training
   - **Input/Output**: Same dimensions, normalized

6. **Activation** ⭐⭐⭐⭐
   - **Purpose**: Apply non-linearity
   - **Input/Output**: Same dimensions, activated

7. **Channel MLP** ⭐⭐⭐
   - **Purpose**: Channel mixing and enhancement
   - **Input/Output**: Same dimensions, enhanced channels

### Key Features

- **Multi-resolution Support**: `output_scaling_factor` enables progressive resolution changes
- **Complex Data Handling**: Optional complex-valued processing
- **Flexible Normalization**: Support for batch, instance, group, or AdaIN
- **Residual Connections**: Skip connections in both FNO and channel MLP paths
- **Spectral Efficiency**: FFT-based global operations with linear complexity

### Performance Characteristics

- **Memory**: O(N × C) where N = spatial points, C = channels
- **Compute**: Linear in spatial resolution, quadratic in channels
- **Key Advantage**: Global receptive field with local complexity