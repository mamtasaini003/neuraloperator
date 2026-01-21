flowchart TD

%% =========================
%% Style definitions
%% =========================
classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
classDef important fill:#fff3e0,stroke:#e65100,stroke-width:3px

%% =========================
%% Input
%% =========================
INPUT["Input: x\n(B, C_in, D1, ..., DN)"]
class INPUT input

%% =========================
%% SimpleFNO main pipeline
%% =========================
POS_EMB["Positional Embedding\nPurpose: Add spatial coordinates\nInput: (B, C_in, D1, ..., DN)\nOutput: (B, C_in + n_dim, D1, ..., DN)"]
class POS_EMB process

LIFT["Lifting Layer\nPurpose: Project to hidden dimension\nInput: (B, C_in + n_dim, D1, ..., DN)\nOutput: (B, hidden_channels, D1, ..., DN)"]
class LIFT process

PAD["Domain Padding\nPurpose: Handle boundary conditions\nInput: (B, hidden_channels, D1, ..., DN)\nOutput: (B, hidden_channels, D1+pad, ..., DN+pad)"]
class PAD process

FNO_BLOCKS["FNO Blocks x n_layers\nPurpose: Core spectral processing\nInput: (B, hidden_channels, D1, ..., DN)\nOutput: (B, hidden_channels, D1_out, ..., DN_out)"]
class FNO_BLOCKS important

%% =========================
%% FNOBlock internal structure
%% =========================
subgraph FNO_BLOCK_DETAIL ["FNOBlock Internal Flow"]
    direction LR

    PRE_NORM["Preactivation Norm\nPurpose: Stabilize training\nInput: (B, C, D1, ..., DN)\nOutput: Same, normalized"]

    SPECTRAL["Spectral Convolution\nPurpose: Global FFT processing\nInput: (B, C, D1, ..., DN)\nOutput: (B, C, D1_out, ..., DN_out)"]
    class SPECTRAL important

    SKIP["Skip Connection\nPurpose: Local information\nInput: (B, C, D1, ..., DN)\nOutput: (B, C, D1_out, ..., DN_out)"]
    class SKIP important

    COMBINE["Combine\nPurpose: Merge paths\nInput: Two tensors\nOutput: (B, C, D1_out, ..., DN_out)"]

    POST_NORM["Post-activation Norm\nPurpose: Stabilize training\nInput: (B, C, D1_out, ..., DN_out)\nOutput: Same, normalized"]

    ACTIVATION["Activation\nPurpose: Non-linearity\nInput: (B, C, D1_out, ..., DN_out)\nOutput: Same, activated"]

    CHANNEL_MLP["Channel MLP\nPurpose: Channel mixing\nInput: (B, C, D1_out, ..., DN_out)\nOutput: Same, mixed"]

    PRE_NORM --> SPECTRAL
    PRE_NORM --> SKIP
    SPECTRAL --> COMBINE
    SKIP --> COMBINE
    COMBINE --> POST_NORM
    POST_NORM --> ACTIVATION
    ACTIVATION --> CHANNEL_MLP
end

%% =========================
%% Output stages
%% =========================
UNPAD["Unpadding\nPurpose: Remove padding\nInput: (B, hidden_channels, D1+pad, ..., DN+pad)\nOutput: (B, hidden_channels, D1, ..., DN)"]
class UNPAD process

PROJ["Projection Layer\nPurpose: Project to output channels\nInput: (B, hidden_channels, D1, ..., DN)\nOutput: (B, C_out, D1, ..., DN)"]
class PROJ process

FINAL_OUTPUT["Final Output\n(B, C_out, D1, ..., DN)"]
class FINAL_OUTPUT output

%% =========================
%% Connections (THIS creates the flowchart)
%% =========================
INPUT --> POS_EMB --> LIFT --> PAD --> FNO_BLOCKS
FNO_BLOCKS --> FNO_BLOCK_DETAIL
FNO_BLOCK_DETAIL --> UNPAD --> PROJ --> FINAL_OUTPUT


%% =========================
%% Legend
%% =========================
subgraph LEGEND ["Legend"]
    direction LR
    L1["Input Data"]
    class L1 input
    L2["Processing Step"]
    class L2 process
    L3["Critical Step"]
    class L3 important
    L4["Output Data"]
    class L4 output
end
