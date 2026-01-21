# Spectral Convolution Dimension Analysis

## Overview
Spectral convolution is the core operation in FNO that performs global convolution via FFT. Here's the complete dimension transformation:

## Input to Output Dimension Flow

### 1. Input Tensor
```
x: (B, C_in, d1, d2, ..., dN)
```
- **B**: Batch size
- **C_in**: Input channels  
- **d1, d2, ..., dN**: Spatial dimensions (N = number of dimensions)
- **Total memory**: B × C_in × d1 × d2 × ... × dN

### 2. Forward FFT Transform
**Location**: `spectral_convolution.py:522-533`

#### For Real Data (most common):
```python
x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
```
```
x_fft: (B, C_in, d1, d2, ..., dN//2 + 1)
```
- **Last dimension truncated** due to FFT redundancy
- **Complex tensor**: Each element now has real + imaginary parts
- **Memory**: ~2 × B × C_in × d1 × d2 × ... × (dN//2 + 1)

#### For Complex Data:
```python
x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
```
```
x_fft: (B, C_in, d1, d2, ..., dN)
```
- **All dimensions preserved** for full complex FFT
- **Complex tensor**: All frequency components
- **Memory**: ~2 × B × C_in × d1 × d2 × ... × dN

### 3. FFT Shift (multi-dimensional only)
**Location**: `spectral_convolution.py:532-533`

```python
x = torch.fft.fftshift(x, dim=dims_to_fft_shift)
```
```
x_shifted: Same dimensions as x_fft
```
- **Purpose**: Center zero frequency
- **No dimension change**: Only reorganizes frequency layout

### 4. Mode Truncation/Indexing
**Location**: `spectral_convolution.py:552-560`

#### Input Slicing: `_get_input_indices()`
```
x_truncated: (B, C_in, m1, m2, ..., mN)
```
- **mi**: Number of modes kept in dimension i (mi ≤ di)
- **Typical**: mi << di for efficiency
- **Example**: If n_modes=(16,16) and input is (64,64), output is (B,C,16,16)

#### Weight Slicing: `_get_weight_indices()`
```
weight: (C_in, C_out, m1, m2, ..., mN)  # Dense
or factorized equivalent
```

### 5. Spectral Contraction
**Location**: `spectral_convolution.py:558-560`

```python
out_fft[slices_x] = self._contract(x[slices_x], weight, separable=self.separable)
```

#### Dense Contraction (default):
```
Input:  (B, C_in, m1, m2, ..., mN)
Weight: (C_in, C_out, m1, m2, ..., mN)
Output: (B, C_out, m1, m2, ..., mN)
```

#### Separable Contraction:
```
Input:  (B, C, m1, m2, ..., mN)
Weight: (C, m1, m2, ..., mN)
Output: (B, C, m1, m2, ..., mN)
```
- **Note**: Only works when C_in = C_out

### 6. Output Buffer Preparation
**Location**: `spectral_convolution.py:544-546`

```python
out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype)
```
```
out_fft: (B, C_out, fft_size)
```
- **Real data**: (B, C_out, d1, d2, ..., dN//2 + 1)
- **Complex data**: (B, C_out, d1, d2, ..., dN)

### 7. Inverse FFT Shift
**Location**: `spectral_convolution.py:574-575`

```python
out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
```
```
out_shifted: Same dimensions as out_fft
```

### 8. Inverse FFT
**Location**: `spectral_convolution.py:578-583`

#### For Real Data:
```python
x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
```
```
Output: (B, C_out, d1', d2', ..., dN')
```

#### For Complex Data:
```python
x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
```
```
Output: (B, C_out, d1', d2', ..., dN')
```

- **mode_sizes**: Original spatial dimensions or scaled if `resolution_scaling_factor`
- **Real output**: Even if FFT was complex, final result is real-valued

### 9. Bias Addition (optional)
**Location**: `spectral_convolution.py:587-588`

```python
if self.bias is not None:
    x = x + self.bias
```
```
Final Output: (B, C_out, d1', d2', ..., dN')
```

## Key Dimension Transformations Summary

| Step | Input Dimensions | Output Dimensions | Purpose |
|------|------------------|-------------------|---------|
| Input | `(B, C_in, d1, ..., dN)` | Same | Raw spatial data |
| FFT | `(B, C_in, d1, ..., dN)` | `(B, C_in, d1, ..., dN//2+1)` or `(B, C_in, d1, ..., dN)` | Transform to frequency domain |
| Mode Truncation | `(B, C_in, d1, ..., dN//2+1)` | `(B, C_in, m1, ..., mN)` | Keep only low frequencies |
| Contraction | `(B, C_in, m1, ..., mN)` | `(B, C_out, m1, ..., mN)` | Channel mixing in frequency |
| Inverse FFT | `(B, C_out, d1, ..., dN//2+1)` | `(B, C_out, d1', ..., dN')` | Back to spatial domain |

## Memory and Complexity Analysis

### Memory Requirements
- **Input spatial domain**: `O(B × C_in × Π di)`
- **FFT frequency domain**: `O(B × C_in × Π di)` (with complex overhead)
- **Truncated modes**: `O(B × C_in × Π mi)` where `mi << di`
- **Weights**: `O(C_in × C_out × Π mi)` or factorized equivalent

### Computational Complexity
- **FFT**: `O(B × C_in × Π di × log(Π di))`
- **Contraction**: `O(B × C_in × C_out × Π mi)`
- **Inverse FFT**: `O(B × C_out × Π di' × log(Π di'))`

### Key Efficiency Gains
1. **Mode truncation**: `mi << di` dramatically reduces contraction cost
2. **Global receptive field**: Single FFT captures all spatial relationships
3. **Linear scaling**: Unlike attention which is `O(N²)`, FNO is `O(N log N)`

## Resolution Scaling

### Forward Scaling (`transform` method)
**Location**: `spectral_convolution.py:397-416`

```python
return torch.nn.functional.interpolate(x, size=out_shape, mode='bilinear')
```
- **Input**: `(B, C, d1, ..., dN)`
- **Output**: `(B, C, round(d1×s1), ..., round(dN×sN))`

### Output Scaling (main forward)
**Location**: `spectral_convolution.py:563-572`

```python
mode_sizes = [round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)]
```
- **Enables multi-resolution FNO**: Different resolutions per layer
- **Memory adaptive**: Higher resolution only where needed

## Example: 2D FNO with 64×64 Input

```
Input:           (B, 32, 64, 64)
FFT:             (B, 32, 64, 33)  # Real FFT truncates last dim
Mode truncation: (B, 32, 16, 16)  # n_modes=(16,16)
Contraction:     (B, 64, 16, 16)  # 32→64 channels
Inverse FFT:     (B, 64, 64, 64)  # Back to original resolution
```

## Parameter Impact on Dimensions

| Parameter | Effect on Dimensions | Typical Values |
|-----------|----------------------|----------------|
| `n_modes` | Controls `mi` in frequency domain | (8,8) to (64,64) |
| `resolution_scaling_factor` | Controls output spatial size `di'` | 0.5, 1.0, 2.0 |
| `complex_data` | Determines FFT redundancy handling | False (most common) |
| `separable` | Weight tensor shape | True: (C, m1, ..., mN)<br>False: (C_in, C_out, m1, ..., mN) |
| `factorization` | Parameter storage efficiency | Dense, Tucker, CP, TT |

This dimension analysis shows how FNO achieves global operations with local complexity through the FFT-based approach.