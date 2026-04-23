# Setup environment variables for nnU-Net v2
$Env:nnUNet_raw = "f:\Workspace\med-img-seg\nnUNet_data\nnUNet_raw"
$Env:nnUNet_preprocessed = "f:\Workspace\med-img-seg\nnUNet_data\nnUNet_preprocessed"
$Env:nnUNet_results = "f:\Workspace\med-img-seg\nnUNet_data\nnUNet_results"

Write-Host "nnU-Net environment variables set:"
Write-Host "nnUNet_raw: $Env:nnUNet_raw"
Write-Host "nnUNet_preprocessed: $Env:nnUNet_preprocessed"
Write-Host "nnUNet_results: $Env:nnUNet_results"
