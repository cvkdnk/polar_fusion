sudo apt-get install libsparsehash-dev
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install opencv-python pytorch-lightning spconv-cu117
pip3 install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
pip3 install open3d numba wandb pyyaml torch_scatter