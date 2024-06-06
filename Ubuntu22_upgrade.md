# 22.04 LTSにアップグレード

Ubuntu 20.04から22.04 LTSにアップグレードの手順
## 前提条件
1. データのバックアップ
2. 安定したインターネット接続

## 1: データのバックアップ　(省略)

ユーザー設定データのみバックアップするため、ホームディレクトリのバックアップを行います。

```bash
rsync -av --progress /home/ai-user1 /path/to/backup/location --exclude '*.cache'
```

## 2: システムのアップグレード準備

システムパッケージを最新に更新します。

```bash
sudo apt update
sudo apt upgrade 
#sudo apt dist-upgrade -y
sudo apt autoremove
```

## 3: Ubuntu 20.04から22.04 LTSへのアップグレード

#### 1. アップグレードのチェック

以下のコマンドでアップグレードが可能かどうかを確認します。

```bash
sudo do-release-upgrade -c
```

アップグレードが可能であれば、次のコマンドで実際にアップグレードを開始します。

#### 2. アップグレードの実施

```bash
sudo do-release-upgrade
```

アップグレードの指示に従って進行します。(設定の内容更新に関してはdefault(N)にしてください)
アップグレードが完了すると、システムが再起動します。

## アップグレード後の設置プログラム
Ubuntu 22.04へのアップグレード後、CUDA 11.7に対応するPyTorch、TensorFlow、およびstable-baselines3[extra]をインストールする手順は以下の通りです。関連ライブラリのバージョンに注意して進めます。

### 前提条件
1. Ubuntu 22.04がインストールされていること
2. CUDA 11.7がインストールされていること

### 確認事項
Ubuntu 22.04へアップグレードした後、CUDA 11.7がすでに`/usr/local/cuda`に設置されている場合、再インストールする必要はありません。ただし、いくつかの点を確認しておくことが重要です。

1. **CUDAパスの確認**:
   CUDAが正しくインストールされ、環境変数が設定されているか確認します。

   ```bash
   echo $PATH | grep /usr/local/cuda
   echo $LD_LIBRARY_PATH | grep /usr/local/cuda
   ```

2. **CUDAの動作確認**:
   CUDAが正しく機能しているか確認します。

   ```bash
   nvcc --version
   nvidia-smi
   ```
アップグレードにより環境変数がリセットされる可能性があるため、必要に応じて再設定します。

#### 環境変数の設定

以下のコマンドを実行して、環境変数を設定します。

```bash
echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### CUDAのサンプルコンパイル確認

CUDAサンプルをコンパイルして、正しくインストールされていることを確認します。

```bash
cd /usr/local/cuda-11.7/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

このコマンドでCUDAデバイス情報が正しく表示されれば、CUDAは正しくインストールされています。

### cuDNNとTensorRT
**cuDNNとTensorRTの対応バージョン**
CUDA 11.7に対応するcuDNNおよびTensorRTのバージョンとインストール手順を以下に示します。
- **cuDNN**: 8.5.0, 8.9.7
- **TensorRT**: 8.4.1, 8.6.1, 9.3
- Compute Capability
    - Geforce RTX 2080	7.5
    - GeForce RTX 3060	8.6
    - GeForce RTX 4060	8.9

#### cuDNN のインストール

ダウンロードしたアーカイブを展開し、適切なディレクトリにコピーします。

#NVIDIAの公式サイトからcuDNN 8.5.0をダウンロードします（NVIDIAアカウントが必要です）。

```bash
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/cudnn-linux-x64-v8.5.0.96.tgz

# 8.5.0の場合
tar -xzvf cudnn-linux-x64-v8.5.0.96.tgz
sudo cp -P cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/

# 8.9.7の場合
tar -xf cudnn-linux-x86_64-8.9.7.1_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.7.1_cuda11-archive/include/* /usr/local/cuda/include
sudo cp cudnn-linux-x86_64-8.9.7.1_cuda11-archive/lib/* /usr/local/cuda/lib64

sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### TensorRTインストール

#### 1. TensorRT ダウンロード

### TensorRT 9.3.0 (RTX4060 CUDA 11.8)
    - cuDNN 8.9.7
    - TensorFlow 2.12.0
    - PyTorch >= 2.0 
    - ONNX 1.14.1
    - CUDA 11.8 /11.7 update 1
    
    ```
    wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/9.3.0/local_repos/nv-tensorrt-repo-ubuntu2204-cuda11.7-trt9.3.0_1-1_amd64.deb
    sudo dpkg -i nv-tensorrt-repo-ubuntu2204-cuda11.7-trt9.3.0_1-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-repo-cuda11.7-trt9.3.0/7fa2af80.pub
    ```

####  TensorRT 8.6.1のインストール
   - NVIDIA cuDNN 8.9.0 
   - PyTorch 1.13.1
   - cuda 11.7 update 1

    ```
    wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-repo-ubuntu2204-cuda11.7-trt8.6.1_1-1_amd64.deb
    sudo dpkg -i nv-tensorrt-repo-ubuntu2204-cuda11.7-trt8.6.1_1-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-repo-cuda11.7-trt8.6.1/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y tensorrt
    sudo apt-get install -y python3-libnvinfer-dev
    ```

####  TensorRT 8.4.1のインストール
- cuDNN 8.4.1
- TensorFlow 1.15.5
- PyTorch 1.9.0
- CUDA 11.7

ダウンロードしたDEBパッケージをインストールします。
NVIDIAの公式サイトからTensorRT 8.4.1をダウンロードします（NVIDIAアカウントが必要です）。

```bash
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1.5/local_repos/nv-tensorrt-repo-ubuntu2204-cuda11.7-trt8.4.1.5-ea-20220622_1-1_amd64.deb
```

```bash
sudo dpkg -i nv-tensorrt-repo-ubuntu2204-cuda11.7-trt8.4.1.5-ea-20220622_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-cuda11.7-trt8.4.1.5-ea-20220622/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y tensorrt
sudo apt-get install -y python3-libnvinfer-dev
```

### 確認

インストールが正しく行われたか確認します。

#### 1. cuDNNのバージョン確認

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

#### 2. TensorRTのバージョン確認

```bash
dpkg -l | grep nvinfer
```

### 4: PyTorchのインストール

CUDA 11.7に対応するPyTorchとtorchvisionをインストールします。

```bash
#pip install torch==1.12.1+cu117 torchvision==0.13.1+cu117 torchaudio==0.12.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 11.7
#pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

### 5: TensorFlowのインストール

CUDA 11.7に対応するTensorFlowをインストールします。

```bash
pip install tensorflow==2.12.0
#TensorFlow 2.10.0以降（例：2.10.0、2.11.0、2.12.0など）
```

### 6: stable-baselines3[extra]のインストール

stable-baselines3と追加の依存関係をインストールします。

```bash
pip install stable-baselines3[extra]
```

### 7: 依存関係の確認

インストールされたライブラリとそのバージョンを確認します。

```bash
pip list | grep -E 'torch|tensorflow|stable-baselines3'
```

## NVIDIA CUDA 注意点
#### CUDA 11.7の再インストール手順
一般的な設置方法だが、nvidia driver versionに異なる場合がある
1. **既存のCUDAのアンインストール**:

   ```bash
   sudo /usr/local/cuda-11.7/bin/cuda-uninstaller
   sudo apt-get --purge remove '*cublas*' 'cuda*' 'nsight*'
   sudo apt-get autoremove
   sudo rm -rf /usr/local/cuda*
   ```

2. **CUDA 11.7のインストール**:

   ```bash
   sudo apt update
   sudo apt install -y build-essential dkms
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```
**CUDA 11.7のローカルインストーラーを使用して、NVIDIAドライバーをインストールせずにCUDAツールキットのみをインストールする**

### 1: NVIDIAドライバーのインストール

Ubuntu 22.04に対応するNVIDIAドライバーをインストールします。

```bash
sudo apt update
sudo apt install -y nvidia-driver-515
```

ドライバーのインストールが完了したら、システムを再起動します。

```bash
sudo reboot
```

再起動後、ドライバーが正しくインストールされているか確認します。

```bash
nvidia-smi
```

### 2: CUDA 11.7のローカルインストーラーを使用したインストール

#### 1. CUDA 11.7のローカルインストーラーをダウンロード

CUDA 11.7のローカルインストーラーをNVIDIAの公式サイトからダウンロードします。

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
```

#### 2. インストーラーを実行

インストーラーを実行しますが、NVIDIAドライバーのインストールをスキップします。

```bash
sudo sh cuda_11.7.1_515.65.01_linux.run
```

インストールプロセス中に、以下のオプションを選択します：

- `Driver`のインストールはスキップします（既にインストール済み）。
- `CUDA Toolkit`のみをインストールします。


## ROS2 Humbleのインストール

## ROS2 Foxy関連の設定を削除

`.bashrc`ファイルからROS2 Foxy関連の設定を削除します。

```bash
nano ~/.bashrc
```

以下のようなFoxy関連の行を見つけて削除します。

```bash
#source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=1
```

変更を保存して`.bashrc`を閉じます。

### 1. ROS2 Humbleのインストール準備

ROS2 Humbleのパッケージを取得するためのリポジトリを追加します。

```bash
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
```

### 2. ROS2 Humbleのインストール

ROS2 Humbleのデスクトップバージョンをインストールします。

```bash
sudo apt update
sudo apt install -y ros-humble-desktop
```

### 3. ROS2 Humbleの設定

`.bashrc`ファイルにROS2 Humbleの設定を追加します。

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 4. 必要なROS2ツールのインストール

ビルドツールやその他の依存関係をインストールします。

```bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep
```













