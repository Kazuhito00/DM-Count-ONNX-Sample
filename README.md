# DM-Count-ONNX-Sample
[DM-Count](https://github.com/cvlab-stonybrook/DM-Count)のONNX変換/推論のサンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>
変換自体を試したい方はColaboratoryなどで[DM_Count_Convert2ONNX.ipynb](DM_Count_Convert2ONNX.ipynb)を使用ください。<br>

https://user-images.githubusercontent.com/37477845/142979000-b040ccd1-f5df-4e21-8920-0d1501b6007d.mp4

# Requirement
* OpenCV 4.5.3.56 or later
* onnxruntime-gpu 1.9.0 or later <br>※onnxruntimeでも動作しますが、推論時間がかかるのでGPUをお勧めします

# Demo
デモの実行方法は以下です。
```bash
python sample_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/DM-Count_QNRF_640_360.onnx
* --input_size<br>
モデルの入力サイズ<br>
デフォルト：640,360

# Reference
* [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MoveNet-Python-Example is under [MITLicense](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[ロンドン市内 雑踏](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002050318_00000)を使用しています。
