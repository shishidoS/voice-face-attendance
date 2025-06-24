# 音声認識・顔認識・自動データ送信システム

このプロジェクトは、音声認識で「出勤」や「退勤」といったワードを検出し、顔認識で該当者を特定し、その情報をサーバーに自動送信するシステムです。Raspberry PiなどのLinux環境で動作します。

---

## 主な機能

- **リアルタイム音声認識**  
  マイクからの音声をリアルタイムでテキスト化し、特定ワード（例：「出勤」「退勤」）を検出します。

- **顔認識**  
  USBカメラやRaspberry Piカメラで撮影した画像から、既知の顔データと照合し該当者を特定します。

- **自動データ送信**  
  認識された名前とステータス（出勤/退勤）をサーバーにPOSTリクエストで送信します。

---

## 必要環境・ライブラリ

- Python 3.7以上
- 必要なPythonライブラリ:
    - `face_recognition`
    - `speechrecognition`
    - `sounddevice`
    - `requests`
    - `opencv-python`
    - `numpy`
    - `pyaudio`（環境によっては必要）

インストール例:
```bash
pip install face_recognition speechrecognition sounddevice requests opencv-python numpy pyaudio

セットアップ手順
リポジトリをクローン

git clone https://github.com/your-username/your-repository.git
cd your-repository

既知の顔データを準備
known_facesディレクトリを作成し、以下のような構造で顔画像を保存してください。
known_faces/
├── 山田太郎/
│   ├── image1.jpg
│   ├── image2.jpg
├── 鈴木花子/
    ├── image1.jpg
    ├── image2.jpg

    「サーバーURLの設定」
main.py内のPOST_URLを適切なサーバーのURLに変更してください。

「使い方」
システムの起動
python main.py

「動作の流れ」
マイクから音声をリアルタイムで認識します。
「出勤」または「退勤」というワードが検出されると、カメラで写真を撮影します。
撮影した写真を既知の顔データと照合し、該当者を特定します。
認識された名前とステータス（出勤/退勤）をサーバーに送信します。
終了方法

実行中にCtrl+Cでシステムを終了できます。


実行時コマンドラインサンプル
$ python main.py
マイクからの音声を認識中...（Ctrl+Cで終了）
認識結果: 出勤
音声認識で「出勤」が検出されました。顔認識を開始します...
写真を撮影します...
captured.jpg に保存されました。
山田太郎さん、おはよーござんす!('◇')ゞ
送信成功: {"status": "success", "message": "データが正常に送信されました。"}

注意事項
カメラとマイクが正しく接続されていることを確認してください。
顔データはknown_facesディレクトリに正しい形式で登録してください。
サーバーが正しく動作していることを確認してください。
音声認識は環境ノイズの影響を受けるため、静かな環境でご利用ください。

ディレクトリ構成例
/project-root
├── main.py
├── known_faces/
│   ├── 山田太郎/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── 鈴木花子/
│       ├── image1.jpg
│       ├── image2.jpg
├── requirements.txt
└── README.md