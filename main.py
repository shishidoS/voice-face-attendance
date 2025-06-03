import sounddevice as sd
import numpy as np
import speech_recognition as sr
import time
from datetime import datetime
import queue
import threading
import requests
import os
import face_recognition
import cv2
from time import sleep

# 設定
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 3  # 秒
OUTPUT_TXT_FILE = "./" + datetime.now().strftime('%Y%m%d_%H_%M') + ".txt"
POST_URL = "https://Flask動作のURLです/receive"  # サーバーのURL（変更）

# 音声データを格納するキュー
q = queue.Queue()

# 音声認識器
recognizer = sr.Recognizer()

def take_photo_from_usb_camera(filename='captured.jpg'):
    """
    USBカメラで写真を撮影する
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("カメラが開けません。USBカメラが正しく接続されているか確認してください。")
        return None
    print("カメラ初期化中…3秒待機します。")
    sleep(3)
    print("写真を撮影します...")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"{filename} に保存されました。")
        cap.release()
        return filename
    else:
        print("画像の取得に失敗しました。")
        cap.release()
        return None

def recognize_person_from_folder(known_folder, unknown_image_path, tolerance=0.45):
    """
    顔認識処理
    """
    unknown_image = face_recognition.load_image_file(unknown_image_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if not unknown_encodings:
        print("顔、出てこんだったに。")
        return None

    unknown_encoding = unknown_encodings[0]

    for person_name in os.listdir(known_folder):
        person_dir = os.path.join(known_folder, person_name)
        if not os.path.isdir(person_dir):
            continue

        encodings = []
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_dir, filename)
                image = face_recognition.load_image_file(image_path)
                face_encs = face_recognition.face_encodings(image)
                encodings.extend(face_encs)

        if not encodings:
            continue

        results = face_recognition.compare_faces(encodings, unknown_encoding, tolerance=tolerance)
        if any(results):
            print(f"{person_name}さん、おはよーござんす!('◇')ゞ")
            return person_name

    print("おめ、だれけ〜？")
    return None

def send_message(person_name, status):
    """
    サーバーに名前とステータスをPOST送信する
    """
    day = datetime.now().strftime('%Y-%m-%d')
    times = datetime.now().strftime('%H:%M')
    try:
        res = requests.post(
            POST_URL,
            json={"送信データ": f"日＝{day},時間＝{times}, 入力テキスト＝{person_name}, status={status}"}
        )
        print("送信成功:", res.text)
    except Exception as e:
        print("送信失敗:", e)

def recognize_from_queue():
    """
    音声認識をキューから処理する
    """
    while True:
        audio_data = q.get()
        if audio_data is None:
            break
        try:
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            audio = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
            text = recognizer.recognize_google(audio, language="ja-JP")
            print("認識結果:", text)
            with open(OUTPUT_TXT_FILE, 'a') as f:
                f.write("\n" + text)

            # 「出勤」または「退勤」が含まれている場合に顔認識を開始
            if "出勤" in text or "退勤" in text:
                status = "出勤" if "出勤" in text else "退勤"
                print(f"音声認識で「{status}」が検出されました。顔認識を開始します...")
                filename = take_photo_from_usb_camera()  # 写真を撮影
                if filename:
                    person_name = recognize_person_from_folder("known_faces", filename)
                    if person_name:
                        send_message(person_name, status)
                    else:
                        print("顔認識に失敗しました。")
        except sr.UnknownValueError:
            print("音声を認識できませんでした")
        except sr.RequestError as e:
            print(f"Google APIエラー: {e}")
        except Exception as e:
            print(f"その他のエラー: {e}")

def realtime_textise():
    """
    音声認識をリアルタイムで処理する
    """
    with open(OUTPUT_TXT_FILE, 'w') as f:
        DATE = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        f.write("日時 : " + DATE + "\n")

    # 音声認識スレッドを開始
    thread = threading.Thread(target=recognize_from_queue, daemon=True)
    thread.start()

    # 録音開始
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            callback=audio_callback,
                            blocksize=int(SAMPLE_RATE * BLOCK_DURATION)):
            print("マイクからの音声を認識中...（Ctrl+Cで終了）")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("終了します...")
    finally:
        q.put(None)
        thread.join()

def audio_callback(indata, frames, time_info, status):
    """
    音声データをキューに格納する
    """
    if status:
        print("録音エラー:", status)
    q.put(indata.copy())

def main():
    realtime_textise()

if __name__ == '__main__':
    main()