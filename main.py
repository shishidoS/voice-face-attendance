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
import pickle
from time import sleep

# 設定
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 3  # 秒
OUTPUT_TXT_FILE = "./" + datetime.now().strftime('%Y%m%d_%H_%M') + ".txt"
POST_URL = "https://96c6-125-103-211-238.ngrok-free.app/receive"  # サーバーのURL（適宜変更）
FACE_ENCODINGS_FILE = 'face_encodings.pkl'  # pickle保存ファイル

# デバッグモード
DEBUG = True

def debug_print(message):
    if DEBUG:
        print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} - {message}")

# 音声データを格納するキュー
q = queue.Queue()

# 音声認識器
recognizer = sr.Recognizer()

def save_face_encodings(known_folder, save_file=FACE_ENCODINGS_FILE):
    """
    既知の顔画像から特徴量を抽出してpickleファイルに保存する
    """
    data = {}
    if not os.path.exists(known_folder):
        print(f"{known_folder} フォルダが存在しません。")
        return False
    
    for person_name in os.listdir(known_folder):
        person_dir = os.path.join(known_folder, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        encodings = []
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encs = face_recognition.face_encodings(image)
                    encodings.extend(face_encs)
                except Exception as e:
                    print(f"画像処理エラー {image_path}: {e}")
                    continue
        
        if encodings:
            data[person_name] = encodings
            print(f"{person_name} の顔を学習しました（{len(encodings)} 件）。")
    
    if data:
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"人物の顔特徴量を {save_file} に保存しました。")
        return True
    else:
        print("学習可能な顔画像が見つかりませんでした。")
        return False

def load_face_encodings(save_file=FACE_ENCODINGS_FILE):
    """
    pickleファイルから顔特徴量を読み込む
    """
    try:
        with open(save_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"{save_file} が見つかりません。顔特徴量を学習します...")
        if save_face_encodings('known_faces'):
            with open(save_file, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        print(f"特徴量ファイル読み込みエラー: {e}")
        return {}

def take_photo_from_usb_camera(filename='captured.jpg'):
    """
    USBカメラより写真を取得する関数
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("カメラが開けません。USBカメラが正しく接続されているか確認してください。")
        return None
    debug_print("カメラ初期化中…3秒待機します。")
    sleep(3)
    debug_print("写真を撮影します...")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        debug_print(f"{filename} に保存されました。")
        cap.release()
        return filename
    else:
        print("画像の取得に失敗しました。")
        cap.release()
        return None

def recognize_person_from_encodings(unknown_image_path, tolerance=0.45):
    """
    pickleファイルから読み込んだ特徴量を使用して顔認識処理を実行する
    """
    debug_print("顔認識処理開始")
    
    # 保存された特徴量を読み込み
    known_encodings_data = load_face_encodings()
    if not known_encodings_data:
        print("学習データが見つかりません。")
        return None
    
    debug_print(f"学習データ読み込み完了: {len(known_encodings_data)} 人")
    
    # 未知の画像を読み込み
    try:
        unknown_image = face_recognition.load_image_file(unknown_image_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        debug_print(f"撮影画像から {len(unknown_encodings)} 個の顔を検出")
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return None
    
    if not unknown_encodings:
        print("顔、出てこんだったに。")
        return None

    unknown_encoding = unknown_encodings[0]

    # 各人物の特徴量と比較
    for person_name, encodings in known_encodings_data.items():
        try:
            debug_print(f"{person_name} と比較中...")
            results = face_recognition.compare_faces(encodings, unknown_encoding, tolerance=tolerance)
            if any(results):
                print(f"{person_name}さん、おはよーござんす!('◇')ゞ")
                return person_name
        except Exception as e:
            print(f"顔認識比較エラー {person_name}: {e}")
            continue

    print("おめ、だれけ〜？")
    return None

def send_message(person_name, status):
    """
    サーバーに名前とステータスをPOST送信する
    """
    day = datetime.now().strftime('%Y-%m-%d')
    times = datetime.now().strftime('%H:%M')
    try:
        debug_print(f"サーバー送信: {person_name} - {status}")
        res = requests.post(
            POST_URL,
            json={
                "日": day,
                "時間": times,
                "入力テキスト": person_name,
                "status": status}
        )
        print("送信ステータス:", res.text)
    except Exception as e:
        print("送信失敗:", e)

def detect_keyword_and_status(text):
    """
    テキストからキーワードを検出してステータスを返す
    """
    # キーワード辞書（類似表現も含む）
    keyword_patterns = {
        "出勤": ["出勤", "しゅっきん", "出社", "しゅっしゃ", "仕事", "会社", "おはよう"],
        "退勤": ["退勤", "たいきん", "帰る", "かえる", "退社", "たいしゃ", "お疲れ", "さようなら", "最近", "ダイキン"],
        "休憩開始": ["休憩", "きゅうけい", "休む", "やすむ", "ちょっと"],
        "休憩終了": ["戻る", "もどる", "復帰", "ふっき", "再開", "さいかい"]
    }
    
    text_lower = text.lower()
    debug_print(f"キーワード検索対象: '{text}'")
    
    for status, keywords in keyword_patterns.items():
        for keyword in keywords:
            if keyword in text or keyword in text_lower:
                debug_print(f"キーワード検出: '{keyword}' → ステータス: {status}")
                return status
    
    debug_print("該当キーワードなし")
    return None

def recognize_from_queue():
    """
    音声認識をキューから処理する
    """
    debug_print("音声認識スレッド開始")
    
    while True:
        audio_data = q.get()
        if audio_data is None:
            debug_print("音声認識スレッド終了")
            break
            
        try:
            debug_print("音声認識処理開始")
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            audio = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
            text = recognizer.recognize_google(audio, language="ja-JP")
            
            print(f"🎤 認識結果: {text}")
            
            with open(OUTPUT_TXT_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n{datetime.now().strftime('%H:%M:%S')} - {text}")

            # キーワード検出
            status = detect_keyword_and_status(text)
            
            if status:
                print(f"✅ 音声認識で「{status}」が検出されました。顔認識を開始します...")
                
                # 写真撮影
                filename = take_photo_from_usb_camera()
                if filename:
                    # 顔認識
                    person_name = recognize_person_from_encodings(filename)
                    if person_name:
                        # サーバー送信
                        send_message(person_name, status)
                        print(f"✅ {person_name}さんの{status}を記録しました！")
                    else:
                        print("❌ 顔認識に失敗しました。")
                else:
                    print("❌ 写真撮影に失敗しました。")
            else:
                debug_print("対象キーワードが含まれていないため、処理をスキップ")
                
        except sr.UnknownValueError:
            debug_print("音声を認識できませんでした")
        except sr.RequestError as e:
            print(f"❌ Google APIエラー: {e}")
        except Exception as e:
            print(f"❌ 音声認識エラー: {e}")

def realtime_textise():
    """
    音声認識をリアルタイムで処理する
    """
    debug_print("リアルタイム音声認識開始")
    
    with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
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
            print("🎙️  システム起動完了！")
            print("📢 以下のようにお話しください:")
            print("   - 出勤します")
            print("   - 退勤します") 
            print("   - 休憩します")
            print("   - 戻ります")
            print("💡 Ctrl+Cで終了")
            
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 終了します...")
    finally:
        q.put(None)
        thread.join()

def audio_callback(indata, frames, time_info, status):
    """
    音声データをキューに格納する
    """
    if status:
        debug_print(f"録音エラー: {status}")
    q.put(indata.copy())

def initialize_face_recognition():
    """
    顔認識システムの初期化
    """
    debug_print("顔認識システム初期化開始")
    
    # known_facesフォルダの確認
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
        print("📁 known_facesフォルダを作成しました。")
        print("👤 このフォルダ内に人物名のサブフォルダを作成し、その中に顔写真を配置してください。")
        print("📝 例: known_faces/田中/photo1.jpg")
        return False
    
    # 特徴量ファイルの確認と生成
    if not os.path.exists(FACE_ENCODINGS_FILE):
        print("🧠 顔特徴量ファイルが見つかりません。学習を開始します...")
        return save_face_encodings('known_faces')
    
    print("✅ 顔特徴量ファイルが見つかりました。")
    return True

def main():
    print("=" * 50)
    print("🤖 出退勤管理システム v2.0")
    print("=" * 50)
    
    # 顔認識システムの初期化
    if initialize_face_recognition():
        print("✅ 顔認識システムが正常に初期化されました。")
    else:
        print("⚠️  顔認識システムの初期化に失敗しました。")
        print("📁 known_facesフォルダに画像を配置してから再実行してください。")
    
    realtime_textise()

if __name__ == '__main__':
    main()