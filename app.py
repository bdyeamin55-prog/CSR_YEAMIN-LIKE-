import os
import sys
import json
import asyncio
import threading
import random
import binascii
import traceback
import time
from flask import Flask, request, jsonify, send_file
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import aiohttp
import requests
import urllib3
from datetime import datetime
from functools import lru_cache

# Import protobuf modules
try:
    import like_pb2
    import like_count_pb2
    import uid_generator_pb2
except ImportError as e:
    print(f"Error importing protobuf modules: {e}")
    # Placeholder classes
    class like_pb2:
        class like:
            def __init__(self):
                self.uid = 0
                self.region = ""
            def SerializeToString(self):
                return b""
    
    class like_count_pb2:
        class Info:
            def __init__(self):
                self.AccountInfo = type('AccountInfo', (), {'Likes': 0, 'UID': 0, 'PlayerNickname': ''})
            def ParseFromString(self, data):
                pass
    
    class uid_generator_pb2:
        class uid_generator:
            def __init__(self):
                self.krishna_ = 0
                self.teamXdarks = 1
            def SerializeToString(self):
                return b""

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
ACCOUNT_BATCH_SIZE = 50  # একসাথে কতগুলো অ্যাকাউন্ট ব্যবহার করবে
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_CACHE = {}  # টোকেন ক্যাশ করার জন্য
TOKEN_CACHE_LOCK = threading.Lock()

# Logger Class
class Logger:
    @staticmethod
    def log(message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}", file=sys.stderr)
        sys.stderr.flush()

    @staticmethod
    def error(message):
        Logger.log(message, "ERROR")
    
    @staticmethod
    def info(message):
        Logger.log(message, "INFO")
    
    @staticmethod
    def debug(message):
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            Logger.log(message, "DEBUG")

# ============================================
# UID + PASSWORD SYSTEM - Main Functions
# ============================================

def get_account_file_path(server_name, for_visit=False):
    """অ্যাকাউন্ট ফাইলের পাথ রিটার্ন করে"""
    accounts_dir = os.path.join(BASE_DIR, 'accounts')
    
    # accounts ফোল্ডার না থাকলে তৈরি করো
    if not os.path.exists(accounts_dir):
        os.makedirs(accounts_dir)
    
    if for_visit:
        filename = f"account_{server_name.lower()}_visit.json"
    else:
        filename = f"account_{server_name.lower()}.json"
    
    return os.path.join(accounts_dir, filename)

def load_accounts(server_name, for_visit=False):
    """
    UID এবং পাসওয়ার্ড লোড করার ফাংশন
    ফাইল ফরম্যাট: accounts/account_server.json
    যেমন: accounts/account_bd.json, accounts/account_ind.json etc.
    """
    
    filepath = get_account_file_path(server_name, for_visit)
    
    try:
        if not os.path.exists(filepath):
            Logger.error(f"Account file not found: {filepath}")
            return []
            
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                Logger.error(f"Account file is empty: {filepath}")
                return []
                
            accounts = json.loads(content)
            
            if isinstance(accounts, list):
                # Validate account format (uid and password required)
                valid_accounts = []
                for acc in accounts:
                    if isinstance(acc, dict) and "uid" in acc and "password" in acc:
                        if acc["uid"] and acc["password"]:
                            valid_accounts.append(acc)
                        else:
                            Logger.warning(f"Invalid account data: {acc}")
                    else:
                        Logger.warning(f"Invalid account format: {acc}")
                
                Logger.info(f"Loaded {len(valid_accounts)} valid accounts from {os.path.basename(filepath)}")
                return valid_accounts
            else:
                Logger.error(f"Invalid account format in {filepath}")
                return []
                
    except FileNotFoundError:
        Logger.error(f"Account file not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        Logger.error(f"Invalid JSON in {filepath}: {e}")
        return []
    except Exception as e:
        Logger.error(f"Unexpected error loading accounts: {e}")
        return []

def login_and_get_token(uid, password, server_name):
    """
    UID এবং পাসওয়ার্ড ব্যবহার করে লগইন করে টোকেন রিটার্ন করে
    টোকেন ক্যাশ করে রাখে ১ ঘন্টার জন্য
    """
    
    # ক্যাশে টোকেন আছে কিনা চেক করুন
    cache_key = f"{uid}_{server_name}"
    with TOKEN_CACHE_LOCK:
        if cache_key in TOKEN_CACHE:
            token_data = TOKEN_CACHE[cache_key]
            if time.time() - token_data['timestamp'] < 3600:  # 1 hour
                Logger.debug(f"Using cached token for UID: {uid}")
                return token_data['token']
    
    try:
        # লগইন API URL
        if server_name == "IND":
            login_url = "https://client.ind.freefiremobile.com/Login"
        elif server_name in {"BR", "US", "SAC", "NA"}:
            login_url = "https://client.us.freefiremobile.com/Login"
        else:
            login_url = "https://clientbp.ggblueshark.com/Login"
        
        # Free Fire লগইন পেলোড (প্রোটোবাফ ফরম্যাট)
        # এটা ডেমো - আসল ফরম্যাট জানা প্রয়োজন
        login_payload = {
            "uid": uid,
            "password": password,
            "region": server_name,
            "version": "OB52"
        }
        
        headers = {
            'User-Agent': "Dalvik/2.1.0 (Linux; U; Android 9; ASUS_Z01QD Build/PI)",
            'Content-Type': "application/json",
            'Accept-Encoding': "gzip",
            'X-Unity-Version': "2018.4.11f1",
            'ReleaseVersion': "OB52"
        }
        
        Logger.info(f"Attempting login for UID: {uid}")
        response = requests.post(login_url, json=login_payload, headers=headers, verify=False, timeout=15)
        
        if response.status_code == 200:
            # TODO: রেসপন্স থেকে টোকেন এক্সট্রাক্ট করুন (আসল ফরম্যাট অনুযায়ী)
            response_data = response.json()
            token = response_data.get("token") or response_data.get("access_token")
            
            if token:
                # টোকেন ক্যাশে সংরক্ষণ
                with TOKEN_CACHE_LOCK:
                    TOKEN_CACHE[cache_key] = {
                        'token': token,
                        'timestamp': time.time()
                    }
                
                Logger.info(f"Login successful for UID: {uid}")
                return token
            else:
                Logger.error(f"No token in response for UID: {uid}")
                return None
        else:
            Logger.error(f"Login failed for UID {uid}: HTTP {response.status_code}")
            return None
            
    except requests.Timeout:
        Logger.error(f"Login timeout for UID: {uid}")
        return None
    except requests.RequestException as e:
        Logger.error(f"Login request error for UID {uid}: {e}")
        return None
    except Exception as e:
        Logger.error(f"Unexpected login error for UID {uid}: {e}")
        return None

def encrypt_message(plaintext):
    """AES এনক্রিপশন"""
    try:
        key = b'Yg&tc%DEuh6%Zc^8'
        iv = b'6oyZDr22E3ychjM%'
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_message = pad(plaintext, AES.block_size)
        encrypted_message = cipher.encrypt(padded_message)
        return binascii.hexlify(encrypted_message).decode('utf-8')
    except Exception as e:
        Logger.error(f"Encryption error: {e}")
        raise

def create_like_protobuf(user_id, region):
    """লাইক প্রোটোবাফ মেসেজ তৈরি"""
    try:
        message = like_pb2.like()
        message.uid = int(user_id)
        message.region = region
        return message.SerializeToString()
    except Exception as e:
        Logger.error(f"Like protobuf creation error: {e}")
        raise

def create_profile_protobuf(uid):
    """প্রোফাইল চেকের জন্য প্রোটোবাফ তৈরি"""
    try:
        message = uid_generator_pb2.uid_generator()
        message.krishna_ = int(uid)
        message.teamXdarks = 1
        return message.SerializeToString()
    except Exception as e:
        Logger.error(f"Profile protobuf error: {e}")
        raise

def enc_profile_check_payload(uid):
    """প্রোফাইল চেকের পেলোড এনক্রিপ্ট"""
    protobuf_data = create_profile_protobuf(uid)
    encrypted_uid = encrypt_message(protobuf_data)
    return encrypted_uid

async def send_single_like(encrypted_payload, token, url):
    """একটি টোকেন দিয়ে লাইক রিকোয়েস্ট পাঠান"""
    try:
        edata = bytes.fromhex(encrypted_payload)
        
        headers = {
            'User-Agent': "Dalvik/2.1.0 (Linux; U; Android 9; ASUS_Z01QD Build/PI)",
            'Connection': "Keep-Alive",
            'Accept-Encoding': "gzip",
            'Authorization': f"Bearer {token}",
            'Content-Type': "application/x-www-form-urlencoded",
            'X-Unity-Version': "2018.4.11f1",
            'ReleaseVersion': "OB52"
        }
        
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=edata, headers=headers, ssl=False) as response:
                status = response.status
                if status != 200:
                    Logger.error(f"Like failed with status: {status}")
                return status
                
    except asyncio.TimeoutError:
        Logger.error("Like request timeout")
        return 998
    except Exception as e:
        Logger.error(f"Like request exception: {e}")
        return 997

async def send_likes_with_accounts(target_uid, server_region, like_api_url, account_list):
    """একাধিক অ্যাকাউন্ট ব্যবহার করে লাইক পাঠায়"""
    if not account_list:
        Logger.error("Empty account list")
        return []
    
    try:
        like_protobuf = create_like_protobuf(target_uid, server_region)
        encrypted_payload = encrypt_message(like_protobuf)
        
        tasks = []
        valid_tokens = 0
        
        for account in account_list:
            # প্রতিটি অ্যাকাউন্টের জন্য লগইন করে টোকেন নাও
            token = login_and_get_token(account["uid"], account["password"], server_region)
            if token:
                tasks.append(send_single_like(encrypted_payload, token, like_api_url))
                valid_tokens += 1
            else:
                Logger.error(f"Failed to get token for account {account['uid']}")
        
        if tasks:
            Logger.info(f"Sending {len(tasks)} like requests from {valid_tokens} accounts")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = sum(1 for r in results if isinstance(r, int) and r == 200)
            Logger.info(f"Like results: {successful}/{len(tasks)} successful")
            
            return results
        else:
            Logger.error("No valid tokens obtained")
            return []
        
    except Exception as e:
        Logger.error(f"Batch send error: {e}")
        return []

def check_profile(encrypted_payload, server_name, account):
    """প্রোফাইল চেক করার ফাংশন (ভিজিট অ্যাকাউন্ট ব্যবহার করে)"""
    try:
        # ভিজিট অ্যাকাউন্ট দিয়ে লগইন
        token = login_and_get_token(account["uid"], account["password"], server_name)
        if not token:
            Logger.error("Failed to get token for profile check")
            return None
        
        if server_name == "IND":
            url = "https://client.ind.freefiremobile.com/GetPlayerPersonalShow"
        elif server_name in {"BR", "US", "SAC", "NA"}:
            url = "https://client.us.freefiremobile.com/GetPlayerPersonalShow"
        else:
            url = "https://clientbp.ggblueshark.com/GetPlayerPersonalShow"
        
        edata = bytes.fromhex(encrypted_payload)
        headers = {
            'User-Agent': "Dalvik/2.1.0 (Linux; U; Android 9; ASUS_Z01QD Build/PI)",
            'Authorization': f"Bearer {token}",
            'Content-Type': "application/x-www-form-urlencoded",
        }
        
        response = requests.post(url, data=edata, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        return decode_profile_info(response.content)
        
    except Exception as e:
        Logger.error(f"Profile check error: {e}")
        return None

def decode_profile_info(binary_data):
    """প্রোফাইল ডেটা ডিকোড করুন"""
    try:
        info = like_count_pb2.Info()
        info.ParseFromString(binary_data)
        return info
    except Exception as e:
        Logger.error(f"Protobuf decode error: {e}")
        return None

def get_next_batch_accounts(server_name, all_accounts):
    """রোটেটিং ব্যাচ সিলেকশন"""
    if not all_accounts:
        return []
    
    total = len(all_accounts)
    
    if total <= ACCOUNT_BATCH_SIZE:
        return all_accounts
    
    with batch_indices_lock:
        if server_name not in current_batch_indices:
            current_batch_indices[server_name] = 0
        
        current = current_batch_indices[server_name]
        start = current
        end = start + ACCOUNT_BATCH_SIZE
        
        if end > total:
            remaining = end - total
            batch = all_accounts[start:total] + all_accounts[0:remaining]
        else:
            batch = all_accounts[start:end]
        
        next_index = (current + ACCOUNT_BATCH_SIZE) % total
        current_batch_indices[server_name] = next_index
        
        return batch

# Flask অ্যাপ
app = Flask(__name__)

@app.route('/')
def home():
    """হোম পেজ"""
    try:
        html_path = os.path.join(BASE_DIR, 'index.html')
        if os.path.exists(html_path):
            return send_file(html_path)
        else:
            return jsonify({
                "status": "running",
                "message": "CSR YEAMIN LIKE - Free Fire Like Booster",
                "version": "2.0.0 (Password System)"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/like', methods=['GET'])
def handle_like_request():
    """লাইক রিকোয়েস্ট হ্যান্ডেল করুন"""
    try:
        target_uid = request.args.get("uid")
        server = request.args.get("server_name", "").upper()
        use_random = request.args.get("random", "false").lower() == "true"
        
        # ভ্যালিডেশন
        if not target_uid or not server:
            return jsonify({"error": "UID and server_name are required"}), 400
        
        if not target_uid.isdigit():
            return jsonify({"error": "UID must contain only digits"}), 400
        
        # ভিজিট অ্যাকাউন্ট লোড (প্রোফাইল চেকের জন্য)
        visit_accounts = load_accounts(server, for_visit=True)
        if not visit_accounts:
            return jsonify({"error": f"No visit accounts for server {server}"}), 500
        
        # রেগুলার অ্যাকাউন্ট লোড (লাইক পাঠানোর জন্য)
        regular_accounts = load_accounts(server, for_visit=False)
        if not regular_accounts:
            return jsonify({"error": f"No regular accounts for server {server}"}), 500
        
        Logger.info(f"Processing UID: {target_uid}, Server: {server}")
        Logger.info(f"Visit accounts: {len(visit_accounts)}, Regular accounts: {len(regular_accounts)}")
        
        # প্রোফাইল চেক পেলোড
        encrypted_profile = enc_profile_check_payload(target_uid)
        
        # Before likes
        before_info = check_profile(encrypted_profile, server, visit_accounts[0])
        before_likes = int(before_info.AccountInfo.Likes) if before_info and hasattr(before_info, 'AccountInfo') else 0
        
        # লাইক API URL
        if server == "IND":
            like_api_url = "https://client.ind.freefiremobile.com/LikeProfile"
        elif server in {"BR", "US", "SAC", "NA"}:
            like_api_url = "https://client.us.freefiremobile.com/LikeProfile"
        else:
            like_api_url = "https://clientbp.ggblueshark.com/LikeProfile"
        
        # অ্যাকাউন্ট ব্যাচ সিলেক্ট
        if use_random:
            account_batch = random.sample(regular_accounts, min(ACCOUNT_BATCH_SIZE, len(regular_accounts)))
        else:
            account_batch = get_next_batch_accounts(server, regular_accounts)
        
        # লাইক পাঠান
        if account_batch:
            Logger.info(f"Sending likes using {len(account_batch)} accounts")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(send_likes_with_accounts(
                    target_uid, server, like_api_url, account_batch
                ))
            finally:
                loop.close()
        
        # After likes
        after_info = check_profile(encrypted_profile, server, visit_accounts[0])
        
        after_likes = before_likes
        nickname = "N/A"
        actual_uid = int(target_uid)
        
        if after_info and hasattr(after_info, 'AccountInfo'):
            after_likes = int(after_info.AccountInfo.Likes)
            actual_uid = int(after_info.AccountInfo.UID)
            nickname = str(after_info.AccountInfo.PlayerNickname) if after_info.AccountInfo.PlayerNickname else "N/A"
        
        increment = after_likes - before_likes
        
        response = {
            "success": True,
            "LikesGiven": increment,
            "LikesAfter": after_likes,
            "LikesBefore": before_likes,
            "PlayerNickname": nickname,
            "UID": actual_uid,
            "Server": server,
            "AccountsUsed": len(account_batch),
            "Message": f"Successfully added {increment} likes"
        }
        
        Logger.info(f"Success: UID {target_uid} got {increment} likes")
        return jsonify(response)
        
    except Exception as e:
        Logger.error(f"Like endpoint error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/account_info', methods=['GET'])
def account_info():
    """অ্যাকাউন্ট ইনফরমেশন এন্ডপয়েন্ট"""
    try:
        servers = ["IND", "BD", "BR", "US", "SAC", "NA"]
        info = {}
        
        for server in servers:
            regular = load_accounts(server, for_visit=False)
            visit = load_accounts(server, for_visit=True)
            info[server] = {
                "regular_accounts": len(regular),
                "visit_accounts": len(visit)
            }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_account', methods=['POST'])
def add_account():
    """নতুন অ্যাকাউন্ট যোগ করার API"""
    try:
        data = request.json
        uid = data.get('uid')
        password = data.get('password')
        server = data.get('server', 'BD').upper()
        is_visit = data.get('is_visit', False)
        
        if not uid or not password:
            return jsonify({"error": "UID and password required"}), 400
        
        filepath = get_account_file_path(server, is_visit)
        
        # বিদ্যমান অ্যাকাউন্ট লোড
        accounts = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                accounts = json.load(f)
        
        # নতুন অ্যাকাউন্ট যোগ
        accounts.append({
            "uid": uid,
            "password": password
        })
        
        # সেভ
        with open(filepath, 'w') as f:
            json.dump(accounts, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": "Account added successfully",
            "total": len(accounts)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """হেলথ চেক"""
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "system": "UID + Password Based",
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Vercel handler
def handler(event, context):
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
