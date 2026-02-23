import os
import sys
import json
import asyncio
import threading
import random
import binascii
import traceback
from flask import Flask, request, jsonify, send_file
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import aiohttp
import requests
import urllib3

# Import protobuf modules
try:
    import like_pb2
    import like_count_pb2
    import uid_generator_pb2
except ImportError as e:
    print(f"Error importing protobuf modules: {e}")
    # Create placeholder classes if import fails
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
TOKEN_BATCH_SIZE = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global State for Batch Management
current_batch_indices = {}
batch_indices_lock = threading.Lock()

# Vercel-এর জন্য লগিং সিস্টেম
class VercelLogger:
    @staticmethod
    def log(message, level="INFO"):
        """Vercel-এ লগ প্রিন্ট করার জন্য"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}", file=sys.stderr)
        sys.stderr.flush()

    @staticmethod
    def error(message):
        VercelLogger.log(message, "ERROR")
    
    @staticmethod
    def info(message):
        VercelLogger.log(message, "INFO")
    
    @staticmethod
    def debug(message):
        if os.environ.get('DEBUG', 'False').lower() == 'true':
            VercelLogger.log(message, "DEBUG")

# Vercel-এর জন্য টোকেন লোড করার ফাংশন
def load_tokens(server_name, for_visit=False):
    """Vercel এনভায়রনমেন্টের জন্য অপ্টিমাইজড টোকেন লোডার"""
    
    # ফাইল নাম নির্ধারণ
    if for_visit:
        if server_name == "IND":
            filename = "token_ind_visit.json"
        elif server_name in {"BR", "US", "SAC", "NA"}:
            filename = "token_br_visit.json"
        else:
            filename = "token_bd_visit.json"
    else:
        if server_name == "IND":
            filename = "token_ind.json"
        elif server_name in {"BR", "US", "SAC", "NA"}:
            filename = "token_br.json"
        else:
            filename = "token_bd.json"
    
    filepath = os.path.join(BASE_DIR, filename)
    
    # Vercel এনভায়রনমেন্ট ভেরিয়েবল থেকে টোকেন লোড করার চেষ্টা
    env_token_key = f"TOKENS_{server_name}{'_VISIT' if for_visit else ''}"
    env_tokens = os.environ.get(env_token_key)
    
    if env_tokens:
        try:
            tokens = json.loads(env_tokens)
            VercelLogger.info(f"Loaded {len(tokens)} tokens from environment for {server_name}")
            return tokens
        except json.JSONDecodeError as e:
            VercelLogger.error(f"Failed to parse environment tokens: {e}")
    
    # ফাইল থেকে লোড করার চেষ্টা
    try:
        if not os.path.exists(filepath):
            VercelLogger.error(f"Token file not found: {filepath}")
            return []
            
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                VercelLogger.error(f"Token file is empty: {filepath}")
                return []
                
            tokens = json.loads(content)
            
            if isinstance(tokens, list):
                # Validate token format
                valid_tokens = []
                for t in tokens:
                    if isinstance(t, dict) and "token" in t and t["token"]:
                        valid_tokens.append(t)
                    elif isinstance(t, dict) and "uid" in t and "token" in t:
                        valid_tokens.append(t)
                
                VercelLogger.info(f"Loaded {len(valid_tokens)} valid tokens from {filename}")
                return valid_tokens
            else:
                VercelLogger.error(f"Invalid token format in {filename}")
                return []
                
    except FileNotFoundError:
        VercelLogger.error(f"Token file not found: {filename}")
        return []
    except json.JSONDecodeError as e:
        VercelLogger.error(f"Invalid JSON in {filename}: {e}")
        return []
    except Exception as e:
        VercelLogger.error(f"Unexpected error loading tokens: {e}")
        return []

def get_next_batch_tokens(server_name, all_tokens):
    """রোটেটিং ব্যাচ সিলেকশন"""
    if not all_tokens:
        return []
    
    total_tokens = len(all_tokens)
    
    if total_tokens <= TOKEN_BATCH_SIZE:
        return all_tokens
    
    with batch_indices_lock:
        if server_name not in current_batch_indices:
            current_batch_indices[server_name] = 0
        
        current_index = current_batch_indices[server_name]
        start_index = current_index
        end_index = start_index + TOKEN_BATCH_SIZE
        
        if end_index > total_tokens:
            remaining = end_index - total_tokens
            batch_tokens = all_tokens[start_index:total_tokens] + all_tokens[0:remaining]
        else:
            batch_tokens = all_tokens[start_index:end_index]
        
        next_index = (current_index + TOKEN_BATCH_SIZE) % total_tokens
        current_batch_indices[server_name] = next_index
        
        return batch_tokens

def get_random_batch_tokens(server_name, all_tokens):
    """র্যান্ডম ব্যাচ সিলেকশন"""
    if not all_tokens:
        return []
    
    total_tokens = len(all_tokens)
    
    if total_tokens <= TOKEN_BATCH_SIZE:
        return all_tokens.copy()
    
    return random.sample(all_tokens, TOKEN_BATCH_SIZE)

def encrypt_message(plaintext):
    """মেসেজ এনক্রিপ্ট করার ফাংশন"""
    try:
        key = b'Yg&tc%DEuh6%Zc^8'
        iv = b'6oyZDr22E3ychjM%'
        cipher = AES.new(key, AES.MODE_CBC, iv)
        padded_message = pad(plaintext, AES.block_size)
        encrypted_message = cipher.encrypt(padded_message)
        return binascii.hexlify(encrypted_message).decode('utf-8')
    except Exception as e:
        VercelLogger.error(f"Encryption error: {e}")
        raise

def create_protobuf_message(user_id, region):
    """লাইক প্রোটোবাফ মেসেজ তৈরি"""
    try:
        message = like_pb2.like()
        message.uid = int(user_id)
        message.region = region
        return message.SerializeToString()
    except Exception as e:
        VercelLogger.error(f"Protobuf creation error: {e}")
        raise

def create_protobuf_for_profile_check(uid):
    """প্রোফাইল চেকের জন্য প্রোটোবাফ তৈরি"""
    try:
        message = uid_generator_pb2.uid_generator()
        message.krishna_ = int(uid)
        message.teamXdarks = 1
        return message.SerializeToString()
    except Exception as e:
        VercelLogger.error(f"Profile check protobuf error: {e}")
        raise

def enc_profile_check_payload(uid):
    """প্রোফাইল চেকের পেলোড এনক্রিপ্ট"""
    protobuf_data = create_protobuf_for_profile_check(uid)
    encrypted_uid = encrypt_message(protobuf_data)
    return encrypted_uid

async def send_single_like_request(encrypted_like_payload, token_dict, url):
    """একটি টোকেন দিয়ে লাইক রিকোয়েস্ট পাঠান"""
    try:
        edata = bytes.fromhex(encrypted_like_payload)
        token_value = token_dict.get("token", "")
        
        if not token_value:
            VercelLogger.error("Empty token in request")
            return 999
        
        headers = {
            'User-Agent': "Dalvik/2.1.0 (Linux; U; Android 9; ASUS_Z01QD Build/PI)",
            'Connection': "Keep-Alive",
            'Accept-Encoding': "gzip",
            'Authorization': f"Bearer {token_value}",
            'Content-Type': "application/x-www-form-urlencoded",
            'Expect': "100-continue",
            'X-Unity-Version': "2018.4.11f1",
            'X-GA': "v1 1",
            'ReleaseVersion': "OB52"
        }
        
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=edata, headers=headers, ssl=False) as response:
                status = response.status
                if status != 200:
                    VercelLogger.error(f"Like failed: {status}")
                return status
                
    except asyncio.TimeoutError:
        VercelLogger.error("Like request timeout")
        return 998
    except Exception as e:
        VercelLogger.error(f"Like request exception: {e}")
        return 997

async def send_likes_with_token_batch(uid, server_region, like_api_url, token_batch):
    """ব্যাচ টোকেন দিয়ে লাইক পাঠান"""
    if not token_batch:
        VercelLogger.error("Empty token batch")
        return []
    
    try:
        like_protobuf = create_protobuf_message(uid, server_region)
        encrypted_payload = encrypt_message(like_protobuf)
        
        tasks = []
        for token_dict in token_batch:
            tasks.append(send_single_like_request(encrypted_payload, token_dict, like_api_url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if isinstance(r, int) and r == 200)
        VercelLogger.info(f"Batch results: {successful}/{len(token_batch)} successful")
        
        return results
        
    except Exception as e:
        VercelLogger.error(f"Batch send error: {e}")
        return []

def make_profile_check_request(encrypted_payload, server_name, token_dict):
    """প্রোফাইল চেক রিকোয়েস্ট পাঠান"""
    try:
        token_value = token_dict.get("token", "")
        if not token_value:
            VercelLogger.error("Empty token in profile check")
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
            'Connection': "Keep-Alive",
            'Accept-Encoding': "gzip",
            'Authorization': f"Bearer {token_value}",
            'Content-Type': "application/x-www-form-urlencoded",
            'X-Unity-Version': "2018.4.11f1",
            'ReleaseVersion': "OB52"
        }
        
        response = requests.post(url, data=edata, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        return decode_protobuf_profile_info(response.content)
        
    except requests.Timeout:
        VercelLogger.error("Profile check timeout")
        return None
    except requests.RequestException as e:
        VercelLogger.error(f"Profile check request error: {e}")
        return None
    except Exception as e:
        VercelLogger.error(f"Profile check error: {e}")
        return None

def decode_protobuf_profile_info(binary_data):
    """প্রোফাইল ডেটা ডিকোড করুন"""
    try:
        info = like_count_pb2.Info()
        info.ParseFromString(binary_data)
        return info
    except Exception as e:
        VercelLogger.error(f"Protobuf decode error: {e}")
        return None

# Flask অ্যাপ তৈরি
app = Flask(__name__)

@app.route('/')
def home():
    """হোম পেজ - HTML ফাইল রিটার্ন করে"""
    try:
        html_path = os.path.join(BASE_DIR, 'index.html')
        if os.path.exists(html_path):
            return send_file(html_path)
        else:
            return jsonify({
                "status": "running",
                "message": "Free Fire Like Booster API",
                "endpoints": {
                    "like": "/like?uid=UID&server_name=SERVER",
                    "token_info": "/token_info"
                }
            })
    except Exception as e:
        VercelLogger.error(f"Home page error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/like', methods=['GET'])
def handle_requests():
    """লাইক রিকোয়েস্ট হ্যান্ডেল করুন"""
    try:
        uid_param = request.args.get("uid")
        server_name_param = request.args.get("server_name", "").upper()
        use_random = request.args.get("random", "false").lower() == "true"
        
        # Input validation
        if not uid_param or not server_name_param:
            return jsonify({"error": "UID and server_name are required"}), 400
        
        if not uid_param.isdigit():
            return jsonify({"error": "UID must contain only digits"}), 400
        
        # Load tokens
        visit_tokens = load_tokens(server_name_param, for_visit=True)
        if not visit_tokens:
            return jsonify({"error": f"No visit tokens for server {server_name_param}"}), 500
        
        all_tokens = load_tokens(server_name_param, for_visit=False)
        if not all_tokens:
            return jsonify({"error": f"No regular tokens for server {server_name_param}"}), 500
        
        VercelLogger.info(f"Processing UID: {uid_param}, Server: {server_name_param}")
        
        # Get token batch for likes
        if use_random:
            token_batch = get_random_batch_tokens(server_name_param, all_tokens)
        else:
            token_batch = get_next_batch_tokens(server_name_param, all_tokens)
        
        # Profile check payload
        encrypted_profile = enc_profile_check_payload(uid_param)
        
        # Before likes count
        before_info = make_profile_check_request(encrypted_profile, server_name_param, visit_tokens[0])
        before_likes = int(before_info.AccountInfo.Likes) if before_info and hasattr(before_info, 'AccountInfo') else 0
        
        # Send likes
        like_api_url = f"https://client.{'ind' if server_name_param == 'IND' else 'us' if server_name_param in {'BR', 'US', 'SAC', 'NA'} else 'bp'}.freefiremobile.com/LikeProfile"
        
        if token_batch:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(send_likes_with_token_batch(
                    uid_param, server_name_param, like_api_url, token_batch
                ))
            finally:
                loop.close()
        
        # After likes count
        after_info = make_profile_check_request(encrypted_profile, server_name_param, visit_tokens[0])
        
        after_likes = before_likes
        nickname = "N/A"
        actual_uid = int(uid_param)
        
        if after_info and hasattr(after_info, 'AccountInfo'):
            after_likes = int(after_info.AccountInfo.Likes)
            actual_uid = int(after_info.AccountInfo.UID)
            nickname = str(after_info.AccountInfo.PlayerNickname) if after_info.AccountInfo.PlayerNickname else "N/A"
        
        increment = after_likes - before_likes
        
        response_data = {
            "LikesGivenByAPI": increment,
            "LikesafterCommand": after_likes,
            "LikesbeforeCommand": before_likes,
            "PlayerNickname": nickname,
            "UID": actual_uid,
            "status": 1 if increment > 0 else (2 if increment == 0 else 3),
            "Note": f"Used batch of {len(token_batch)} tokens"
        }
        
        VercelLogger.info(f"Success: UID {uid_param} got {increment} likes")
        return jsonify(response_data)
        
    except Exception as e:
        VercelLogger.error(f"Like endpoint error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/token_info', methods=['GET'])
def token_info():
    """টোকেন ইনফরমেশন এন্ডপয়েন্ট"""
    try:
        servers = ["IND", "BD", "BR", "US", "SAC", "NA"]
        info = {}
        
        for server in servers:
            regular = load_tokens(server, for_visit=False)
            visit = load_tokens(server, for_visit=True)
            info[server] = {
                "regular_tokens": len(regular),
                "visit_tokens": len(visit)
            }
        
        return jsonify(info)
        
    except Exception as e:
        VercelLogger.error(f"Token info error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """হেলথ চেক এন্ডপয়েন্ট"""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "environment": os.environ.get('VERCEL_ENV', 'development')
    })

@app.errorhandler(404)
def not_found(error):
    """404 এরর হ্যান্ডলার"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """500 এরর হ্যান্ডলার"""
    VercelLogger.error(f"500 error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Vercel-এর জন্য হ্যান্ডলার
def handler(event, context):
    """Vercel Serverless ফাংশন হ্যান্ডলার"""
    return app

# লোকাল ডেভেলপমেন্টের জন্য
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)